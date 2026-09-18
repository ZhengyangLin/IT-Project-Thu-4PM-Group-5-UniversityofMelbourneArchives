from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from functools import partial
import json
import logging
import multiprocessing
from pathlib import Path
from threading import Lock
import time
from typing import Any

from ....unimelb_result_codec import (
    OCR_RESULT_FIELDS,
    evidence_regions_bbox,
    extract_ocr_result,
    note_values,
    split_reason_and_notes,
    tokens_bbox,
)
from ...schemas import (
    UnimelbImageSelectItem,
    UnimelbImageSelectResponse,
    UnimelbTaskCreatedResponse,
)
from ..Unimelb_service import (
    UnimelbService,
    UnimelbTaskRunningError,
)
from ...workers.Unimelb_ocr_worker import (
    OcrImageJob,
    OcrImageOutcome,
    initialize_ocr_worker,
    process_ocr_image,
    write_batch_summary as write_worker_batch_summary,
)


PROJECT_DIR = Path(__file__).resolve().parents[4]
PLAN_SAMPLE_DIR = PROJECT_DIR / "plan_sample"
OCR_FOLDERS = ("YFA_cabana", "YFA_state_government")
IMAGE_EXTENSIONS = {
    ".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".webp",
}
log = logging.getLogger(__name__)
UNIMELB_OCR_ENGINES = ("paddle",)
UNIMELB_OCR_OUTPUT_ROOT = PROJECT_DIR / "output"
UNIMELB_VERBOSE_PROMPT = True

UNIMELB_CONFIG_PATH = PROJECT_DIR / "config.yaml"
UNIMELB_LOG_DIR = UNIMELB_OCR_OUTPUT_ROOT / "logs"
IMAGE_STATUS_FILE = UNIMELB_OCR_OUTPUT_ROOT / "unimelb_image_status.json"
IMAGE_STATUS_PENDING = 0
IMAGE_STATUS_RUNNING = 1
IMAGE_STATUS_COMPLETED = 2
IMAGE_STATUS_FAILED = 3
UNIMELB_SHARED_TASK_UUID = "00000000-0000-4000-8000-000000000001"
RESTART_INTERRUPTED_MESSAGE = "Service restarted before the OCR task completed"
LEGACY_IMAGE_STATUSES = {
    "not_started": IMAGE_STATUS_PENDING,
    "queued": IMAGE_STATUS_PENDING,
    "running": IMAGE_STATUS_RUNNING,
    "success": IMAGE_STATUS_COMPLETED,
    "failed": IMAGE_STATUS_FAILED,
}


class UnimelbServiceImpl(UnimelbService):
    def __init__(self, status_file: Path | None = None) -> None:
        self._tasks: dict[str, dict[str, Any]] = {}
        self._tasks_lock = Lock()
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._rerun_image_owners_lock = Lock()
        self._rerun_image_owners: dict[str, object] = {}
        self._status_file = status_file or IMAGE_STATUS_FILE
        self._image_statuses, statuses_migrated = (
            self._load_image_statuses()
        )

        self._ocr_executor: ProcessPoolExecutor | None = None
        self._ocr_executor_lock = Lock()
        self._closing = False


        with self._tasks_lock:
            interrupted = self._mark_interrupted_running_images_failed_locked()
            if statuses_migrated or interrupted:
                self._persist_image_statuses_locked()
            self._restore_shared_task_locked(self._collect_images())


    def _ensure_ocr_executor_locked(self) -> ProcessPoolExecutor:
        if self._closing:
            raise RuntimeError("OCR service is shutting down")
        if self._ocr_executor is None:
            self._ocr_executor = ProcessPoolExecutor(
                max_workers=2,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_ocr_worker,
                initargs=(
                    str(UNIMELB_CONFIG_PATH),
                    tuple(UNIMELB_OCR_ENGINES),
                    UNIMELB_VERBOSE_PROMPT,
                    str(UNIMELB_LOG_DIR),
                ),
            )
            log.info("unimelb OCR executor created: max_workers=2")
        return self._ocr_executor

    def _get_ocr_executor(self) -> ProcessPoolExecutor:
        with self._ocr_executor_lock:
            return self._ensure_ocr_executor_locked()

    def _discard_broken_executor(self, executor: ProcessPoolExecutor,) -> None:
        with self._ocr_executor_lock:
            if self._ocr_executor is not executor:
                return
            self._ocr_executor = None
        executor.shutdown(wait=False, cancel_futures=True)


    async def get_image_select(self) -> UnimelbImageSelectResponse:
        self._validate_folders()
        paths = await asyncio.to_thread(self._collect_images)
        counts = {name.upper(): 0 for name in OCR_FOLDERS}
        status_counts = {"0": 0, "1": 0, "2": 0, "3": 0}
        with self._tasks_lock:
            statuses = deepcopy(self._image_statuses)
        images: list[UnimelbImageSelectItem] = []
        for path in paths:
            relative = path.relative_to(PLAN_SAMPLE_DIR)
            folder = relative.parts[0].upper()
            node_id = path.parent.name
            counts[folder] += 1
            image_key = f"{node_id}/{path.name}"
            image_status = statuses.get(image_key, {})
            status = self._normalize_image_status(
                image_status.get("status", IMAGE_STATUS_PENDING)
            )
            status_counts[str(status)] += 1
            images.append(UnimelbImageSelectItem(
                path=f"{folder}/{node_id}/{path.name}",
                image_key=image_key,
                folder=folder,
                node_id=node_id,
                image_name=path.name,
                image_url="",
                status=status,
                task_uuid=image_status.get("task_uuid"),
                error=image_status.get("message"),
                message=image_status.get("message"),
                updated_at=image_status.get("updated_at"),
                manual_reviewed=self._normalize_manual_reviewed(
                    image_status.get("manual_reviewed")
                ),
                result=self._format_ocr_result_for_response(
                    image_status.get("result")
                ),
            ))
        return UnimelbImageSelectResponse(
            task_uuid=UNIMELB_SHARED_TASK_UUID,
            folders=counts,
            total=len(images),
            status_counts=status_counts,
            images=images,
        )

    def resolve_image(self, folder: str, node_id: str, image_name: str,) -> Path | None:
        folder_map = {name.upper(): name for name in OCR_FOLDERS}
        actual_folder = folder_map.get(folder.upper())
        if actual_folder is None:
            return None
        if Path(node_id).name != node_id or Path(image_name).name != image_name:
            return None

        allowed_root = (PLAN_SAMPLE_DIR / actual_folder).resolve()
        candidate = (allowed_root / node_id / image_name).resolve()
        try:
            candidate.relative_to(allowed_root)
        except ValueError:
            return None
        if (not candidate.is_file() or candidate.suffix.lower() not in IMAGE_EXTENSIONS):
            return None
        return candidate

    async def create_ocr_task(self, force: bool) -> UnimelbTaskCreatedResponse:
        self._validate_folders()
        images = await asyncio.to_thread(self._collect_images)
        if not images:
            raise FileNotFoundError("No images available for OCR were found")

        task_uuid = UNIMELB_SHARED_TASK_UUID
        output_dir = UNIMELB_OCR_OUTPUT_ROOT / "tasks" / task_uuid
        now = self._now_ms()
        image_keys = [self._image_key(image) for image in images]
        all_image_keys = set(image_keys)
        reservation_token = object()
        with self._rerun_image_owners_lock:
            all_owned_image_keys = list(self._rerun_image_owners)
            owned_conflicting_image_keys = [
                image_key
                for image_key in image_keys
                if image_key in self._rerun_image_owners
            ]
            if force and all_owned_image_keys:
                raise UnimelbTaskRunningError(
                    "An OCR task is running or images are reserved by another "
                    "task: " + ", ".join(all_owned_image_keys)
                )
            claimed_image_keys = (
                all_image_keys
                if force else all_image_keys.difference(
                    owned_conflicting_image_keys
                )
            )
            for image_key in claimed_image_keys:
                self._rerun_image_owners[image_key] = reservation_token

        run_images: list[Path] = []
        not_started_response: UnimelbTaskCreatedResponse | None = None
        try:
            with self._tasks_lock:
                existing = self._restore_shared_task_locked(images)
                running_image_keys = {
                    item["image_key"]
                    for item in (existing or {}).get("running", [])
                    if isinstance(item, dict)
                    and isinstance(item.get("image_key"), str)
                }
                if force and running_image_keys:
                    raise UnimelbTaskRunningError(
                        "An OCR task is already running and cannot be forcibly "
                        "interrupted; wait for the task to finish"
                    )

                reset_all = force or (
                    existing is None
                    and claimed_image_keys == all_image_keys
                )
                previous_statuses = deepcopy(self._image_statuses)

                if reset_all:
                    self._image_statuses = {}
                    self._tasks.pop(task_uuid, None)

                for image in images:
                    image_key = self._image_key(image)
                    if image_key not in claimed_image_keys:
                        continue
                    state = self._image_statuses.get(image_key)
                    belongs_to_task = (
                        state is not None
                        and state.get("task_uuid") == task_uuid
                    )
                    status = (
                        self._normalize_image_status(state.get("status"))
                        if state is not None else IMAGE_STATUS_PENDING
                    )
                    if status == IMAGE_STATUS_RUNNING:
                        continue
                    if (reset_all or not belongs_to_task or status == IMAGE_STATUS_PENDING):
                        self._image_statuses[image_key] = (
                            self._new_image_state(
                                image,
                                task_uuid,
                                IMAGE_STATUS_PENDING,
                                now,
                            )
                        )
                        run_images.append(image)

                if not run_images:
                    restored = self._restore_shared_task_locked(images)
                    not_started_response = UnimelbTaskCreatedResponse(
                        task_uuid=task_uuid,
                        status=(
                            restored["status"]
                            if restored is not None
                            else IMAGE_STATUS_PENDING
                        ),
                        total=len(images),
                        started=False,
                    )
                else:
                    try:
                        self._persist_image_statuses_locked()
                    except Exception:
                        self._image_statuses = previous_statuses
                        self._restore_shared_task_locked(images)
                        raise
                    self._restore_shared_task_locked(images)
        except BaseException:
            self._release_rerun_image_owners(
                reservation_token, claimed_image_keys
            )
            raise

        run_image_keys = {
            self._image_key(image) for image in run_images
        }
        self._release_rerun_image_owners(
            reservation_token,
            claimed_image_keys.difference(run_image_keys),
        )
        if not_started_response is not None:
            return not_started_response

        try:
            task = asyncio.create_task(
                self._run_ocr_task(
                    task_uuid,
                    run_images,
                    output_dir,
                    write_batch_summary=(run_image_keys == all_image_keys),
                    reservation_token=reservation_token,
                )
            )
        except Exception:
            try:
                self._fail_unfinished_images(
                    task_uuid,
                    "Failed to start OCR background task",
                    image_keys=run_image_keys,
                    reservation_token=reservation_token,
                )
            finally:
                self._release_rerun_image_owners(
                    reservation_token, run_image_keys
                )
            raise
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return UnimelbTaskCreatedResponse(
            task_uuid=task_uuid,
            status=IMAGE_STATUS_RUNNING,
            total=len(images),
            started=True,
        )

    def _collect_images(self) -> list[Path]:
        return [
            path
            for name in OCR_FOLDERS
            for path in sorted((PLAN_SAMPLE_DIR / name).rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

    def _restore_shared_task_locked(self, images: list[Path],) -> dict[str, Any] | None:
        restored: list[tuple[Path, dict[str, Any] | None, int]] = []
        has_task_record = False
        for image in images:
            state = self._image_statuses.get(self._image_key(image))
            if state and state.get("task_uuid") == UNIMELB_SHARED_TASK_UUID:
                has_task_record = True
                status = self._normalize_image_status(state.get("status"))
            else:
                state = None
                status = IMAGE_STATUS_PENDING
            restored.append((
                image,
                state,
                status,
            ))
        if not images or not has_task_record:
            self._tasks.pop(UNIMELB_SHARED_TASK_UUID, None)
            return None

        statuses = [status for _, _, status in restored]
        if IMAGE_STATUS_RUNNING in statuses:
            task_status = IMAGE_STATUS_RUNNING
        elif IMAGE_STATUS_PENDING in statuses:
            task_status = IMAGE_STATUS_PENDING
        elif all(status == IMAGE_STATUS_COMPLETED for status in statuses):
            task_status = IMAGE_STATUS_COMPLETED
        else:
            task_status = IMAGE_STATUS_FAILED

        successful: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        running: list[dict[str, Any]] = []
        updated_times: list[int] = []
        current_image: str | None = None
        current_node_id: str | None = None
        for image, state, status in restored:
            if state is None:
                continue
            item = {
                "image_key": self._image_key(image),
                "image": self._display_image_path(image),
                "node_id": image.parent.name,
                "error": state.get("message"),
            }
            if status == IMAGE_STATUS_COMPLETED:
                item["error"] = None
                successful.append(item)
            elif status == IMAGE_STATUS_FAILED:
                failed.append(item)
            elif status == IMAGE_STATUS_RUNNING:
                running.append(item)
                if current_image is None:
                    current_image = item["image"]
                    current_node_id = item["node_id"]
            updated_at = state.get("updated_at")
            if isinstance(updated_at, int):
                updated_times.append(updated_at)

        previous = self._tasks.get(UNIMELB_SHARED_TASK_UUID)
        timestamp = max(updated_times) if updated_times else (
            previous.get("created_at") if previous else self._now_ms()
        )
        created_at = (
            previous.get("created_at")
            if previous else min(updated_times) if updated_times else timestamp
        )
        started_at = (
            previous.get("started_at")
            if previous else min(updated_times) if updated_times else timestamp
        )
        candidate = {
            "task_uuid": UNIMELB_SHARED_TASK_UUID,
            "status": task_status,
            "total": len(images),
            "completed": len(successful) + len(failed),
            "success_count": len(successful),
            "failed_count": len(failed),
            "successful": successful,
            "failed": failed,
            "running": running,
            "output_dir": str(
                (UNIMELB_OCR_OUTPUT_ROOT / "tasks" / UNIMELB_SHARED_TASK_UUID)
                .relative_to(PROJECT_DIR)
            ),
            "error": (
                failed[0]["error"]
                if task_status == IMAGE_STATUS_FAILED and failed else None
            ),
            "current_image": current_image,
            "current_node_id": current_node_id,
            "created_at": created_at,
            "started_at": started_at,
            "finished_at": (
                timestamp
                if task_status in {
                    IMAGE_STATUS_COMPLETED, IMAGE_STATUS_FAILED,
                }
                else None
            ),
        }
        self._tasks[UNIMELB_SHARED_TASK_UUID] = candidate
        return candidate

    @staticmethod
    def _display_image_path(image: Path) -> str:
        try:
            return str(image.relative_to(PROJECT_DIR))
        except ValueError:
            return str(image)

    @staticmethod
    def _validate_folders() -> None:
        missing = [
            name for name in OCR_FOLDERS
            if not (PLAN_SAMPLE_DIR / name).is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                "OCR image directories not found: " + ", ".join(missing)
            )

    async def _run_ocr_task(self,task_uuid: str,images: list[Path],output_dir: Path,*,reservation_token: object,write_batch_summary: bool = True,) -> None:
        image_keys = {self._image_key(image) for image in images}
        try:
                summary_rows: list[dict[str, Any]] = []
                for image in images:
                    node_id = image.parent.name
                    image_key = self._image_key(image)
                    item_output_dir = self._item_output_dir(
                        output_dir, node_id, image.name
                    )
                    self._set_image_status(
                        image_key, task_uuid, IMAGE_STATUS_RUNNING
                    )
                    job = OcrImageJob(
                        image_path=str(image),
                        node_id=node_id,
                        item_output_dir=str(item_output_dir),
                    )

                    worker_task = asyncio.create_task(
                        self._submit_ocr_image(job)
                    )
                    outcome, cancellation = await self._await_shielded_task(
                        worker_task
                    )
                    if outcome.summary_row is not None:
                        summary_rows.append(outcome.summary_row)
                    self._set_serialized_image_result(
                        image_key,
                        task_uuid,
                        (
                            IMAGE_STATUS_COMPLETED
                            if outcome.ok else IMAGE_STATUS_FAILED
                        ),
                        outcome.result,
                        message=outcome.error,
                        reservation_token=reservation_token,
                    )
                    if cancellation is not None:
                        raise cancellation

                if summary_rows and write_batch_summary:
                    summary_task = asyncio.create_task(
                        self._submit_batch_summary(output_dir, summary_rows)
                    )
                    _, cancellation = await self._await_shielded_task(
                        summary_task
                    )
                    if cancellation is not None:
                        raise cancellation
        except asyncio.CancelledError:
            self._fail_unfinished_images(
                task_uuid,
                "OCR task was cancelled",
                image_keys=image_keys,
                reservation_token=reservation_token,
            )
            raise
        except BrokenProcessPool as exc:
            error = f"OCR worker exited unexpectedly: {exc}"
            log.exception("OCR worker exited unexpectedly: task_uuid=%s", task_uuid)
            self._fail_unfinished_images(
                task_uuid,
                error,
                image_keys=image_keys,
                reservation_token=reservation_token,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log.exception("OCR batch task failed: task_uuid=%s", task_uuid)
            self._fail_unfinished_images(
                task_uuid,
                error,
                image_keys=image_keys,
                reservation_token=reservation_token,
            )
        finally:
            self._release_rerun_image_owners(
                reservation_token, image_keys
            )

    async def _submit_ocr_image(self, job: OcrImageJob,) -> OcrImageOutcome:
        executor = self._get_ocr_executor()
        try:
            return await asyncio.get_running_loop().run_in_executor(
                executor, process_ocr_image, job
            )
        except BrokenProcessPool:
            self._discard_broken_executor(executor)
            raise

    async def _submit_batch_summary(self, output_dir: Path, rows: list[dict[str, Any]],) -> str:
        executor = self._get_ocr_executor()
        try:
            return await asyncio.get_running_loop().run_in_executor(
                executor,
                write_worker_batch_summary,
                str(output_dir),
                rows,
            )
        except BrokenProcessPool:
            self._discard_broken_executor(executor)
            raise

    @staticmethod
    async def _await_shielded_task(worker_task: asyncio.Task[Any],) -> tuple[Any, asyncio.CancelledError | None]:
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                result = await asyncio.shield(worker_task)
                return result, cancellation
            except asyncio.CancelledError as exc:
                if worker_task.cancelled():
                    raise
                if cancellation is None:
                    cancellation = exc

    @staticmethod
    def _item_output_dir(output_dir: Path, node_id: str, image_name: str,) -> Path:
        return output_dir / "items" / node_id / image_name

    @staticmethod
    def _new_image_state(image: Path, task_uuid: str, status: int, updated_at: int,) -> dict[str, Any]:
        return {
            "node_id": image.parent.name,
            "image_name": image.name,
            "status": status,
            "task_uuid": task_uuid,
            "message": None,
            "updated_at": updated_at,
            "manual_reviewed": 0,
            "result": {},
        }

    def _set_image_status(self, image_key: str, task_uuid: str, status: int,error: str | None = None,) -> None:
        node_id, image_name = image_key.split("/", 1)
        with self._tasks_lock:
            self._image_statuses[image_key] = {
                "node_id": node_id,
                "image_name": image_name,
                "status": status,
                "task_uuid": task_uuid,
                "message": error,
                "manual_reviewed": 0,
                "result": {},
                "updated_at": self._now_ms(),
            }
            self._persist_image_statuses_locked()

    def _set_image_result(self,image_key: str,task_uuid: str,status: int,record: Any | None,message: str | None = None,
        next_image_key: str | None = None,reservation_token: object | None = None,release_owner: bool = True,) -> None:
        self._set_serialized_image_result(
            image_key,
            task_uuid,
            status,
            self._extract_ocr_result(record),
            message=message,
            next_image_key=next_image_key,
            reservation_token=reservation_token,
            release_owner=release_owner,
        )

    def _set_serialized_image_result(self,image_key: str,task_uuid: str,status: int,result: dict[str, Any],
        message: str | None = None,next_image_key: str | None = None,reservation_token: object | None = None,release_owner: bool = True,) -> None:
        node_id, image_name = image_key.split("/", 1)
        now = self._now_ms()
        with self._tasks_lock:
            self._image_statuses[image_key] = {
                "node_id": node_id,
                "image_name": image_name,
                "status": status,
                "task_uuid": task_uuid,
                "message": message,
                "updated_at": now,
                "manual_reviewed": 0,
                "result": deepcopy(result),
            }
            if next_image_key is not None:
                next_node_id, next_image_name = next_image_key.split("/", 1)
                self._image_statuses[next_image_key] = {
                    "node_id": next_node_id,
                    "image_name": next_image_name,
                    "status": IMAGE_STATUS_RUNNING,
                    "task_uuid": task_uuid,
                    "message": None,
                    "updated_at": now,
                    "manual_reviewed": 0,
                    "result": {},
                }
            self._persist_image_statuses_locked()
        if reservation_token is not None and release_owner:
            self._release_rerun_image_owners(
                reservation_token, {image_key}
            )

    @classmethod
    def _extract_ocr_result(cls, record: Any | None) -> dict[str, Any]:
        return extract_ocr_result(record)

    @classmethod
    def _format_ocr_result_for_response(cls, result: Any,) -> dict[str, dict[str, Any]]:
        if not isinstance(result, dict) or not result:
            return {}
        nested: dict[str, dict[str, Any]] = {}
        suffixes = (
            "verbatim",
            "corrected",
            "correction_basis",
            "conf",
            "source",
            "tokens",
            "tokens_bbox",
            "reason",
            "notes",
            "review_status",
        )
        for field_name in OCR_RESULT_FIELDS:
            field_keys = (
                field_name,
                *(f"{field_name}_{suffix}" for suffix in suffixes),
            )
            if not any(key in result for key in field_keys):
                continue

            verbatim = result.get(f"{field_name}_verbatim")
            corrected = result.get(f"{field_name}_corrected")
            if field_name in result:
                value = result.get(field_name)
            else:
                value = corrected or verbatim
            tokens = result.get(f"{field_name}_tokens")
            if not isinstance(tokens, list):
                tokens = []
            tokens_bbox = result.get(f"{field_name}_tokens_bbox")
            if not isinstance(tokens_bbox, dict):
                tokens_bbox = None
            reason, notes = cls._split_reason_and_notes(
                result.get(f"{field_name}_reason"),
                result.get(f"{field_name}_notes"),
            )

            nested[field_name] = {
                "value": value,
                "verbatim": verbatim,
                "corrected": corrected,
                "correction_basis": result.get(
                    f"{field_name}_correction_basis"
                ),
                "confidence": result.get(f"{field_name}_conf"),
                "source": result.get(f"{field_name}_source"),
                "tokens": list(tokens),
                "tokens_bbox": deepcopy(tokens_bbox),
                "reason": reason,
                "notes": notes,
                "review_status": result.get(
                    f"{field_name}_review_status"
                ),
            }
        return nested

    @staticmethod
    def _note_values(*values: Any) -> list[str]:
        return note_values(*values)

    @classmethod
    def _split_reason_and_notes(
        cls, raw_reason: Any, raw_notes: Any,
    ) -> tuple[str | None, list[str]]:
        return split_reason_and_notes(raw_reason, raw_notes)

    @classmethod
    def _normalize_result_reason_keys(
        cls, result: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        normalized = dict(result)
        migrated = False
        for field_name in OCR_RESULT_FIELDS:
            reason_key = f"{field_name}_reason"
            notes_key = f"{field_name}_notes"
            if reason_key not in normalized and notes_key not in normalized:
                continue
            reason, notes = cls._split_reason_and_notes(
                normalized.get(reason_key), normalized.get(notes_key)
            )
            if (
                reason_key not in normalized
                or normalized.get(reason_key) != reason
            ):
                normalized[reason_key] = reason
                migrated = True
            if (
                notes_key not in normalized
                or normalized.get(notes_key) != notes
            ):
                normalized[notes_key] = notes
                migrated = True
        return normalized, migrated

    @staticmethod
    def _evidence_regions_bbox(signals: Any) -> dict[str, int] | None:
        return evidence_regions_bbox(signals)

    @staticmethod
    def _tokens_bbox(tokens: list[int], token_map: dict[Any, Any],) -> dict[str, int] | None:
        return tokens_bbox(tokens, token_map)

    def _fail_unfinished_images(self,task_uuid: str,error: str,*,image_keys: set[str] | None = None,reservation_token: object | None = None,) -> None:
        now = self._now_ms()
        owned_image_keys = image_keys
        if reservation_token is not None:
            with self._rerun_image_owners_lock:
                owned_image_keys = {
                    image_key
                    for image_key, owner in self._rerun_image_owners.items()
                    if owner is reservation_token
                    and (image_keys is None or image_key in image_keys)
                }
            if not owned_image_keys:
                return

        with self._tasks_lock:
            changed = False
            for image_key, state in self._image_statuses.items():
                if ((owned_image_keys is None or image_key in owned_image_keys)
                    and state.get("task_uuid") == task_uuid
                    and self._normalize_image_status(state.get("status"))
                    in {IMAGE_STATUS_PENDING, IMAGE_STATUS_RUNNING}
                ):
                    state.update(
                        status=IMAGE_STATUS_FAILED,
                        message=error,
                        updated_at=now,
                    )
                    changed = True
            if changed:
                self._persist_image_statuses_locked()

    def _release_rerun_image_owners(self, reservation_token: object, image_keys: set[str],) -> None:
        with self._rerun_image_owners_lock:
            self._release_rerun_image_owners_locked(
                reservation_token, image_keys
            )

    def _release_rerun_image_owners_locked(
        self, reservation_token: object, image_keys: set[str],
    ) -> None:
        for image_key in image_keys:
            if (self._rerun_image_owners.get(image_key) is reservation_token):
                del self._rerun_image_owners[image_key]

    def _mark_interrupted_running_images_failed_locked(self) -> bool:
        now = self._now_ms()
        changed = False
        for state in self._image_statuses.values():
            if (state.get("task_uuid") == UNIMELB_SHARED_TASK_UUID and self._normalize_image_status(state.get("status"))== IMAGE_STATUS_RUNNING):
                state.update(
                    status=IMAGE_STATUS_FAILED,
                    message=RESTART_INTERRUPTED_MESSAGE,
                    updated_at=now,
                )
                changed = True
        return changed

    @staticmethod
    def _image_key(image: Path) -> str:
        return f"{image.parent.name}/{image.name}"

    def _load_image_statuses(self,) -> tuple[dict[str, dict[str, Any]], bool]:
        if not self._status_file.is_file():
            return {}, False
        try:
            payload = json.loads(self._status_file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return {}, False
            legacy_images = payload.get("images")
            images = (
                legacy_images
                if isinstance(legacy_images, dict) else payload
            )
            normalized: dict[str, dict[str, Any]] = {}
            statuses_migrated = False
            for key, value in images.items():
                if not isinstance(key, str) or not isinstance(value, dict):
                    continue
                parts = tuple(
                    part for part in key.replace("\\", "/").split("/")
                    if part
                )
                if len(parts) < 2:
                    continue
                node_id = str(value.get("node_id") or parts[-2])
                image_name = str(value.get("image_name") or parts[-1])
                image_key = f"{node_id}/{image_name}"
                raw_result = (
                    value.get("result")
                    if isinstance(value.get("result"), dict) else {}
                )
                result, migrated = self._normalize_result_reason_keys(
                    raw_result
                )
                statuses_migrated = statuses_migrated or migrated
                raw_manual_reviewed = value.get("manual_reviewed")
                manual_reviewed = self._normalize_manual_reviewed(
                    raw_manual_reviewed
                )
                if (type(raw_manual_reviewed) is not int or raw_manual_reviewed not in {0, 1}):
                    statuses_migrated = True
                normalized[image_key] = {
                    "node_id": node_id,
                    "image_name": image_name,
                    "message": value.get("message") or value.get("error"),
                    "task_uuid": value.get("task_uuid"),
                    "updated_at": value.get("updated_at"),
                    "manual_reviewed": manual_reviewed,
                    "result": result,
                    "status": self._normalize_image_status(
                        value.get("status")
                    ),
                }
            return normalized, statuses_migrated
        except (OSError, json.JSONDecodeError, AttributeError):
            log.exception("Failed to read image status cache: %s", self._status_file)
            return {}, False

    @staticmethod
    def _normalize_image_status(value: Any) -> int:
        if isinstance(value, str):
            if value in LEGACY_IMAGE_STATUSES:
                return LEGACY_IMAGE_STATUSES[value]
            if value.isdigit():
                value = int(value)
        if value in {
            IMAGE_STATUS_PENDING,
            IMAGE_STATUS_RUNNING,
            IMAGE_STATUS_COMPLETED,
            IMAGE_STATUS_FAILED,
        }:
            return int(value)
        return IMAGE_STATUS_PENDING

    @staticmethod
    def _normalize_manual_reviewed(value: Any) -> int:
        if isinstance(value, str):
            value = value.strip()
            if value in {"0", "1"}:
                value = int(value)
        return 1 if value == 1 else 0

    def _persist_image_statuses_locked(self) -> None:
        temp_file = self._status_file.with_suffix(
            self._status_file.suffix + ".tmp"
        )
        try:
            self._status_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file.write_text(
                json.dumps(
                    self._image_statuses, ensure_ascii=False, indent=2
                ),
                encoding="utf-8",
            )
            temp_file.replace(self._status_file)
        except OSError:
            log.exception("Failed to write image status cache: %s", self._status_file)
            raise

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)
