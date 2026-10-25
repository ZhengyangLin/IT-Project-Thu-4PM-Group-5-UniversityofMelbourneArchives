import json
from pathlib import Path

import cv2
import numpy as np

from uma.models import FieldValue, SheetRecord
from uma.pipeline import record_to_row, write_outputs
from uma.triage import build_manifest, infer_series, summarize
from uma.unimelb_result_codec import extract_ocr_result


def make_record() -> SheetRecord:
    return SheetRecord(
        node_id="460674",
        path="sample.jpg",
        series="YFA-StateGov",
        fields={
            "project_name": FieldValue(
                name="project_name",
                verbatim="STATE GOVERNMENT OFFICES",
                corrected="STATE GOVERNMENT OFFICES",
                confidence=0.95,
                source="rule",
                status="locked",
                tokens=[1, 2],
            )
        },
        token_map={
            "1": {"bbox": [10, 20, 100, 40]},
            "2": {"bbox": [105, 20, 220, 40]},
        },
    )


def write_sample_image(path: Path) -> None:
    image = np.full((240, 320), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "STATE GOVERNMENT",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        0,
        2,
        cv2.LINE_AA,
    )
    assert cv2.imwrite(str(path), image)


def test_record_can_flow_to_csv_row_and_codec_result():
    record = make_record()

    row = record_to_row(record)
    result = extract_ocr_result(record)

    assert row["project_name"] == "STATE GOVERNMENT OFFICES"
    assert row["project_name_status"] == "locked"
    assert result["project_name"] == "STATE GOVERNMENT OFFICES"
    assert result["project_name_conf"] == 0.95
    assert result["project_name_tokens_bbox"] == {
        "x0": 10,
        "y0": 20,
        "x1": 220,
        "y1": 40,
    }


def test_record_can_be_written_to_sidecar_and_summary_files(tmp_path: Path):
    paths = write_outputs([make_record()], str(tmp_path))

    csv_path = Path(paths["csv"])
    sidecar_path = Path(paths["sidecar_dir"]) / "460674.json"

    assert csv_path.exists()
    assert sidecar_path.exists()
    assert "STATE GOVERNMENT OFFICES" in csv_path.read_text(encoding="utf-8-sig")

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["node_id"] == "460674"
    assert sidecar["fields"]["project_name"]["status"] == "locked"


def test_drawing_geometry_can_be_used_by_the_next_component():
    from uma.models import iou, union

    first_box = (0, 0, 10, 10)
    second_box = (5, 5, 15, 15)

    assert iou(first_box, second_box) == 25 / 175
    assert union([first_box, second_box]) == (0, 0, 15, 15)


def test_sheet_record_holds_extracted_metadata():
    field = FieldValue(
        name="project_name",
        verbatim="STATE GOVERNMENT OFFICES",
        corrected="STATE GOVERNMENT OFFICES",
        confidence=0.95,
        source="rule",
        status="locked",
    )
    record = SheetRecord(
        node_id="460674",
        path="sample.jpg",
        fields={"project_name": field},
    )

    assert record.fields["project_name"].corrected == (
        "STATE GOVERNMENT OFFICES"
    )
    assert record.fields["project_name"].status == "locked"


def test_metadata_record_can_be_saved_and_read_by_next_component(tmp_path: Path):
    record = SheetRecord(
        node_id="460674",
        path="sample.jpg",
        fields={
            "project_name": FieldValue(
                name="project_name",
                verbatim="STATE GOVERNMENT OFFICES",
                corrected="STATE GOVERNMENT OFFICES",
                confidence=0.95,
                source="rule",
                status="locked",
            )
        },
    )

    result = extract_ocr_result(record)
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    loaded_result = json.loads(result_path.read_text(encoding="utf-8"))

    assert loaded_result["project_name"] == "STATE GOVERNMENT OFFICES"
    assert loaded_result["project_name_conf"] == 0.95
    assert loaded_result["project_name_source"] == "rule"


def test_token_evidence_is_preserved_in_serialized_result():
    record = SheetRecord(
        node_id="460674",
        path="sample.jpg",
        fields={
            "drawing_title": FieldValue(
                name="drawing_title",
                verbatim="ROOF DETAILS",
                tokens=[7, 8],
            )
        },
        token_map={
            7: {"bbox": [10, 20, 30, 40]},
            8: {"bbox": [32, 20, 60, 40]},
        },
    )

    result = extract_ocr_result(record)

    assert result["drawing_title"] == "ROOF DETAILS"
    assert result["drawing_title_tokens_bbox"] == {
        "x0": 10,
        "y0": 20,
        "x1": 60,
        "y1": 40,
    }


def test_manifest_identifies_series_and_summarizes_images(tmp_path: Path):
    image_path = tmp_path / "YFA_state_government" / "460674" / "drawing.jpg"
    image_path.parent.mkdir(parents=True)
    write_sample_image(image_path)

    rows = build_manifest(tmp_path)
    summary = summarize(rows)

    assert len(rows) == 1
    assert rows[0]["node_id"] == "460674"
    assert rows[0]["series"] == "YFA-StateGov"
    assert rows[0]["sha256"]
    assert summary["total"] == 1
    assert summary["series"]["YFA-StateGov"]["count"] == 1


def test_series_inference_handles_known_and_unknown_paths():
    assert infer_series(Path("plan_sample/YFA_cabana/123/image.jpg")) == "YFA-Cabana"
    assert infer_series(Path("plan_sample/BSM/123/image.jpg")) == "BSM"
    assert infer_series(Path("other/123/image.jpg")) == "unknown"
