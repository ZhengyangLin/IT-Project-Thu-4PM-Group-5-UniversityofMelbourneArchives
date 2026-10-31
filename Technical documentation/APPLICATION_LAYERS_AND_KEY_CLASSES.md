## Application Layers and Key Classes

Four logical layers organise the web application. Classes are listed alongside the responsibilities they implement. Backend paths below are relative to `uma/`.

| Layer | Main code and classes | Responsibility |
| --- | --- | --- |
| Presentation | `front-end/src/App.vue`; `front-end/src/components/FabricImg.vue` | Task/result display, processing controls, exports, drawing inspection, and manual editing. |
| HTTP API | `web/app.py`; `web/api/`: `BaseAPI` (router base), `UnimelbAPI` (endpoints); `web/schemas/` | Application assembly, router structure, endpoints, request/response schemas, and delegation to the service. |
| Application service | `web/services/`: `UnimelbService` (contract), `UnimelbServiceImpl` (implementation) | Image discovery, background tasks, state persistence, manual reviews, and exports. |
| OCR processing | `pipeline.py`: `Pipeline` | Coordinates the processing modules for one drawing and returns a `SheetRecord`. |

Processing modules within `Pipeline`:

| Processing area | Modules and key classes | Responsibility |
| --- | --- | --- |
| Image preparation | `preprocess.py`, `triage.py` | Grayscale preparation, small-skew correction, contrast enhancement, and readability assessment. |
| OCR and fusion | `ocr/engines.py`: `Engine`, `PaddleEngine`; `ocr/fusion.py` | OCR adapter interface and PaddleOCR implementation; alternative readings and pass merging. Web workers use PaddleOCR. |
| Localisation | `locate.py`, `table_regions.py` | Title-block selection using image geometry and OCR evidence. |
| Layout | `layout.py`, `views.py` | Token rows, spatial text grids, and indexed text views. |
| Extraction | `rules.py`, `patterns.py`, `llm.py`: `LlmClient` | Field patterns and spatial rules; LLM completion of unresolved fields, with client availability checks and request retries. |
| Validation | `llm.py`, `verify.py` | Model-evidence checks, combined confidence scores, and field review statuses. |

Full-page OCR precedes localisation; cropped OCR follows it. Model evidence is checked during field assignment.

Supporting code:

- **Configuration:** `config.py` — `Config`, a dataclass holding grouped settings, loads recognised YAML values over defaults.
- **Diagnostics:** `trace.py` — `Tracer` writes per-drawing intermediate traces; `viz.py` produces debug images; `logging_utils.py` handles logs.
- **Results:** `unimelb_result_codec.py` converts records for the web service; `pipeline.write_outputs()` writes processing outputs. Image-state persistence belongs to the service.

Data objects:

| Class | Module | Stored information |
| --- | --- | --- |
| `OcrImageJob` / `OcrImageOutcome` | `web/workers/Unimelb_ocr_worker.py` | Cross-process image request / result or error. |
| `RawToken` | `ocr/engines.py` | Pre-fusion text, box, confidence, and engine name. |
| `Candidate` | `models.py` | One OCR reading: text, engine, confidence, optional source box. |
| `Token` | `models.py` | Selected text, box, confidence, alternative readings, token ID. |
| `Line` | `models.py` | Tokens in one row, with combined text and box. |
| `FieldValue` | `models.py` | Verbatim/corrected values, evidence token IDs, confidence, review status. |
| `SheetRecord` | `models.py` | One drawing's fields, title-block location, text views, token map, diagnostics. |
| `TableProposal` | `table_regions.py` | `find_table()` candidate region (possibly absent), reliability flag, method, diagnostics. |
| `LocateResult` | `locate.py` | `locate()` selected region, method, confidence, diagnostics. |

Execution and review:

- **Workers:** The service submits work through `web/workers/Unimelb_ocr_worker.py` to a pool of up to two processes; images within each batch run sequentially. Each worker reuses a `Pipeline`, calls `process()`, writes outputs, and returns an `OcrImageOutcome`.
- **Status:** Image processing status, field review status, and `manual_reviewed` are independent. `Success` means processing completed successfully; fields may still need review.
- **Review save:** Updates submitted non-null field values and sets `manual_reviewed=1`; preserves OCR evidence, confidence scores, and field review statuses.

