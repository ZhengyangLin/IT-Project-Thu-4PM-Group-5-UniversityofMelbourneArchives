## 1. Application Layers and Key Classes

| Layer | Main code | Responsibility |
| --- | --- | --- |
| Frontend | `front-end/src/App.vue` | Displays image tasks, filters and paginates results, starts processing/rechecks, and selects images for export. |
| Drawing review | `front-end/src/components/FabricImg.vue` | Displays drawings and evidence boxes; supports zoom, pan, field editing, and saving manual reviews. |
| Backend service | `uma/web/app.py::create_app`, `UnimelbService`, `UnimelbServiceImpl` in `uma/web/services/` | Manages application lifecycle, image discovery, task coordination, saved results, review, and export. |
| Worker execution | `uma/web/workers/Unimelb_ocr_worker.py` | `OcrImageJob` carries work to a process; `OcrImageOutcome` returns results/errors. Each worker reuses one `Pipeline`. |
| Pipeline | `uma/pipeline.py::Pipeline` | Coordinates preprocessing, OCR, localisation, extraction, evidence checks, scoring, and output generation. |
| Preprocessing and readability | `uma/preprocess.py::run`, `uma/triage.py::assess` | Produces grayscale image variants, corrects small skew, enhances contrast, and assesses readability. |
| OCR and fusion | `uma/ocr/engines.py`, `uma/ocr/fusion.py` | `Engine` defines the adapter interface; `PaddleEngine` performs current web recognition. Fusion retains alternative readings and merges passes. |
| Localisation and layout | `uma/locate.py`, `uma/table_regions.py`, `uma/layout.py`, `uma/views.py` | Finds the title block, groups text into rows, and builds numbered tokens and spatial text views. |
| Field extraction | `uma/rules.py::extract`, `uma/patterns.py`, `uma/llm.py::LlmClient` | Combines field patterns and spatial rules with LLM completion of unresolved fields. |
| Validation and scoring | `uma/llm.py::verify_evidence`, `uma/verify.py::fuse` | Checks available OCR evidence, combines confidence signals, and assigns field review statuses. |
| Results and diagnostics | `uma/unimelb_result_codec.py`, `uma/trace.py::Tracer`, `uma/viz.py`, `uma/logging_utils.py` | Converts results for the interface and records processing details, debug images, and logs. |
| Configuration | `uma/config.py::Config` | Merges recognised YAML settings into dataclass defaults; the web worker then enforces its Paddle-only engine selection. |

Core objects in `uma/models.py` are `Candidate` (OCR reading), `Token` (text, box, confidence, alternatives and ID), `Line` (text row), `FieldValue` (value and evidence), and `SheetRecord` (one drawing's results and diagnostics). `LocateResult` and `TableProposal` carry localisation decisions in their respective modules.

The backend uses a process pool with up to two workers, while each individual batch processes its images sequentially. Image processing status, field review status, and `manual_reviewed` are separate: a `Success` image can still require manual review. Saving a review changes final values and the review marker while retaining original OCR evidence and confidence.

