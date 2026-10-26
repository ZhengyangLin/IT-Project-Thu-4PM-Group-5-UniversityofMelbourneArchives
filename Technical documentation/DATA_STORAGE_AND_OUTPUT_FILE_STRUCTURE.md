## 4. Output File Structure

The default web-processing structure is:

```text
output/
|-- unimelb_image_status.json
|-- logs/
|   `-- YYYY-MM-DD.log
`-- tasks/
    `-- <shared_task_uuid>/
        |-- sheets.csv
        `-- items/
            `-- <NodeID>/
                `-- <image_name>/
                    |-- sheets.csv
                    |-- sidecar/
                    |   `-- <NodeID>.json
                    |-- trace/
                    |   `-- <NodeID>.txt
                    `-- debug/
                        |-- <NodeID>_0_tokens.jpg
                        |-- <NodeID>_1_overview.jpg
                        `-- <NodeID>_2_titleblock.jpg
```

| File | Contents |
| --- | --- |
| `unimelb_image_status.json` | Current image statuses, saved field results, errors, timestamps, and manual-review markers/values. |
| `logs/YYYY-MM-DD.log` | Daily application and worker logs. |
| Task-level `sheets.csv` | Aggregate processing rows; refreshed only when a create-task batch covers the full collected image set. Partial runs and selected rechecks do not rebuild it. |
| Per-image `sheets.csv` | The image's extracted values, original/corrected text, confidence/status fields, and processing/localisation summary. |
| `sidecar/<NodeID>.json` | Detailed OCR record: fields, evidence tokens, bounding boxes, text views, localisation, and diagnostics. |
| `trace/<NodeID>.txt` | Intermediate processing text, including token readings, model input/output, and final fields when tracing is enabled. |
| `debug/*.jpg` | Token boxes, localisation overview, and title-block crop when visualisation is enabled and processing reaches those stages. |

The current shared task UUID is `00000000-0000-4000-8000-000000000001`. `<image_name>` is a directory named after the input file, including its extension. Reprocessing uses the same locations. If writing `sheets.csv` raises `PermissionError`, the writer attempts `sheets_HHMMSS.csv` instead.

Manual review updates `unimelb_image_status.json`; it does not rewrite recognition sidecars or CSV files. Excel export uses the current saved results and is downloaded by the browser, rather than saved as a fixed file in this tree.

In normal command-line processing with `run.py`, CSV, `sidecar/`, `trace/`, `debug/`, and `logs/` are written directly under the selected output directory rather than under the web task/item hierarchy. `--viz-only` skips CSV and sidecar export. The `--triage-only` mode writes `manifest.csv` instead of running field extraction. Failed or interrupted processing may leave only some artifacts.

### Exported Excel File (`.xlsx`)

The Export action downloads `unimelb_image_status.xlsx` (frontend fallback: `export.xlsx`). It is generated in memory and downloaded by the browser, outside the server's `output/` tree.

| Item | Contents |
| --- | --- |
| Worksheet | `OCR Results`; one row per included image, using current saved values, including manual edits. |
| Column order | `image_key`, `drawing_number`, `project_name`, `drawing_title`, `architect`, `draughtsperson`, `date`, `scale`, `drawing_type`. |
| Confidence columns | If any included image is unreviewed, each field is followed by `<field>_conf` (17 columns total); reviewed rows leave these cells blank. If all images are reviewed, confidence columns are omitted (9 columns). |
| Highlighting | Reviewed images have a green `image_key` cell. For unreviewed images, fields with confidence below the current `t_high` (`0.75`) have red value and confidence cells. |
| Formatting | Text values remain text; confidence uses three decimal places. The header and first column are frozen, with header filters enabled. |

The interface exports selected images, or all images matching the current filters when none are selected. Running images or images reserved by another operation block the export.
