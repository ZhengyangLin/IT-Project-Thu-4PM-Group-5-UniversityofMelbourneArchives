## Third-Party Systems and How They Are Called

| Component | Actual use and call mechanism |
| --- | --- |
| PaddleOCR / PaddlePaddle | `PaddleEngine` loads `PaddleOCR` lazily and calls supported `predict()` or `ocr()` interfaces on image arrays. Text, scores, and polygons become `RawToken` objects. This is the active web OCR engine. |
| LLM service / OpenAI Python SDK | `LlmClient` calls `chat.completions.create()` at the configured compatible endpoint. Input is OCR-derived text; returned text is parsed into fields. This does not imply the provider is OpenAI. |
| OpenCV and NumPy | Image decoding, enhancement, rotation, geometry, resizing, and numerical arrays. |
| RapidFuzz and SciPy | Fuzzy text comparisons and global anchor/value assignment. |
| PyYAML, pandas, and openpyxl | Configuration loading, processing CSV summaries, and downloadable Excel workbooks. |
| Vue, Element Plus, Axios, and Fabric.js | Interface state and controls, backend requests, drawing display, zoom/pan, and evidence overlays. |
| FastAPI and Uvicorn | Backend application lifecycle and request dispatch to the service layer. |
| Pydantic | API request and response schemas use `BaseModel`, `Field`, and validators for data validation and serialization. Manual-review results are converted with `model_dump()` before being passed to the service layer. |
| Vite / Node.js | Node.js runs the Vite development server through `yarn dev`. Vite serves the Vue frontend and proxies `/Unimelb` requests to the FastAPI backend. |
| @element-plus/icons-vue | Vue icon components are imported and rendered in the application interface, including the page heading, status cards, and action controls. |

