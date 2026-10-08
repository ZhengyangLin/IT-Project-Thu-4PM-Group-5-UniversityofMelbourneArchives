from __future__ import annotations

import logging
import os
from dataclasses import dataclass

# Configure Paddle runtime compatibility before importing OCR libraries.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
import cv2
import numpy as np
log = logging.getLogger(__name__)

@dataclass
class RawToken:
    text: str
    bbox: tuple[int, int, int, int]
    conf: float
    engine: str

# Base interface for OCR engine adapters.
class Engine:
    name = "base"

    def available(self) -> bool:
        raise NotImplementedError

    def read(self, gray: np.ndarray, upscale: float = 1.0, **kw) -> list[RawToken]:
        raise NotImplementedError

    @staticmethod
    def _scale_up(gray: np.ndarray, f: float,
                  max_side: int = 0) -> tuple[np.ndarray, float]:
        eff = f
        if max_side:
            longest = max(gray.shape) * f
            if longest > max_side:
                eff = f * max_side / longest
        if abs(eff - 1.0) < 1e-6:
            return gray, 1.0
        interp = cv2.INTER_CUBIC if eff > 1 else cv2.INTER_AREA
        return cv2.resize(gray, None, fx=eff, fy=eff, interpolation=interp), eff

    @staticmethod
    def _scale_box(b, f: float):
        import math
        return (int(math.floor(b[0] / f)), int(math.floor(b[1] / f)),
                int(math.ceil(b[2] / f)), int(math.ceil(b[3] / f)))

# PaddleOCR engine adapter.
class PaddleEngine(Engine):
    name = "paddle"

    PADDLE_DEFAULT_MAX_SIDE = 4000

    def __init__(self, lang: str = "en", use_angle: bool = True,
                 max_side_limit: int = 0):
        self._ocr = None
        self._lang, self._use_angle = lang, use_angle
        self._max_side = max_side_limit

    def _lazy(self):
        if self._ocr is not None:
            return self._ocr
        from paddleocr import PaddleOCR
        v3_off = dict(use_doc_orientation_classify=False, use_doc_unwarping=False)
        import os as _os
        _os.environ.setdefault("FLAGS_use_mkldnn", "0")
        attempts = [
            dict(lang=self._lang, use_textline_orientation=self._use_angle,
                 enable_mkldnn=False,
                 device="gpu:0",
                 **v3_off),
            dict(lang=self._lang, use_textline_orientation=self._use_angle,
                 device="gpu:0",
                 **v3_off),
            dict(lang=self._lang,
                 device="gpu:0",
                 **v3_off),
            dict(lang=self._lang, use_angle_cls=self._use_angle,
                 device="gpu:0",
                 show_log=False),
            dict(lang=self._lang),
        ]
        last = None
        for kw in attempts:
            try:
                self._ocr = PaddleOCR(**kw)
                self._api = "v3" if "use_textline_orientation" in kw else "v2"
                log.info("PaddleOCR initialized successfully with parameters: %s", kw)
                return self._ocr
            except (TypeError, ValueError) as e:
                last = e
        raise RuntimeError(
            f"PaddleOCR initialization failed after trying {len(attempts)} "
            f"parameter sets: {last}")

    def available(self) -> bool:
        try:
            import paddleocr
            return True
        except Exception as e:
            log.warning("PaddleOCR is unavailable: %s", e)
            return False

    def read(self, gray, upscale=1.0, det_thresh=None, **kw) -> list[RawToken]:
        import time
        img, eff = self._scale_up(gray, upscale, kw.get('max_side', 0))
        if abs(eff - upscale) > 1e-6:
            log.info('Paddle scale adjusted by the long-side limit: %.4f → %.4f',
                     upscale, eff)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        out: list[RawToken] = []
        ocr = self._lazy()
        log.info("Paddle inference started for %dx%d (large images may take "
                 "several minutes on CPU)",
                 img.shape[1], img.shape[0])
        t0 = time.time()
        need = int(max(img.shape[:2]) * 1.02) + 32
        limit = max(self._max_side, need) if self._max_side > 0 else need
        log.info("Paddle max_side_limit=%d (input long side: %d; internal "
                 "scaling disabled)",
                 limit, max(img.shape[:2]))

        calls = []
        if hasattr(ocr, "predict"):
            calls.append(lambda: ocr.predict(rgb, max_side_limit=limit))
            calls.append(lambda: ocr.predict(rgb))
        calls.append(lambda: ocr.ocr(rgb, cls=self._use_angle))
        calls.append(lambda: ocr.ocr(rgb))

        res, last = None, None
        try:
            for fn in calls:
                try:
                    res = fn()
                    break
                except TypeError as e:
                    last = e
                    continue
            if res is None:
                raise last or RuntimeError("All PaddleOCR invocation methods failed")
        except Exception as e:
            log.error("PaddleOCR text recognition failed: %s", e)
            return out
        toks = self._parse(res, eff)
        H0, W0 = gray.shape[:2]
        bad = [t for t in toks if t.bbox[2] > W0 * 1.02 or t.bbox[3] > H0 * 1.02]
        if bad:
            log.error("Coordinate restoration error: %d/%d boxes exceed the "
                      "%dx%d source image, reaching (%d,%d). OCR coordinates "
                      "are unreliable; check the PaddleOCR version",
                      len(bad), len(toks), W0, H0,
                      max(t.bbox[2] for t in bad), max(t.bbox[3] for t in bad))
        _check_extent(toks, gray.shape, self.name)
        log.info("PaddleOCR finished in %.1fs: %d blocks (scale %.4fx)",
                 time.time() - t0, len(toks), eff)
        return toks

    def _parse(self, res, eff: float) -> list[RawToken]:
        out: list[RawToken] = []
        if not res:
            return out

        def push(poly, txt, conf):
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            box = self._scale_box((min(xs), min(ys), max(xs), max(ys)), eff)
            if txt and txt.strip():
                out.append(RawToken(txt.strip(), box, float(conf), self.name))

        for page in res:
            if page is None:
                continue
            data = page
            if hasattr(page, "json"):
                data = page.json.get("res", page.json)
            if isinstance(data, dict):
                texts = data.get("rec_texts") or []
                scores = data.get("rec_scores") or []
                polys = data.get("rec_polys") or data.get("dt_polys") or []
                for i, txt in enumerate(texts):
                    conf = scores[i] if i < len(scores) else 0.0
                    poly = polys[i] if i < len(polys) else [[0, 0]]
                    push(poly, txt, conf)
                continue
            for item in (data or []):
                try:
                    poly, (txt, conf) = item[0], item[1]
                    push(poly, txt, conf)
                except (TypeError, ValueError, IndexError):
                    continue
        return out

# Tesseract OCR engine adapter.
class TesseractEngine(Engine):
    name = "tesseract"

    def __init__(self, lang: str = "eng", psms=(6, 11)):
        self._lang, self._psms = lang, psms

    def available(self) -> bool:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception as e:
            log.warning("Tesseract is unavailable: %s", e)
            return False

    def read(self, gray, upscale=1.0, psms=None, whitelist=None, **kw) -> list[RawToken]:
        import pytesseract
        img, eff = self._scale_up(gray, upscale, kw.get('max_side', 0))
        out, seen = [], set()
        for psm in (psms or self._psms):
            cfg = f"--psm {psm}"
            if whitelist:
                cfg += f" -c tessedit_char_whitelist={whitelist}"
            try:
                data = pytesseract.image_to_data(
                    img, lang=self._lang, config=cfg,
                    output_type=pytesseract.Output.DICT)
            except Exception as e:
                log.error("Tesseract failed with psm=%s: %s", psm, e)
                continue
            for i, txt in enumerate(data["text"]):
                txt = (txt or "").strip()
                conf = float(data["conf"][i])
                if not txt or conf < 0:
                    continue
                x, y = data["left"][i], data["top"][i]
                box = self._scale_box((x, y, x + data["width"][i],
                                       y + data["height"][i]), eff)
                key = (txt, box)
                if key in seen:
                    continue
                seen.add(key)
                out.append(RawToken(txt, box, conf / 100.0, self.name))
        _check_extent(out, gray.shape, self.name)
        return out

# docTR OCR engine adapter.
class DoctrEngine(Engine):
    name = "doctr"

    def __init__(self, **kw):
        self._model = None
        self._failed = False

    def _lazy(self):
        if self._model is None and not self._failed:
            try:
                from doctr.models import ocr_predictor
                self._model = ocr_predictor(pretrained=True)
            except Exception as e:
                self._failed = True
                log.error("Failed to download docTR weights; docTR has been "
                          "disabled: %s", e)
                log.error("  Download the weights manually to "
                          "~/.cache/doctr/models/, or select another available "
                          "engine with --engines")
        return self._model

    def available(self) -> bool:
        try:
            import doctr
            return True
        except Exception as e:
            log.warning("docTR is unavailable: %s", e)
            return False

    def read(self, gray, upscale=1.0, **kw) -> list[RawToken]:
        img, eff = self._scale_up(gray, upscale, kw.get('max_side', 0))
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        h, w = img.shape[:2]
        out = []
        model = self._lazy()
        if model is None:
            return out
        try:
            res = model([rgb])
        except Exception as e:
            log.error("docTR text recognition failed: %s", e)
            return out
        for page in res.pages:
            for blk in page.blocks:
                for line in blk.lines:
                    for word in line.words:
                        (rx0, ry0), (rx1, ry1) = word.geometry
                        box = self._scale_box(
                            (rx0 * w, ry0 * h, rx1 * w, ry1 * h), eff)
                        out.append(RawToken(word.value.strip(), box,
                                            float(word.confidence), self.name))
        _check_extent(out, gray.shape, self.name)
        return out

# Validate OCR coordinate coverage against the source image.
def _check_extent(tokens: list[RawToken], shape, engine: str) -> None:
    if not tokens:
        return
    H, W = shape[:2]
    x1 = max(t.bbox[2] for t in tokens)
    y1 = max(t.bbox[3] for t in tokens)
    cover_x, cover_y = x1 / max(W, 1), y1 / max(H, 1)
    if cover_x > 1.05 or cover_y > 1.05:
        log.error("[%s] Coordinates out of bounds: maximum (%d,%d) exceeds "
                  "source image (%d,%d); scale mapping is incorrect",
                  engine, x1, y1, W, H)
    elif cover_x < 0.55 or cover_y < 0.55:
        log.warning("[%s] Coordinates cover only %.0f%% × %.0f%% of the "
                    "source image; uncompensated internal scaling may be "
                    "present. Check the debug image",
                    engine, cover_x * 100, cover_y * 100)
    else:
        log.info("[%s] Coordinate coverage is %.0f%% × %.0f%%; mapping is valid",
                 engine, cover_x * 100, cover_y * 100)

REGISTRY = {"paddle": PaddleEngine, "tesseract": TesseractEngine, "doctr": DoctrEngine}

# Build the OCR engines selected in configuration.
def build_engines(names: list[str], lang: str = "en",
                  paddle_max_side: int = 0,
                  tess_psms: tuple = (6, 11)) -> list[Engine]:
    engines = []
    for n in names:
        cls = REGISTRY.get(n)
        if cls is None:
            log.warning("Unknown OCR engine: %s", n)
            continue
        if n == "paddle":
            eng = cls(lang=lang, max_side_limit=paddle_max_side)
        elif n == "tesseract":
            eng = cls(lang="eng", psms=tuple(tess_psms))
        else:
            eng = cls()
        if eng.available():
            engines.append(eng)
    if not engines:
        raise RuntimeError(
            "No OCR engine is available; cannot continue. See the installation "
            "instructions in README.md.")
    return engines