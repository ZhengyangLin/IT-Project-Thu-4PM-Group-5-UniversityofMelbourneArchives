"""Step 1 · Prepare the image for OCR.

The preprocessing stage intentionally avoids several aggressive enhancement
methods that may alter the original character shapes or introduce artifacts.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import PreprocessCfg


def load_gray(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Unable to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def text_mask(gray: np.ndarray) -> np.ndarray:
    """Create an approximate text mask for skew estimation."""
    binary = cv2.adaptiveThreshold(
        gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10)
    # Connect nearby characters horizontally to form line-like regions for angle estimation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def estimate_skew(gray: np.ndarray, cfg: PreprocessCfg) -> tuple[float, str]:
    """Estimate the skew angle and return its processing status."""
    mask = text_mask(gray)
    ratio = float(mask.mean())
    if ratio < cfg.min_text_ratio:
        return 0.0, "skipped_sparse"

    pts = cv2.findNonZero(mask)
    if pts is None or len(pts) < 50:
        return 0.0, "skipped_sparse"

    angle = cv2.minAreaRect(pts)[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    if abs(angle) > 15:  # Reject unusually large estimates to avoid incorrect rotation
        return 0.0, "rejected_outlier"
    return float(angle), "ok"


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def enhance(gray: np.ndarray, cfg: PreprocessCfg) -> np.ndarray:
    """Improve local contrast using CLAHE while limiting excessive amplification.

    Local contrast enhancement is preferred because scanned drawings may have
    uneven background brightness across different regions.
    """
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip,
                            tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
    return clahe.apply(gray)


def run(path: str, cfg: PreprocessCfg
        ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return the enhanced image, plain grayscale image, and preprocessing metadata."""
    gray = load_gray(path)
    meta = {"width": int(gray.shape[1]), "height": int(gray.shape[0])}

    angle, status = estimate_skew(gray, cfg)
    meta["skew_angle"] = round(angle, 3)
    meta["skew_status"] = status
    if status == "ok" and abs(angle) >= cfg.deskew_deadzone_deg:
        gray = rotate(gray, angle)
        meta["deskewed"] = True
    else:
        meta["deskewed"] = False

    # Produce two image variants for different downstream processing stages
    # enhanced: used for OCR because local contrast enhancement improves faint text visibility
    # plain: retained for layout analysis to avoid introducing additional background texture
    enhanced = enhance(gray, cfg)
    return enhanced, gray, meta