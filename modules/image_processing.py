
"""
Advanced image preprocessing for the SIH26034 project.

This module is intentionally separate from OCR.
It prepares label images and creates useful diagnostic variants.
"""

from typing import Any, Dict
import cv2
import numpy as np
from PIL import Image


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    if image is None:
        raise ValueError("Image cannot be None.")
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    if image is None:
        raise ValueError("Image cannot be None.")
    if image.ndim == 2:
        return Image.fromarray(image.astype(np.uint8)).convert("RGB")
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def image_quality(image: Image.Image) -> Dict[str, Any]:
    """Return practical capture-quality indicators."""
    bgr = pil_to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    glare_ratio = float(np.mean(gray >= 245) * 100)
    dark_ratio = float(np.mean(gray <= 35) * 100)

    warnings = []
    suggestions = []

    if min(w, h) < 500:
        warnings.append("Image resolution is low.")
        suggestions.append("Move closer to the package.")

    if sharpness < 70:
        warnings.append("Image may be blurred.")
        suggestions.append("Hold the camera steady and refocus.")

    if brightness < 45:
        warnings.append("Image is too dark.")
        suggestions.append("Use better lighting.")

    if brightness > 220:
        warnings.append("Image is very bright.")
        suggestions.append("Avoid direct light on the package.")

    if glare_ratio > 3:
        warnings.append("Strong glare detected.")
        suggestions.append("Tilt the package slightly to reduce reflection.")

    if dark_ratio > 35:
        warnings.append("Large dark region detected.")
        suggestions.append("Capture the label with more even lighting.")

    score = 100.0
    score -= min(25.0, max(0.0, 70.0 - sharpness) * 0.20)
    score -= min(20.0, max(0.0, 45.0 - brightness) * 0.35)
    score -= min(20.0, max(0.0, brightness - 220.0) * 0.25)
    score -= min(20.0, glare_ratio * 3.0)
    score -= min(15.0, max(0.0, dark_ratio - 35.0) * 0.25)
    score = max(0.0, min(100.0, score))

    status = "GOOD" if score >= 75 else "ACCEPTABLE" if score >= 50 else "POOR"

    return {
        "width": w,
        "height": h,
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "sharpness": round(sharpness, 1),
        "glare_ratio": round(glare_ratio, 2),
        "dark_ratio": round(dark_ratio, 2),
        "score": round(score, 1),
        "status": status,
        "warnings": warnings,
        "suggestions": suggestions,
    }


def _resize(bgr: np.ndarray, max_side: int = 2200) -> np.ndarray:
    h, w = bgr.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return bgr
    scale = max_side / float(side)
    return cv2.resize(
        bgr,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Main preprocessing path:
    RGB -> resize -> illumination correction -> denoise ->
    CLAHE -> gamma/contrast -> sharpen.
    """
    bgr = _resize(pil_to_bgr(image))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Local contrast enhancement.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Mild denoising.
    denoised = cv2.fastNlMeansDenoising(
        enhanced_gray, None, h=6, templateWindowSize=7, searchWindowSize=21
    )

    # Unsharp masking.
    blurred = cv2.GaussianBlur(denoised, (0, 0), 1.2)
    sharpened = cv2.addWeighted(denoised, 1.35, blurred, -0.35, 0)

    return bgr_to_pil(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR))


def create_preprocessing_variants(image: Image.Image) -> Dict[str, Image.Image]:
    """
    Creates preprocessing variants only.
    OCR should decide which variants to read.
    """
    bgr = _resize(pil_to_bgr(image))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(
        clahe_img, None, h=6, templateWindowSize=7, searchWindowSize=21
    )

    blur = cv2.GaussianBlur(denoised, (0, 0), 1.2)
    sharpened = cv2.addWeighted(denoised, 1.35, blur, -0.35, 0)

    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    otsu = cv2.threshold(
        sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    return {
        "enhanced": bgr_to_pil(cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)),
        "denoised": bgr_to_pil(cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)),
        "sharpened": bgr_to_pil(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)),
        "adaptive_threshold": bgr_to_pil(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)),
        "otsu_threshold": bgr_to_pil(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)),
    }
