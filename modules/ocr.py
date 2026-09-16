import re
import math
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import cv2
import easyocr
import numpy as np
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

OCR_LANGUAGES = ["en"]

# CPU safe.
# Change to True only if CUDA + compatible GPU setup is working.
USE_GPU = False

# ------------------------------------------------------------
# IMAGE SIZE
# ------------------------------------------------------------

MAX_IMAGE_SIDE = 2800
MIN_IMAGE_SIDE = 1100

# ------------------------------------------------------------
# OCR
# ------------------------------------------------------------

MIN_CONFIDENCE = 0.15

# Detection settings.
TEXT_THRESHOLD = 0.30
LOW_TEXT = 0.10
LINK_THRESHOLD = 0.25

WIDTH_THRESHOLD = 0.75
HEIGHT_THRESHOLD = 0.75

# ------------------------------------------------------------
# TILES
# ------------------------------------------------------------

TILE_ROWS = 3
TILE_COLS = 3
TILE_OVERLAP = 0.18

MAX_TILES = 9

# ------------------------------------------------------------
# DUPLICATE MERGING
# ------------------------------------------------------------

TEXT_SIMILARITY_THRESHOLD = 0.84
NEAR_TEXT_SIMILARITY_THRESHOLD = 0.68

SPATIAL_DISTANCE_RATIO = 0.035

# ------------------------------------------------------------
# ROTATION
# ------------------------------------------------------------

ENABLE_ROTATION_PASSES = True

# ------------------------------------------------------------
# SPECIAL REGIONS
# ------------------------------------------------------------

ENABLE_SPECIAL_REGIONS = True

# ============================================================
# EASY OCR INITIALIZATION
# ============================================================

reader = easyocr.Reader(
    OCR_LANGUAGES,
    gpu=USE_GPU,
    verbose=False
)


# ============================================================
# BASIC TEXT NORMALIZATION
# ============================================================

def normalize_text(text: Any) -> str:
    """
    Basic OCR text normalization.

    IMPORTANT:
    We do not aggressively correct arbitrary letters because
    product names, company names and batch numbers must remain
    unchanged as much as possible.
    """

    if text is None:
        return ""

    text = str(text).strip()

    text = text.replace("\r", " ")
    text = text.replace("\n", " ")

    # Unicode whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Repeated punctuation.
    text = re.sub(
        r"([.,:;!?])\1+",
        r"\1",
        text
    )

    return text.strip()


def repair_field_text(text: str) -> str:
    """
    Conservative correction of common OCR mistakes in
    FIELD HEADINGS.

    We intentionally don't modify arbitrary product text.
    """

    if not text:
        return ""

    result = text

    replacements = [

        # MRP
        (
            r"\bM\s*\.?\s*R\s*\.?\s*P\s*\.?\b",
            "MRP"
        ),

        # Net weight
        (
            r"\bNET\s+WT\.?\b",
            "NET WEIGHT"
        ),

        (
            r"\bNET\s+QTY\.?\b",
            "NET QUANTITY"
        ),

        # Batch
        (
            r"\bBATCH\s+NO\.?\b",
            "BATCH NO"
        ),

        (
            r"\bLOT\s+NO\.?\b",
            "LOT NO"
        ),

        # Manufacturing
        (
            r"\bMFD\.?\s+BY\b",
            "MANUFACTURED BY"
        ),

        (
            r"\bMFG\.?\s+BY\b",
            "MANUFACTURED BY"
        ),

        # FSSAI
        (
            r"\bFSSAI\s+LIC\.?\s+NO\.?\b",
            "FSSAI LIC NO"
        ),
    ]

    for pattern, replacement in replacements:

        result = re.sub(
            pattern,
            replacement,
            result,
            flags=re.IGNORECASE
        )

    return normalize_text(
        result
    )


def is_valid_text(text: str) -> bool:
    """
    Remove obvious OCR garbage.
    """

    text = normalize_text(
        text
    )

    if len(text) < 2:
        return False

    # Must contain at least two useful characters.
    alphanumeric = re.findall(
        r"[A-Za-z0-9]",
        text
    )

    if len(alphanumeric) < 2:
        return False

    # Reject extreme punctuation garbage.
    punctuation = re.findall(
        r"[^A-Za-z0-9\s]",
        text
    )

    if len(punctuation) > len(text) * 0.75:
        return False

    return True


# ============================================================
# IMAGE SAFETY
# ============================================================

def ensure_bgr(
    image: np.ndarray
) -> np.ndarray:
    """
    Guarantee a 3-channel BGR uint8 image.

    This prevents the OpenCV:

        Invalid number of channels
        scn is 1

    error.
    """

    if image is None:
        raise ValueError(
            "OCR received an empty image."
        )

    image = np.asarray(
        image
    )

    if image.dtype != np.uint8:

        image = np.clip(
            image,
            0,
            255
        ).astype(
            np.uint8
        )

    # Grayscale.
    if image.ndim == 2:

        return np.ascontiguousarray(
            cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )
        )

    if image.ndim != 3:

        raise ValueError(
            f"Unsupported image shape: {image.shape}"
        )

    channels = image.shape[2]

    # H x W x 1.
    if channels == 1:

        return np.ascontiguousarray(
            cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )
        )

    # H x W x 3.
    if channels == 3:

        return np.ascontiguousarray(
            image
        )

    # H x W x 4.
    if channels == 4:

        return np.ascontiguousarray(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2BGR
            )
        )

    raise ValueError(
        f"Unsupported channel count: {channels}"
    )


def pil_to_cv(
    image: Image.Image
) -> np.ndarray:
    """
    PIL -> 3-channel OpenCV BGR.
    """

    if image is None:
        raise ValueError(
            "OCR received no image."
        )

    image = image.convert(
        "RGB"
    )

    array = np.asarray(
        image,
        dtype=np.uint8
    )

    return ensure_bgr(
        cv2.cvtColor(
            array,
            cv2.COLOR_RGB2BGR
        )
    )


# ============================================================
# IMAGE RESIZING
# ============================================================

def resize_for_ocr(
    image: np.ndarray
) -> np.ndarray:
    """
    Resize while preserving aspect ratio.
    """

    image = ensure_bgr(
        image
    )

    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError(
            "Image has invalid dimensions."
        )

    largest = max(
        height,
        width
    )

    smallest = min(
        height,
        width
    )

    # --------------------------------------------------------
    # Downscale large images.
    # --------------------------------------------------------

    if largest > MAX_IMAGE_SIDE:

        scale = (
            MAX_IMAGE_SIDE
            / largest
        )

        new_width = max(
            1,
            int(width * scale)
        )

        new_height = max(
            1,
            int(height * scale)
        )

        image = cv2.resize(
            image,
            (
                new_width,
                new_height
            ),
            interpolation=cv2.INTER_AREA
        )

    # --------------------------------------------------------
    # Upscale small images.
    # --------------------------------------------------------

    elif smallest < MIN_IMAGE_SIDE:

        scale = (
            MIN_IMAGE_SIDE
            / smallest
        )

        new_width = int(
            width * scale
        )

        new_height = int(
            height * scale
        )

        # Never exceed maximum side.
        scale_limit = (
            MAX_IMAGE_SIDE
            / max(
                new_width,
                new_height
            )
        )

        if scale_limit < 1:

            new_width = int(
                new_width
                * scale_limit
            )

            new_height = int(
                new_height
                * scale_limit
            )

        image = cv2.resize(
            image,
            (
                max(1, new_width),
                max(1, new_height)
            ),
            interpolation=cv2.INTER_CUBIC
        )

    return np.ascontiguousarray(
        image
    )


# ============================================================
# IMAGE QUALITY
# ============================================================

def calculate_brightness(
    image: np.ndarray
) -> float:

    gray = cv2.cvtColor(
        ensure_bgr(image),
        cv2.COLOR_BGR2GRAY
    )

    return float(
        np.mean(gray)
    )


def calculate_sharpness(
    image: np.ndarray
) -> float:

    gray = cv2.cvtColor(
        ensure_bgr(image),
        cv2.COLOR_BGR2GRAY
    )

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )


def calculate_glare(
    image: np.ndarray
) -> float:

    gray = cv2.cvtColor(
        ensure_bgr(image),
        cv2.COLOR_BGR2GRAY
    )

    if gray.size == 0:
        return 0.0

    glare = np.sum(
        gray >= 245
    )

    return float(
        glare
        / gray.size
        * 100
    )


def get_image_quality(
    image: np.ndarray
) -> Dict[str, Any]:

    image = ensure_bgr(
        image
    )

    brightness = calculate_brightness(
        image
    )

    sharpness = calculate_sharpness(
        image
    )

    glare = calculate_glare(
        image
    )

    score = 100.0

    # Brightness.
    if brightness < 35:
        score -= 30

    elif brightness < 55:
        score -= 15

    elif brightness < 75:
        score -= 5

    elif brightness > 240:
        score -= 25

    elif brightness > 225:
        score -= 10

    # Sharpness.
    if sharpness < 30:
        score -= 35

    elif sharpness < 60:
        score -= 20

    elif sharpness < 100:
        score -= 10

    # Glare.
    if glare > 40:
        score -= 25

    elif glare > 25:
        score -= 15

    elif glare > 15:
        score -= 5

    score = max(
        0.0,
        min(
            100.0,
            score
        )
    )

    if score >= 75:
        status = "GOOD"

    elif score >= 50:
        status = "FAIR"

    else:
        status = "POOR"

    warnings = []

    if brightness < 50:
        warnings.append(
            "Image is dark."
        )

    if brightness > 225:
        warnings.append(
            "Image is overexposed."
        )

    if sharpness < 70:
        warnings.append(
            "Image may be blurry."
        )

    if glare > 20:
        warnings.append(
            "Strong glare detected."
        )

    return {
        "score": round(
            score,
            1
        ),
        "status": status,
        "brightness": round(
            brightness,
            2
        ),
        "sharpness": round(
            sharpness,
            2
        ),
        "glare": round(
            glare,
            2
        ),
        "warnings": warnings
    }


# ============================================================
# BASIC IMAGE PROCESSING
# ============================================================

def to_gray(
    image: np.ndarray
) -> np.ndarray:

    if image.ndim == 2:
        return image

    return cv2.cvtColor(
        ensure_bgr(image),
        cv2.COLOR_BGR2GRAY
    )


def denoise(
    gray: np.ndarray
) -> np.ndarray:

    gray = to_gray(
        gray
    )

    return cv2.fastNlMeansDenoising(
        gray,
        None,
        h=6,
        templateWindowSize=7,
        searchWindowSize=21
    )


def clahe(
    gray: np.ndarray
) -> np.ndarray:

    gray = to_gray(
        gray
    )

    enhancer = cv2.createCLAHE(
        clipLimit=2.4,
        tileGridSize=(8, 8)
    )

    return enhancer.apply(
        gray
    )


def sharpen(
    gray: np.ndarray
) -> np.ndarray:

    gray = to_gray(
        gray
    )

    blurred = cv2.GaussianBlur(
        gray,
        (0, 0),
        1.15
    )

    result = cv2.addWeighted(
        gray,
        1.45,
        blurred,
        -0.45,
        0
    )

    return np.clip(
        result,
        0,
        255
    ).astype(
        np.uint8
    )


def gamma_correction(
    image: np.ndarray,
    gamma: float
) -> np.ndarray:

    image = np.asarray(
        image,
        dtype=np.uint8
    )

    if gamma <= 0:
        gamma = 1.0

    inverse = 1.0 / gamma

    table = np.array(
        [
            (
                (i / 255.0)
                ** inverse
            )
            * 255
            for i in range(256)
        ],
        dtype=np.uint8
    )

    return cv2.LUT(
        image,
        table
    )


# ============================================================
# THRESHOLDING
# ============================================================

def adaptive_threshold(
    gray: np.ndarray
) -> np.ndarray:

    gray = to_gray(
        gray
    )

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8
    )


def otsu_threshold(
    gray: np.ndarray
) -> np.ndarray:

    gray = to_gray(
        gray
    )

    _, result = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    return result


def blackhat(
    gray: np.ndarray
) -> np.ndarray:
    """
    Helps detect dark text on bright packaging.
    """

    gray = to_gray(
        gray
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (25, 25)
    )

    result = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    result = cv2.normalize(
        result,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return result.astype(
        np.uint8
    )


# ============================================================
# PERSPECTIVE / DOCUMENT CORRECTION
# ============================================================

def order_points(
    points: np.ndarray
) -> np.ndarray:
    """
    Order four points:

        top-left
        top-right
        bottom-right
        bottom-left
    """

    points = np.asarray(
        points,
        dtype=np.float32
    )

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    sums = points.sum(
        axis=1
    )

    differences = np.diff(
        points,
        axis=1
    ).reshape(-1)

    ordered[0] = points[
        np.argmin(sums)
    ]

    ordered[2] = points[
        np.argmax(sums)
    ]

    ordered[1] = points[
        np.argmin(differences)
    ]

    ordered[3] = points[
        np.argmax(differences)
    ]

    return ordered


def four_point_transform(
    image: np.ndarray,
    points: np.ndarray
) -> np.ndarray:
    """
    Perspective correction for a detected package/document.
    """

    image = ensure_bgr(
        image
    )

    rect = order_points(
        points
    )

    tl, tr, br, bl = rect

    width_a = np.linalg.norm(
        br - bl
    )

    width_b = np.linalg.norm(
        tr - tl
    )

    max_width = int(
        max(
            width_a,
            width_b
        )
    )

    height_a = np.linalg.norm(
        tr - br
    )

    height_b = np.linalg.norm(
        tl - bl
    )

    max_height = int(
        max(
            height_a,
            height_b
        )
    )

    if max_width < 50 or max_height < 50:
        return image

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ],
        dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(
        rect,
        destination
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (
            max_width,
            max_height
        ),
        borderMode=cv2.BORDER_REPLICATE
    )

    return warped


def try_document_correction(
    image: np.ndarray
) -> np.ndarray:
    """
    Conservative perspective correction.

    We only apply it when a strong large quadrilateral is found.

    This avoids destroying normal package images.
    """

    image = ensure_bgr(
        image
    )

    height, width = image.shape[:2]

    gray = to_gray(
        image
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blur,
        60,
        180
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = (
        width * height
    )

    best = None
    best_area = 0

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < image_area * 0.30:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue

        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approximation) != 4:
            continue

        if area > best_area:

            best_area = area
            best = approximation.reshape(
                4,
                2
            )

    if best is None:
        return image

    try:

        corrected = four_point_transform(
            image,
            best
        )

        # Don't accept absurd transformations.
        if corrected.shape[0] < 200:
            return image

        if corrected.shape[1] < 200:
            return image

        return corrected

    except Exception:
        return image


# ============================================================
# IMAGE VARIANTS
# ============================================================

def create_variants(
    image: np.ndarray
) -> List[Tuple[str, np.ndarray]]:
    """
    Generate high-value OCR variants.

    We don't create dozens of nearly identical variants.
    """

    image = ensure_bgr(
        image
    )

    gray = to_gray(
        image
    )

    denoised = denoise(
        gray
    )

    enhanced = clahe(
        denoised
    )

    sharp = sharpen(
        enhanced
    )

    adaptive = adaptive_threshold(
        enhanced
    )

    otsu = otsu_threshold(
        enhanced
    )

    dark = gamma_correction(
        enhanced,
        1.30
    )

    bright = gamma_correction(
        enhanced,
        0.78
    )

    black = blackhat(
        enhanced
    )

    variants = [

        (
            "original",
            image
        ),

        (
            "clahe",
            cv2.cvtColor(
                enhanced,
                cv2.COLOR_GRAY2BGR
            )
        ),

        (
            "sharpened",
            cv2.cvtColor(
                sharp,
                cv2.COLOR_GRAY2BGR
            )
        ),

        (
            "adaptive",
            cv2.cvtColor(
                adaptive,
                cv2.COLOR_GRAY2BGR
            )
        ),

        (
            "otsu",
            cv2.cvtColor(
                otsu,
                cv2.COLOR_GRAY2BGR
            )
        ),

        (
            "dark_gamma",
            cv2.cvtColor(
                dark,
                cv2.COLOR_GRAY2BGR
            )
        ),

        (
            "bright_gamma",
            cv2.cvtColor(
                bright,
                cv2.COLOR_GRAY2BGR
            )
        ),

        (
            "blackhat",
            cv2.cvtColor(
                black,
                cv2.COLOR_GRAY2BGR
            )
        )
    ]

    return variants


# ============================================================
# OCR BOX UTILITIES
# ============================================================

def normalize_box(
    box
) -> np.ndarray:

    points = np.asarray(
        box,
        dtype=np.float32
    )

    if points.shape != (4, 2):

        points = points.reshape(
            -1,
            2
        )

    return points


def box_center(
    box
) -> Tuple[float, float]:

    points = normalize_box(
        box
    )

    return (
        float(
            np.mean(
                points[:, 0]
            )
        ),
        float(
            np.mean(
                points[:, 1]
            )
        )
    )


def box_size(
    box
) -> Tuple[float, float]:

    points = normalize_box(
        box
    )

    width = (
        np.max(
            points[:, 0]
        )
        -
        np.min(
            points[:, 0]
        )
    )

    height = (
        np.max(
            points[:, 1]
        )
        -
        np.min(
            points[:, 1]
        )
    )

    return (
        float(width),
        float(height)
    )


def box_area(
    box
) -> float:

    points = normalize_box(
        box
    )

    return float(
        cv2.contourArea(
            points
        )
    )


def box_iou(
    box_a,
    box_b
) -> float:
    """
    Approximate IoU using bounding rectangles.
    """

    a = normalize_box(
        box_a
    )

    b = normalize_box(
        box_b
    )

    ax1 = float(
        np.min(a[:, 0])
    )

    ay1 = float(
        np.min(a[:, 1])
    )

    ax2 = float(
        np.max(a[:, 0])
    )

    ay2 = float(
        np.max(a[:, 1])
    )

    bx1 = float(
        np.min(b[:, 0])
    )

    by1 = float(
        np.min(b[:, 1])
    )

    bx2 = float(
        np.max(b[:, 0])
    )

    by2 = float(
        np.max(b[:, 1])
    )

    intersection_x1 = max(
        ax1,
        bx1
    )

    intersection_y1 = max(
        ay1,
        by1
    )

    intersection_x2 = min(
        ax2,
        bx2
    )

    intersection_y2 = min(
        ay2,
        by2
    )

    intersection_width = max(
        0,
        intersection_x2
        -
        intersection_x1
    )

    intersection_height = max(
        0,
        intersection_y2
        -
        intersection_y1
    )

    intersection = (
        intersection_width
        *
        intersection_height
    )

    area_a = max(
        0,
        (ax2 - ax1)
        *
        (ay2 - ay1)
    )

    area_b = max(
        0,
        (bx2 - bx1)
        *
        (by2 - by1)
    )

    union = (
        area_a
        +
        area_b
        -
        intersection
    )

    if union <= 0:
        return 0.0

    return float(
        intersection / union
    )


# ============================================================
# MAP BOX FROM CROP TO ORIGINAL
# ============================================================

def transform_box_from_crop(
    box,
    offset_x: float,
    offset_y: float,
    scale_x: float = 1.0,
    scale_y: float = 1.0
) -> List[List[float]]:
    """
    Convert OCR coordinates from a resized crop back into
    original-image coordinates.

    This fixes one of the biggest problems in the previous
    implementation.
    """

    points = normalize_box(
        box
    )

    result = []

    for x, y in points:

        original_x = (
            x / scale_x
            + offset_x
        )

        original_y = (
            y / scale_y
            + offset_y
        )

        result.append(
            [
                float(original_x),
                float(original_y)
            ]
        )

    return result


# ============================================================
# ROTATION BOX TRANSFORMATION
# ============================================================

def inverse_rotate_box(
    box,
    original_shape,
    angle: int
) -> List[List[float]]:
    """
    Convert a bounding box from rotated-image coordinates
    back into original-image coordinates.
    """

    points = normalize_box(
        box
    )

    height, width = original_shape[:2]

    transformed = []

    for x, y in points:

        if angle == 90:

            # Rotated clockwise.
            original_x = y
            original_y = (
                height
                - 1
                - x
            )

        elif angle == 180:

            original_x = (
                width
                - 1
                - x
            )

            original_y = (
                height
                - 1
                - y
            )

        elif angle == 270:

            original_x = (
                width
                - 1
                - y
            )

            original_y = x

        else:

            original_x = x
            original_y = y

        transformed.append(
            [
                float(original_x),
                float(original_y)
            ]
        )

    return transformed


# ============================================================
# OCR ENGINE
# ============================================================

def run_ocr(
    image: np.ndarray,
    source: str,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    original_shape=None,
    rotation_angle: int = 0
) -> List[Dict[str, Any]]:
    """
    Run EasyOCR and correctly map boxes to the original image.
    """

    image = ensure_bgr(
        image
    )

    if original_shape is None:
        original_shape = image.shape

    try:

        results = reader.readtext(
            image,
            detail=1,
            paragraph=False,

            text_threshold=TEXT_THRESHOLD,
            low_text=LOW_TEXT,
            link_threshold=LINK_THRESHOLD,

            width_ths=WIDTH_THRESHOLD,
            height_ths=HEIGHT_THRESHOLD,

            mag_ratio=1.0,

            decoder="greedy"
        )

    except Exception as error:

        print(
            f"OCR error [{source}]: {error}"
        )

        return []

    detections = []

    for result in results:

        if not isinstance(
            result,
            (list, tuple)
        ):
            continue

        if len(result) != 3:
            continue

        box, text, confidence = result

        try:

            confidence = float(
                confidence
            )

        except (
            TypeError,
            ValueError
        ):

            continue

        text = repair_field_text(
            text
        )

        if not is_valid_text(
            text
        ):
            continue

        if confidence < MIN_CONFIDENCE:
            continue

        # Crop coordinates -> original image.
        mapped_box = transform_box_from_crop(
            box,
            offset_x,
            offset_y,
            scale_x,
            scale_y
        )

        # Rotated coordinates -> original image.
        if rotation_angle != 0:

            mapped_box = inverse_rotate_box(
                mapped_box,
                original_shape,
                rotation_angle
            )

        detections.append(
            {
                "text": text,
                "confidence": confidence,
                "box": mapped_box,
                "source": source,
            }
        )

    return detections


# ============================================================
# TEXT SIMILARITY
# ============================================================

def text_similarity(
    text_a: str,
    text_b: str
) -> float:

    a = re.sub(
        r"[^a-z0-9]",
        "",
        text_a.lower()
    )

    b = re.sub(
        r"[^a-z0-9]",
        "",
        text_b.lower()
    )

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


# ============================================================
# SPATIAL DISTANCE
# ============================================================

def spatial_distance(
    box_a,
    box_b
) -> float:

    ax, ay = box_center(
        box_a
    )

    bx, by = box_center(
        box_b
    )

    return math.sqrt(
        (ax - bx) ** 2
        +
        (ay - by) ** 2
    )


def spatially_close(
    box_a,
    box_b,
    image_shape
) -> bool:

    height, width = image_shape[:2]

    threshold = (
        max(
            width,
            height
        )
        *
        SPATIAL_DISTANCE_RATIO
    )

    return (
        spatial_distance(
            box_a,
            box_b
        )
        <= threshold
    )


# ============================================================
# DUPLICATE MERGING
# ============================================================

def merge_duplicates(
    detections: List[Dict[str, Any]],
    image_shape
) -> List[Dict[str, Any]]:
    """
    Merge repeated OCR detections.

    Multiple OCR passes may read the same text:

        original -> MRP
        CLAHE    -> MRP
        tile     -> MRP
        rotated  -> MRP

    This function keeps the strongest useful detection.
    """

    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda item: (
            item["confidence"]
        ),
        reverse=True
    )

    merged = []

    for item in detections:

        duplicate_index = None

        for index, existing in enumerate(
            merged
        ):

            similarity = text_similarity(
                item["text"],
                existing["text"]
            )

            iou = box_iou(
                item["box"],
                existing["box"]
            )

            close = spatially_close(
                item["box"],
                existing["box"],
                image_shape
            )

            # Strong exact/similar text.
            if similarity >= TEXT_SIMILARITY_THRESHOLD:

                if (
                    iou >= 0.20
                    or close
                ):

                    duplicate_index = index
                    break

            # Different OCR spelling in same place.
            if (
                close
                and
                similarity >=
                NEAR_TEXT_SIMILARITY_THRESHOLD
            ):

                duplicate_index = index
                break

        if duplicate_index is None:

            merged.append(
                item
            )

        else:

            existing = merged[
                duplicate_index
            ]

            # Keep stronger confidence.
            if item["confidence"] > existing[
                "confidence"
            ]:

                merged[
                    duplicate_index
                ] = item

    return merged


# ============================================================
# CREATE TILES
# ============================================================

def create_tiles(
    image: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Create overlapping tiles AND preserve their original
    coordinates.

    This is critical for correct bounding boxes.
    """

    image = ensure_bgr(
        image
    )

    height, width = image.shape[:2]

    tiles = []

    tile_width = int(
        width / TILE_COLS
    )

    tile_height = int(
        height / TILE_ROWS
    )

    for row in range(
        TILE_ROWS
    ):

        for col in range(
            TILE_COLS
        ):

            x1 = int(
                col
                * tile_width
                -
                tile_width
                * TILE_OVERLAP
            )

            y1 = int(
                row
                * tile_height
                -
                tile_height
                * TILE_OVERLAP
            )

            x2 = int(
                (col + 1)
                * tile_width
                +
                tile_width
                * TILE_OVERLAP
            )

            y2 = int(
                (row + 1)
                * tile_height
                +
                tile_height
                * TILE_OVERLAP
            )

            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                width,
                x2
            )

            y2 = min(
                height,
                y2
            )

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            tiles.append(
                {
                    "name":
                        f"tile_{row}_{col}",

                    "image":
                        crop,

                    "x":
                        x1,

                    "y":
                        y1,

                    "width":
                        x2 - x1,

                    "height":
                        y2 - y1,
                }
            )

    return tiles[
        :MAX_TILES
    ]


# ============================================================
# SPECIAL REGIONS
# ============================================================

def create_special_regions(
    image: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Create high-resolution regions where small regulatory
    declarations are commonly found.

    IMPORTANT:
    These are not assumed to contain a specific field.
    They simply increase OCR recall.
    """

    image = ensure_bgr(
        image
    )

    height, width = image.shape[:2]

    regions = []

    # --------------------------------------------------------
    # Bottom strip
    # --------------------------------------------------------

    bottom_y = int(
        height * 0.55
    )

    regions.append(
        {
            "name": "bottom_region",

            "image":
                image[
                    bottom_y:height,
                    0:width
                ],

            "x": 0,
            "y": bottom_y,
        }
    )

    # --------------------------------------------------------
    # Top strip
    # --------------------------------------------------------

    top_y = int(
        height * 0.45
    )

    regions.append(
        {
            "name": "top_region",

            "image":
                image[
                    0:top_y,
                    0:width
                ],

            "x": 0,
            "y": 0,
        }
    )

    # --------------------------------------------------------
    # Left side
    # --------------------------------------------------------

    left_x = int(
        width * 0.50
    )

    regions.append(
        {
            "name": "left_region",

            "image":
                image[
                    0:height,
                    0:left_x
                ],

            "x": 0,
            "y": 0,
        }
    )

    # --------------------------------------------------------
    # Right side
    # --------------------------------------------------------

    regions.append(
        {
            "name": "right_region",

            "image":
                image[
                    0:height,
                    left_x:width
                ],

            "x": left_x,
            "y": 0,
        }
    )

    return [
        region
        for region in regions
        if region["image"].size > 0
    ]


# ============================================================
# ENLARGE CROP
# ============================================================

def enlarge_region(
    image: np.ndarray,
    factor: float = 1.6
) -> Tuple[np.ndarray, float, float]:

    image = ensure_bgr(
        image
    )

    if factor <= 1:
        return (
            image,
            1.0,
            1.0
        )

    enlarged = cv2.resize(
        image,
        None,
        fx=factor,
        fy=factor,
        interpolation=cv2.INTER_CUBIC
    )

    return (
        enlarged,
        factor,
        factor
    )


# ============================================================
# SORT DETECTIONS
# ============================================================

def sort_detections(
    detections: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Sort text into approximate reading order.

    Uses dynamic line grouping.
    """

    if not detections:
        return []

    prepared = []

    for item in detections:

        x, y = box_center(
            item["box"]
        )

        _, height = box_size(
            item["box"]
        )

        prepared.append(
            {
                "item": item,
                "x": x,
                "y": y,
                "height": max(
                    height,
                    8
                )
            }
        )

    prepared.sort(
        key=lambda item:
        item["y"]
    )

    lines = []

    for entry in prepared:

        y = entry["y"]
        height = entry["height"]

        tolerance = max(
            10,
            height * 0.60
        )

        best_line = None
        best_difference = float(
            "inf"
        )

        for line in lines:

            difference = abs(
                y - line["y"]
            )

            if difference <= tolerance:

                if difference < best_difference:

                    best_difference = difference
                    best_line = line

        if best_line is None:

            lines.append(
                {
                    "y": y,
                    "items": [
                        entry
                    ]
                }
            )

        else:

            best_line["items"].append(
                entry
            )

            best_line["y"] = (
                best_line["y"]
                * 0.7
                +
                y
                * 0.3
            )

    lines.sort(
        key=lambda line:
        line["y"]
    )

    final = []

    for line in lines:

        line["items"].sort(
            key=lambda item:
            item["x"]
        )

        final.extend(
            item["item"]
            for item in line["items"]
        )

    return final


# ============================================================
# BUILD OCR TEXT
# ============================================================

def build_final_text(
    detections: List[Dict[str, Any]]
) -> str:
    """
    Build final OCR text.

    Duplicate text is removed, but line order is preserved.
    """

    if not detections:
        return ""

    lines = []

    seen = set()

    for item in detections:

        text = repair_field_text(
            item["text"]
        )

        if not is_valid_text(
            text
        ):
            continue

        key = re.sub(
            r"[^a-z0-9]",
            "",
            text.lower()
        )

        if not key:
            continue

        if key in seen:
            continue

        seen.add(
            key
        )

        lines.append(
            text
        )

    return "\n".join(
        lines
    )


# ============================================================
# ROTATE
# ============================================================

def rotate_image(
    image: np.ndarray,
    angle: int
) -> np.ndarray:

    image = ensure_bgr(
        image
    )

    if angle == 90:

        return cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE
        )

    if angle == 180:

        return cv2.rotate(
            image,
            cv2.ROTATE_180
        )

    if angle == 270:

        return cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )

    return image


# ============================================================
# DETECTION QUALITY
# ============================================================

def detection_quality_score(
    item: Dict[str, Any]
) -> float:
    """
    Calculate a ranking score.

    OCR confidence remains the primary signal.
    """

    score = float(
        item.get(
            "confidence",
            0.0
        )
    )

    source = str(
        item.get(
            "source",
            ""
        )
    ).lower()

    # Specialized regions can reveal small text.
    if "tile" in source:
        score += 0.015

    if "region" in source:
        score += 0.015

    if "bottom" in source:
        score += 0.020

    return score


# ============================================================
# MAIN OCR PIPELINE
# ============================================================

def extract_text_with_details(
    image: Image.Image
) -> Dict[str, Any]:
    """
    FULL HIGH-RECALL OCR PIPELINE.

    Pipeline:

        PIL
         ↓
        BGR safety
         ↓
        resize
         ↓
        quality analysis
         ↓
        perspective attempt
         ↓
        full-image OCR variants
         ↓
        special high-resolution regions
         ↓
        overlapping tiles
         ↓
        optional rotations
         ↓
        coordinate normalization
         ↓
        duplicate fusion
         ↓
        reading-order sorting
         ↓
        final OCR text
    """

    # ========================================================
    # 1. PIL -> BGR
    # ========================================================

    cv_image = pil_to_cv(
        image
    )

    # ========================================================
    # 2. RESIZE
    # ========================================================

    cv_image = resize_for_ocr(
        cv_image
    )

    original_shape = cv_image.shape

    # ========================================================
    # 3. QUALITY
    # ========================================================

    quality = get_image_quality(
        cv_image
    )

    all_detections = []

    # ========================================================
    # 4. PERSPECTIVE-CORRECTED VERSION
    # ========================================================

    corrected = try_document_correction(
        cv_image
    )

    corrected_is_different = (
        corrected.shape[:2]
        !=
        cv_image.shape[:2]
    )

    # Only use corrected image if it is genuinely different.
    if corrected_is_different:

        for source, variant in create_variants(
            corrected
        )[:3]:

            detections = run_ocr(
                variant,
                "perspective_" + source
            )

            all_detections.extend(
                detections
            )

    # ========================================================
    # 5. FULL IMAGE OCR
    # ========================================================

    full_variants = create_variants(
        cv_image
    )

    # High-value variants first.
    for source, variant in full_variants:

        detections = run_ocr(
            variant,
            source
        )

        all_detections.extend(
            detections
        )

    # ========================================================
    # 6. SPECIAL REGIONS
    # ========================================================

    if ENABLE_SPECIAL_REGIONS:

        regions = create_special_regions(
            cv_image
        )

        for region in regions:

            region_image = region[
                "image"
            ]

            enlarged, sx, sy = (
                enlarge_region(
                    region_image,
                    factor=1.65
                )
            )

            # Original enlarged region.
            detections = run_ocr(
                enlarged,

                region["name"],

                offset_x=region["x"],
                offset_y=region["y"],

                scale_x=sx,
                scale_y=sy,

                original_shape=original_shape
            )

            all_detections.extend(
                detections
            )

            # Enhanced enlarged region.
            gray = to_gray(
                enlarged
            )

            enhanced = clahe(
                gray
            )

            enhanced = sharpen(
                enhanced
            )

            enhanced_bgr = cv2.cvtColor(
                enhanced,
                cv2.COLOR_GRAY2BGR
            )

            detections = run_ocr(
                enhanced_bgr,

                region["name"]
                + "_enhanced",

                offset_x=region["x"],
                offset_y=region["y"],

                scale_x=sx,
                scale_y=sy,

                original_shape=original_shape
            )

            all_detections.extend(
                detections
            )

    # ========================================================
    # 7. OVERLAPPING TILES
    # ========================================================

    tiles = create_tiles(
        cv_image
    )

    for tile in tiles:

        tile_image = tile[
            "image"
        ]

        enlarged, sx, sy = (
            enlarge_region(
                tile_image,
                factor=1.55
            )
        )

        # ----------------------------------------------------
        # Tile original.
        # ----------------------------------------------------

        detections = run_ocr(
            enlarged,

            tile["name"],

            offset_x=tile["x"],
            offset_y=tile["y"],

            scale_x=sx,
            scale_y=sy,

            original_shape=original_shape
        )

        all_detections.extend(
            detections
        )

        # ----------------------------------------------------
        # Tile enhanced.
        # ----------------------------------------------------

        gray = to_gray(
            enlarged
        )

        enhanced = clahe(
            gray
        )

        enhanced = sharpen(
            enhanced
        )

        enhanced_bgr = cv2.cvtColor(
            enhanced,
            cv2.COLOR_GRAY2BGR
        )

        detections = run_ocr(
            enhanced_bgr,

            tile["name"]
            + "_enhanced",

            offset_x=tile["x"],
            offset_y=tile["y"],

            scale_x=sx,
            scale_y=sy,

            original_shape=original_shape
        )

        all_detections.extend(
            detections
        )

    # ========================================================
    # 8. ROTATION PASSES
    # ========================================================

    if ENABLE_ROTATION_PASSES:

        for angle in (
            90,
            180,
            270
        ):

            rotated = rotate_image(
                cv_image,
                angle
            )

            # Use only strongest preprocessing variant
            # for rotation to keep CPU usage reasonable.
            detections = run_ocr(
                rotated,

                f"rotation_{angle}",

                original_shape=original_shape,

                rotation_angle=angle
            )

            all_detections.extend(
                detections
            )

    # ========================================================
    # 9. REMOVE INVALID
    # ========================================================

    cleaned = []

    for item in all_detections:

        text = normalize_text(
            item["text"]
        )

        if not is_valid_text(
            text
        ):
            continue

        if item["confidence"] < MIN_CONFIDENCE:
            continue

        item["text"] = repair_field_text(
            text
        )

        cleaned.append(
            item
        )

    # ========================================================
    # 10. MERGE DUPLICATES
    # ========================================================

    merged = merge_duplicates(
        cleaned,
        original_shape
    )

    # ========================================================
    # 11. SORT
    # ========================================================

    sorted_detections = sort_detections(
        merged
    )

    # ========================================================
    # 12. FINAL TEXT
    # ========================================================

    final_text = build_final_text(
        sorted_detections
    )

    # ========================================================
    # 13. CONFIDENCE
    # ========================================================

    if sorted_detections:

        average_confidence = (
            sum(
                item["confidence"]
                for item in sorted_detections
            )
            /
            len(
                sorted_detections
            )
        )

        maximum_confidence = max(
            item["confidence"]
            for item in sorted_detections
        )

    else:

        average_confidence = 0.0
        maximum_confidence = 0.0

    # ========================================================
    # 14. FIELD HINTS
    # ========================================================

    field_hints = detect_field_hints(
        final_text
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "text":
            final_text,

        "detections":
            sorted_detections,

        "total_detections":
            len(
                sorted_detections
            ),

        "average_confidence":
            round(
                average_confidence,
                4
            ),

        "average_confidence_percent":
            round(
                average_confidence * 100,
                1
            ),

        "maximum_confidence":
            round(
                maximum_confidence,
                4
            ),

        "quality":
            quality,

        "field_hints":
            field_hints
    }


# ============================================================
# FIELD HINT DETECTION
# ============================================================

def detect_field_hints(
    text: str
) -> Dict[str, bool]:
    """
    Detect whether important packaged-commodity fields
    appear somewhere in OCR output.

    This does NOT extract the value.
    It simply tells the AI extraction layer what was detected.
    """

    normalized = text.lower()

    return {

        "mrp":
            bool(
                re.search(
                    r"\bmrp\b|maximum\s+retail\s+price",
                    normalized
                )
            ),

        "net_quantity":
            bool(
                re.search(
                    r"net\s+(?:quantity|qty|weight|wt|content)",
                    normalized
                )
            ),

        "manufacturer":
            bool(
                re.search(
                    r"manufactured\s+by|"
                    r"manufactured\s*&\s*marketed\s*by|"
                    r"packed\s+by|"
                    r"marketed\s+by",
                    normalized
                )
            ),

        "batch":
            bool(
                re.search(
                    r"\bbatch\b|\blot\s+no",
                    normalized
                )
            ),

        "fssai":
            bool(
                re.search(
                    r"\bfssai\b",
                    normalized
                )
            ),

        "expiry":
            bool(
                re.search(
                    r"expiry|"
                    r"best\s+before|"
                    r"use\s+by",
                    normalized
                )
            ),

        "manufacturing_date":
            bool(
                re.search(
                    r"date\s+of\s+manufacture|"
                    r"mfg\s+date|"
                    r"manufacturing\s+date",
                    normalized
                )
            ),

        "ingredients":
            bool(
                re.search(
                    r"\bingredients?\b",
                    normalized
                )
            ),

        "country_of_origin":
            bool(
                re.search(
                    r"country\s+of\s+origin|"
                    r"made\s+in",
                    normalized
                )
            ),

        "customer_care":
            bool(
                re.search(
                    r"customer\s+care|"
                    r"helpline|"
                    r"customer\s+service",
                    normalized
                )
            ),
    }


# ============================================================
# SIMPLE API
# ============================================================

def extract_text(
    image: Image.Image
) -> str:
    """
    Main function.

    Existing app.py can continue using:

        extracted_text = extract_text(image)
    """

    result = extract_text_with_details(
        image
    )

    return result[
        "text"
    ]


# ============================================================
# OCR CONFIDENCE API
# ============================================================

def get_ocr_confidence(
    image: Image.Image
) -> Dict[str, Any]:
    """
    Return OCR text and quality information.
    """

    result = extract_text_with_details(
        image
    )

    return {

        "text":
            result["text"],

        "average_confidence":
            result[
                "average_confidence"
            ],

        "average_confidence_percent":
            result[
                "average_confidence_percent"
            ],

        "maximum_confidence":
            result[
                "maximum_confidence"
            ],

        "total_detections":
            result[
                "total_detections"
            ],

        "quality":
            result[
                "quality"
            ],

        "field_hints":
            result[
                "field_hints"
            ]
    }


# ============================================================
# FULL DEBUG API
# ============================================================

def get_ocr_debug(
    image: Image.Image
) -> Dict[str, Any]:
    """
    Return everything produced by the OCR pipeline.
    """

    return extract_text_with_details(
        image
    )