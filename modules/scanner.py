# ============================================================
# ADVANCED PACKAGE SCANNER
# Packaged Commodity Compliance Project
# ============================================================

import cv2
import numpy as np

from PIL import Image

from typing import Optional, Dict, Any, Tuple


# ============================================================
# CONFIGURATION
# ============================================================

MAX_WIDTH = 2200

# Outer package detection
MIN_OUTER_AREA_RATIO = 0.72
MIN_CANDIDATE_AREA_RATIO = 0.15
MIN_RECTANGLE_CONFIDENCE = 55.0

# Full-frame label fallback
# If the uploaded image is already a cropped label/panel,
# we don't need to detect another rectangle around it.
FULL_FRAME_MIN_WIDTH = 300
FULL_FRAME_MIN_HEIGHT = 200
FULL_FRAME_MIN_CONTRAST = 12.0
FULL_FRAME_MIN_EDGE_DENSITY = 0.008

# Image quality
BLUR_THRESHOLD = 70.0
DARK_THRESHOLD = 45.0
BRIGHT_THRESHOLD = 220.0

# Kept for compatibility
GLARE_THRESHOLD = 245

MAX_DESKEW_ANGLE = 3.0

PERSPECTIVE_MARGIN = 0.025

SCANNER_VERSION = "5.0-FULL-FRAME-SAFE"


# ============================================================
# BASIC IMAGE CONVERSION
# ============================================================

def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """
    Convert PIL image to OpenCV BGR.

    Handles:
        RGB
        RGBA
        grayscale
        palette images
    """

    if image is None:
        raise ValueError("Image cannot be None.")

    image = image.convert("RGB")

    rgb = np.asarray(image)

    if rgb.ndim != 3:
        raise ValueError(
            "Invalid PIL image format."
        )

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )


def bgr_to_pil(
    image: np.ndarray
) -> Image.Image:
    """
    Convert OpenCV BGR/grayscale image to PIL RGB.
    """

    if image is None:
        raise ValueError(
            "Image cannot be None."
        )

    if not isinstance(
        image,
        np.ndarray
    ):
        raise TypeError(
            "Expected numpy.ndarray."
        )

    if image.ndim == 2:

        return Image.fromarray(
            image.astype(np.uint8)
        ).convert("RGB")

    image = ensure_bgr(image)

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(rgb)


def ensure_bgr(
    image: np.ndarray
) -> np.ndarray:
    """
    Guarantee that an OpenCV image is
    3-channel BGR.

    This prevents OpenCV channel errors.
    """

    if image is None:
        raise ValueError(
            "Image cannot be None."
        )

    if not isinstance(
        image,
        np.ndarray
    ):
        raise TypeError(
            "Expected numpy.ndarray."
        )

    if image.ndim == 2:

        return cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    if image.ndim != 3:

        raise ValueError(
            "Unsupported image dimensions."
        )

    channels = image.shape[2]

    if channels == 1:

        return cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    if channels == 3:

        return image

    if channels == 4:

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2BGR
        )

    raise ValueError(
        f"Unsupported channel count: {channels}"
    )


# ============================================================
# RESOLUTION ANALYSIS
# ============================================================

def analyze_resolution(
    image: np.ndarray
) -> Dict[str, Any]:
    """
    Analyze image resolution.
    """

    image = ensure_bgr(image)

    height, width = image.shape[:2]

    megapixels = (
        width * height
    ) / 1_000_000

    warnings = []

    if width < 500 or height < 400:

        warnings.append(
            "Image resolution is low. "
            "Move closer to the package."
        )

    return {
        "width": width,
        "height": height,
        "megapixels": round(
            megapixels,
            2
        ),
        "sufficient": not warnings,
        "warnings": warnings
    }


# ============================================================
# RESIZE
# ============================================================

def resize_for_processing(
    image: np.ndarray,
    max_width: int = MAX_WIDTH
) -> np.ndarray:
    """
    Resize large images while preserving
    aspect ratio.
    """

    image = ensure_bgr(image)

    height, width = image.shape[:2]

    if width <= max_width:

        return image

    scale = (
        max_width /
        float(width)
    )

    new_width = int(
        width * scale
    )

    new_height = int(
        height * scale
    )

    return cv2.resize(
        image,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# IMAGE QUALITY
# ============================================================

def calculate_brightness(
    image: np.ndarray
) -> float:
    """
    Calculate average brightness.
    """

    image = ensure_bgr(image)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return float(
        np.mean(gray)
    )


def calculate_contrast(
    image: np.ndarray
) -> float:
    """
    Calculate grayscale contrast.
    """

    image = ensure_bgr(image)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return float(
        np.std(gray)
    )


def calculate_blur(
    image: np.ndarray
) -> float:
    """
    Estimate sharpness using
    Laplacian variance.

    Higher = sharper.
    """

    image = ensure_bgr(image)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )


# ============================================================
# IMPROVED GLARE DETECTION
# ============================================================

def detect_glare(
    image: np.ndarray
) -> float:
    """
    Estimate probable camera glare.

    IMPORTANT:

    A white printed label is NOT automatically glare.

    The old approach counted every very bright pixel,
    which could incorrectly report something like:

        Glare = 82%

    on a perfectly usable white label.

    This version looks for:
        - very bright pixels
        - low saturation
        - local brightness significantly above surroundings
    """

    image = ensure_bgr(image)

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]

    # Very bright and almost white.
    bright = (
        value >= 250
    )

    low_saturation = (
        saturation <= 18
    )

    candidate = (
        bright &
        low_saturation
    ).astype(
        np.uint8
    ) * 255

    # Remove tiny isolated pixels.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # Compare each pixel with its local background.
    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        15
    )

    local_excess = (
        gray.astype(np.float32)
        -
        background.astype(np.float32)
    )

    specular = (
        local_excess >= 18.0
    )

    glare_mask = (
        (candidate > 0)
        &
        specular
    )

    total_pixels = max(
        glare_mask.size,
        1
    )

    return float(
        np.count_nonzero(
            glare_mask
        )
        /
        total_pixels
        *
        100.0
    )


def detect_dark_regions(
    image: np.ndarray
) -> float:
    """
    Estimate percentage of very dark pixels.
    """

    image = ensure_bgr(image)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    dark_pixels = np.sum(
        gray <= 25
    )

    return float(
        dark_pixels /
        max(
            gray.size,
            1
        )
        *
        100
    )


def check_image_quality(
    image: np.ndarray
) -> Dict[str, Any]:
    """
    Complete image quality analysis.
    """

    image = ensure_bgr(image)

    resolution = analyze_resolution(
        image
    )

    brightness = calculate_brightness(
        image
    )

    contrast = calculate_contrast(
        image
    )

    blur_score = calculate_blur(
        image
    )

    glare = detect_glare(
        image
    )

    dark_ratio = detect_dark_regions(
        image
    )

    warnings = list(
        resolution["warnings"]
    )

    suggestions = []

    # Resolution
    if not resolution["sufficient"]:

        suggestions.append(
            "Move closer to the product."
        )

    # Brightness
    if brightness < DARK_THRESHOLD:

        warnings.append(
            "Image is too dark."
        )

        suggestions.append(
            "Increase lighting."
        )

    elif brightness > BRIGHT_THRESHOLD:

        warnings.append(
            "Image is too bright."
        )

        suggestions.append(
            "Reduce direct lighting."
        )

    # Blur
    if blur_score < BLUR_THRESHOLD:

        warnings.append(
            "Image appears blurry."
        )

        suggestions.append(
            "Hold the camera steady."
        )

    # Contrast
    if contrast < 25:

        warnings.append(
            "Image has low contrast."
        )

        suggestions.append(
            "Use better lighting."
        )

    # Glare
    if glare > 8:

        warnings.append(
            "Possible strong glare detected."
        )

        suggestions.append(
            "Tilt the package slightly "
            "to remove reflections."
        )

    # Dark regions
    if dark_ratio > 45:

        warnings.append(
            "Large dark regions detected."
        )

    # --------------------------------------------------------
    # Quality score
    # --------------------------------------------------------

    score = 100.0

    if brightness < DARK_THRESHOLD:
        score -= 20

    elif brightness > BRIGHT_THRESHOLD:
        score -= 15

    if blur_score < BLUR_THRESHOLD:
        score -= 30

    if contrast < 25:
        score -= 15

    if glare > 8:
        score -= 15

    if not resolution["sufficient"]:
        score -= 20

    score = max(
        0.0,
        min(
            100.0,
            score
        )
    )

    if score >= 80:

        status = "GOOD"

    elif score >= 55:

        status = "ACCEPTABLE"

    else:

        status = "POOR"

    return {

        "status":
            status,

        "score":
            round(
                score,
                1
            ),

        "brightness":
            round(
                brightness,
                2
            ),

        "contrast":
            round(
                contrast,
                2
            ),

        "blur_score":
            round(
                blur_score,
                2
            ),

        "sharpness":
            round(
                blur_score,
                2
            ),

        "glare_ratio":
            round(
                glare,
                2
            ),

        "glare":
            round(
                glare,
                2
            ),

        "dark_ratio":
            round(
                dark_ratio,
                2
            ),

        "warnings":
            warnings,

        "suggestions":
            suggestions,

        "resolution":
            resolution
    }


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def denoise_image(
    gray: np.ndarray
) -> np.ndarray:
    """
    Remove noise while preserving text edges.
    """

    if gray.ndim != 2:

        gray = cv2.cvtColor(
            ensure_bgr(gray),
            cv2.COLOR_BGR2GRAY
        )

    return cv2.fastNlMeansDenoising(
        gray,
        None,
        h=8,
        templateWindowSize=7,
        searchWindowSize=21
    )


def correct_illumination(
    gray: np.ndarray
) -> np.ndarray:
    """
    Reduce uneven illumination.
    """

    if gray.ndim != 2:

        gray = cv2.cvtColor(
            ensure_bgr(gray),
            cv2.COLOR_BGR2GRAY
        )

    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        25
    )

    background = np.maximum(
        background,
        1
    )

    normalized = cv2.divide(
        gray,
        background,
        scale=255
    )

    return cv2.normalize(
        normalized,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(
        np.uint8
    )


def enhance_contrast(
    gray: np.ndarray
) -> np.ndarray:
    """
    CLAHE local contrast enhancement.
    """

    if gray.ndim != 2:

        gray = cv2.cvtColor(
            ensure_bgr(gray),
            cv2.COLOR_BGR2GRAY
        )

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    return clahe.apply(
        gray
    )


def sharpen_image(
    image: np.ndarray
) -> np.ndarray:
    """
    Controlled sharpening.
    """

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        2
    )

    sharpened = cv2.addWeighted(
        image,
        1.35,
        blurred,
        -0.35,
        0
    )

    return np.clip(
        sharpened,
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# OCR IMAGE CREATION
# ============================================================

def create_ocr_image(
    image: np.ndarray
) -> np.ndarray:
    """
    Create main OCR image.

    Always returns 3-channel BGR.
    """

    bgr = ensure_bgr(
        image
    )

    gray = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2GRAY
    )

    gray = denoise_image(
        gray
    )

    gray = correct_illumination(
        gray
    )

    gray = enhance_contrast(
        gray
    )

    gray = sharpen_image(
        gray
    )

    return cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )


def create_ocr_variants(
    image: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Create multiple OCR variants.

    Every returned image is 3-channel BGR.
    """

    bgr = ensure_bgr(
        image
    )

    gray = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2GRAY
    )

    denoised = denoise_image(
        gray
    )

    illumination = correct_illumination(
        denoised
    )

    contrast = enhance_contrast(
        illumination
    )

    sharpened = sharpen_image(
        contrast
    )

    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    otsu = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )[1]

    inverted = cv2.bitwise_not(
        adaptive
    )

    return {

        "original":
            bgr.copy(),

        "enhanced":
            cv2.cvtColor(
                sharpened,
                cv2.COLOR_GRAY2BGR
            ),

        "adaptive":
            cv2.cvtColor(
                adaptive,
                cv2.COLOR_GRAY2BGR
            ),

        "otsu":
            cv2.cvtColor(
                otsu,
                cv2.COLOR_GRAY2BGR
            ),

        "inverted":
            cv2.cvtColor(
                inverted,
                cv2.COLOR_GRAY2BGR
            )
    }


# ============================================================
# EDGE DETECTION
# ============================================================

def detect_edges(
    image: np.ndarray
) -> np.ndarray:
    """
    Detect stable package boundaries.
    """

    image = ensure_bgr(
        image
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return edges


# ============================================================
# CORNER ORDERING
# ============================================================

def order_points(
    points: np.ndarray
) -> np.ndarray:
    """
    Order four corners as:

        top-left
        top-right
        bottom-right
        bottom-left
    """

    points = np.asarray(
        points,
        dtype=np.float32
    )

    if points.shape != (4, 2):

        raise ValueError(
            "Exactly four points are required."
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


# ============================================================
# RECTANGLE CANDIDATE ANALYSIS
# ============================================================

def candidate_properties(
    contour,
    image_shape: Tuple[int, ...]
) -> Optional[Dict[str, Any]]:
    """
    Analyze possible outer package rectangle.
    """

    image_height = image_shape[0]
    image_width = image_shape[1]

    image_area = (
        image_height *
        image_width
    )

    contour_area = cv2.contourArea(
        contour
    )

    if contour_area <= 0:

        return None

    area_ratio = (
        contour_area /
        float(image_area)
    )

    if (
        area_ratio <
        MIN_CANDIDATE_AREA_RATIO
    ):

        return None

    perimeter = cv2.arcLength(
        contour,
        True
    )

    if perimeter <= 0:

        return None

    polygon = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    if len(polygon) != 4:

        return None

    if not cv2.isContourConvex(
        polygon
    ):

        return None

    points = polygon.reshape(
        4,
        2
    ).astype(
        np.float32
    )

    x, y, w, h = cv2.boundingRect(
        polygon
    )

    if w <= 0 or h <= 0:

        return None

    bounding_area = (
        w * h
    )

    rectangularity = (
        contour_area /
        float(bounding_area)
    )

    if rectangularity < 0.60:

        return None

    left_gap = (
        x /
        max(
            image_width,
            1
        )
    )

    top_gap = (
        y /
        max(
            image_height,
            1
        )
    )

    right_gap = (
        image_width -
        (x + w)
    ) / max(
        image_width,
        1
    )

    bottom_gap = (
        image_height -
        (y + h)
    ) / max(
        image_height,
        1
    )

    border_gap = max(
        left_gap,
        top_gap,
        right_gap,
        bottom_gap
    )

    edge_score = max(
        0.0,
        1.0 -
        min(
            border_gap / 0.12,
            1.0
        )
    )

    area_score = min(
        area_ratio /
        MIN_OUTER_AREA_RATIO,
        1.0
    )

    rectangularity_score = min(
        rectangularity,
        1.0
    )

    score = (
        0.65 * area_score +
        0.20 * rectangularity_score +
        0.15 * edge_score
    )

    return {

        "points":
            points,

        "area_ratio":
            area_ratio,

        "rectangularity":
            rectangularity,

        "edge_score":
            edge_score,

        "score":
            score,

        "bbox":
            (
                x,
                y,
                w,
                h
            )
    }


# ============================================================
# FIND OUTER PACKAGE / FULL FRAME LABEL
# ============================================================

def find_outer_package(
    image: np.ndarray
) -> Tuple[
    Optional[np.ndarray],
    float,
    Dict[str, Any]
]:
    """
    Advanced label/package detection.

    Two valid detection modes exist:

    1. OUTER_RECTANGLE
       A strong package boundary was found.

    2. FULL_IMAGE_LABEL
       The uploaded image itself is treated as the label.

    This is important because a user may upload:

        - a cropped label
        - a declaration panel
        - a scanner crop
        - a close-up
        - a screenshot
        - a complete package already filling the frame
    """

    image = ensure_bgr(
        image
    )

    height, width = image.shape[:2]

    # ========================================================
    # STEP 1
    # Try to find an actual outer rectangle.
    # ========================================================

    try:

        edges = detect_edges(
            image
        )

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

    except cv2.error:

        contours = []

    candidates = []

    for contour in contours:

        try:

            props = candidate_properties(
                contour,
                image.shape
            )

            if props is not None:

                candidates.append(
                    props
                )

        except Exception:

            continue

    # ========================================================
    # STEP 2
    # Accept strong outer rectangle.
    # ========================================================

    if candidates:

        candidates.sort(
            key=lambda item: (
                item["score"],
                item["area_ratio"]
            ),
            reverse=True
        )

        best = candidates[0]

        confidence = (
            float(
                best["score"]
            )
            *
            100.0
        )

        if (
            best["area_ratio"]
            >= MIN_OUTER_AREA_RATIO
            and
            confidence
            >= MIN_RECTANGLE_CONFIDENCE
        ):

            return (

                best["points"],

                round(
                    confidence,
                    2
                ),

                {

                    "reason":
                        "Strong outer package "
                        "boundary detected.",

                    "detection_mode":
                        "OUTER_RECTANGLE",

                    "candidate_count":
                        len(candidates),

                    "area_ratio":
                        round(
                            best[
                                "area_ratio"
                            ] * 100.0,
                            2
                        ),

                    "rectangularity":
                        round(
                            best[
                                "rectangularity"
                            ] * 100.0,
                            2
                        ),

                    "edge_score":
                        round(
                            best[
                                "edge_score"
                            ] * 100.0,
                            2
                        )
                }
            )

    # ========================================================
    # STEP 3
    # FULL IMAGE LABEL FALLBACK
    # ========================================================

    # This is the important fix.
    #
    # If the user uploads a cropped label such as:
    #
    #     NET WEIGHT - 200g
    #
    # there may be no second rectangle to detect.
    #
    # Therefore the entire uploaded image becomes the
    # label region.

    if (
        width >= FULL_FRAME_MIN_WIDTH
        and
        height >= FULL_FRAME_MIN_HEIGHT
    ):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        contrast = float(
            np.std(gray)
        )

        small_edges = cv2.Canny(
            gray,
            50,
            150
        )

        edge_density = float(
            np.count_nonzero(
                small_edges
            )
            /
            max(
                small_edges.size,
                1
            )
        )

        if (
            contrast
            >= FULL_FRAME_MIN_CONTRAST
            or
            edge_density
            >= FULL_FRAME_MIN_EDGE_DENSITY
        ):

            return (

                None,

                88.0,

                {

                    "reason":
                        "The uploaded image is already "
                        "a usable label/panel. "
                        "Full image retained.",

                    "detection_mode":
                        "FULL_IMAGE_LABEL",

                    "candidate_count":
                        len(candidates),

                    "area_ratio":
                        100.0,

                    "contrast":
                        round(
                            contrast,
                            2
                        ),

                    "edge_density":
                        round(
                            edge_density,
                            4
                        )
                }
            )

    # ========================================================
    # STEP 4
    # Nothing confidently detected.
    # ========================================================

    return (

        None,

        0.0,

        {

            "reason":
                "No reliable label region detected.",

            "detection_mode":
                "UNKNOWN",

            "candidate_count":
                len(candidates),

            "area_ratio":
                0.0
        }
    )


# ============================================================
# EXPAND PACKAGE RECTANGLE
# ============================================================

def expand_rectangle(
    points: np.ndarray,
    image_shape: Tuple[int, ...],
    margin_ratio: float = PERSPECTIVE_MARGIN
) -> np.ndarray:
    """
    Slightly expand detected package rectangle.
    """

    height = image_shape[0]
    width = image_shape[1]

    points = order_points(
        points
    )

    center = np.mean(
        points,
        axis=0
    )

    expanded = (
        center
        +
        (
            points -
            center
        )
        *
        (
            1.0 +
            margin_ratio
        )
    )

    expanded[:, 0] = np.clip(
        expanded[:, 0],
        0,
        width - 1
    )

    expanded[:, 1] = np.clip(
        expanded[:, 1],
        0,
        height - 1
    )

    return expanded.astype(
        np.float32
    )


# ============================================================
# PERSPECTIVE CORRECTION
# ============================================================

def correct_perspective(
    image: np.ndarray,
    points: np.ndarray
) -> np.ndarray:
    """
    Correct perspective of package.
    """

    image = ensure_bgr(
        image
    )

    points = order_points(
        points
    )

    tl, tr, br, bl = points

    width_top = np.linalg.norm(
        tr - tl
    )

    width_bottom = np.linalg.norm(
        br - bl
    )

    height_left = np.linalg.norm(
        bl - tl
    )

    height_right = np.linalg.norm(
        br - tr
    )

    width = int(
        max(
            width_top,
            width_bottom
        )
    )

    height = int(
        max(
            height_left,
            height_right
        )
    )

    if (
        width < 100
        or
        height < 100
    ):

        return image

    width = min(
        width,
        MAX_WIDTH
    )

    height = min(
        height,
        int(
            MAX_WIDTH *
            1.5
        )
    )

    destination = np.array(
        [
            [0, 0],

            [
                width - 1,
                0
            ],

            [
                width - 1,
                height - 1
            ],

            [
                0,
                height - 1
            ]
        ],
        dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(
        points,
        destination
    )

    corrected = cv2.warpPerspective(
        image,
        matrix,
        (
            width,
            height
        ),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return ensure_bgr(
        corrected
    )


# ============================================================
# SAFE HOUGH LINE EXTRACTION
# ============================================================

def extract_hough_lines(
    lines: Optional[np.ndarray]
) -> np.ndarray:
    """
    Safely normalize OpenCV HoughLinesP output.

    OpenCV can return:

        (N, 1, 4)

    or:

        (N, 4)

    This function always returns:

        (N, 4)

    Therefore the scanner never relies on:

        lines[:, 0]
    """

    if lines is None:

        return np.empty(
            (0, 4),
            dtype=np.float32
        )

    try:

        array = np.asarray(
            lines
        )

    except Exception:

        return np.empty(
            (0, 4),
            dtype=np.float32
        )

    if array.size == 0:

        return np.empty(
            (0, 4),
            dtype=np.float32
        )

    flat = array.reshape(
        -1
    )

    if flat.size < 4:

        return np.empty(
            (0, 4),
            dtype=np.float32
        )

    usable_size = (
        flat.size // 4
    ) * 4

    flat = flat[
        :usable_size
    ]

    normalized = flat.reshape(
        -1,
        4
    )

    return normalized.astype(
        np.float32
    )


# ============================================================
# SAFE DESKEW
# ============================================================

def deskew_image(
    image: np.ndarray
) -> Tuple[
    np.ndarray,
    float
]:
    """
    Correct small rotation.

    Uses safe HoughLinesP parsing.
    """

    bgr = ensure_bgr(
        image
    )

    gray = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    min_dimension = min(
        gray.shape[:2]
    )

    min_line_length = max(
        80,
        min_dimension // 5
    )

    try:

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=80,
            minLineLength=min_line_length,
            maxLineGap=15
        )

    except cv2.error:

        return bgr, 0.0

    normalized_lines = (
        extract_hough_lines(
            lines
        )
    )

    if normalized_lines.shape[0] == 0:

        return bgr, 0.0

    angles = []

    for line in normalized_lines:

        if len(line) != 4:

            continue

        try:

            x1, y1, x2, y2 = map(
                float,
                line
            )

        except (
            TypeError,
            ValueError
        ):

            continue

        dx = x2 - x1
        dy = y2 - y1

        if abs(dx) < 1e-6:

            continue

        angle = float(
            np.degrees(
                np.arctan2(
                    dy,
                    dx
                )
            )
        )

        # Normalize around horizontal.
        if angle > 45:

            angle -= 90

        elif angle < -45:

            angle += 90

        if (
            abs(angle)
            <= MAX_DESKEW_ANGLE
        ):

            angles.append(
                angle
            )

    if len(angles) < 3:

        return bgr, 0.0

    angle = float(
        np.median(
            np.asarray(
                angles,
                dtype=np.float32
            )
        )
    )

    if abs(angle) < 0.5:

        return bgr, angle

    height, width = (
        bgr.shape[:2]
    )

    center = (
        width // 2,
        height // 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        bgr,
        matrix,
        (
            width,
            height
        ),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return (
        ensure_bgr(
            rotated
        ),
        angle
    )


# ============================================================
# MAIN SCANNER
# ============================================================

def process_scan(
    image: Image.Image
) -> Dict[str, Any]:
    """
    Complete intelligent scanner.

    Pipeline:

        Original image
             ↓
        Resize
             ↓
        Quality analysis
             ↓
        Outer package detection
             ↓
        Full-image fallback
             ↓
        Perspective correction
             ↓
        Safe deskew
             ↓
        OCR preprocessing
             ↓
        OCR variants

    IMPORTANT:

    A cropped label does NOT require an outer rectangle.
    """

    # --------------------------------------------------------
    # PIL → BGR
    # --------------------------------------------------------

    original = pil_to_bgr(
        image
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    original = resize_for_processing(
        original
    )

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    quality = check_image_quality(
        original
    )

    # --------------------------------------------------------
    # Start with complete image
    # --------------------------------------------------------

    working_image = (
        original.copy()
    )

    rectangle = None

    label_confidence = 0.0

    label_detected = False

    perspective_applied = False

    rectangle_info = {}

    # --------------------------------------------------------
    # Detect label
    # --------------------------------------------------------

    try:

        (
            rectangle,
            label_confidence,
            rectangle_info
        ) = find_outer_package(
            original
        )

    except Exception as error:

        # Even if detection fails, preserve the image.
        rectangle = None

        label_confidence = 0.0

        rectangle_info = {

            "reason":
                "Boundary detection failed. "
                "Full image retained.",

            "detection_mode":
                "FULL_IMAGE_LABEL",

            "candidate_count":
                0,

            "error":
                str(error)
        }

    detection_mode = (
        rectangle_info.get(
            "detection_mode",
            "UNKNOWN"
        )
    )

    # ========================================================
    # OUTER RECTANGLE
    # ========================================================

    if (
        rectangle is not None
        and
        detection_mode ==
        "OUTER_RECTANGLE"
    ):

        try:

            expanded = expand_rectangle(
                rectangle,
                original.shape
            )

            corrected = (
                correct_perspective(
                    original,
                    expanded
                )
            )

            original_area = (
                original.shape[0]
                *
                original.shape[1]
            )

            corrected_area = (
                corrected.shape[0]
                *
                corrected.shape[1]
            )

            transformed_ratio = (
                corrected_area
                /
                max(
                    original_area,
                    1
                )
            )

            if (
                transformed_ratio
                >= 0.55
            ):

                working_image = (
                    corrected
                )

                label_detected = True

                perspective_applied = True

            else:

                working_image = (
                    original.copy()
                )

                # The rectangle itself was detected,
                # so the label is still considered detected.
                label_detected = True

                perspective_applied = False

        except Exception as error:

            working_image = (
                original.copy()
            )

            # Detection succeeded; only perspective
            # correction failed.
            label_detected = True

            perspective_applied = False

            rectangle_info[
                "perspective_error"
            ] = str(error)

    # ========================================================
    # FULL IMAGE LABEL
    # ========================================================

    elif (
        detection_mode ==
        "FULL_IMAGE_LABEL"
    ):

        # ----------------------------------------------------
        # KEY FIX
        # ----------------------------------------------------
        #
        # The entire uploaded image is already the label.
        #
        # Do NOT:
        #
        #   - search for another rectangle
        #   - crop the image
        #   - perspective-correct it
        #   - mark detection as false
        #
        # This fixes:
        #
        #     Label Detection = 0%
        #
        # for close-up/cropped label scans.
        # ----------------------------------------------------

        working_image = (
            original.copy()
        )

        label_detected = True

        perspective_applied = False

        rectangle_info[
            "area_ratio"
        ] = 100.0

    # ========================================================
    # UNKNOWN
    # ========================================================

    else:

        # We still preserve the image so OCR can attempt
        # extraction instead of failing completely.

        working_image = (
            original.copy()
        )

        label_detected = False

        perspective_applied = False

    # ========================================================
    # SAFE DESKEW
    # ========================================================

    try:

        (
            working_image,
            deskew_angle
        ) = deskew_image(
            working_image
        )

    except Exception:

        deskew_angle = 0.0

        working_image = ensure_bgr(
            working_image
        )

    # ========================================================
    # OCR PREPROCESSING
    # ========================================================

    try:

        ocr_image = create_ocr_image(
            working_image
        )

    except Exception:

        ocr_image = ensure_bgr(
            working_image
        )

    # ========================================================
    # OCR VARIANTS
    # ========================================================

    try:

        ocr_variants = (
            create_ocr_variants(
                working_image
            )
        )

    except Exception:

        ocr_variants = {

            "original":
                ensure_bgr(
                    working_image
                )
        }

    # ========================================================
    # PIL OUTPUTS
    # ========================================================

    processed_image = (
        bgr_to_pil(
            working_image
        )
    )

    enhanced_pil = (
        bgr_to_pil(
            ocr_image
        )
    )

    # ========================================================
    # PROCESSING MODE
    # ========================================================

    if perspective_applied:

        processing_mode = (
            "OUTER_PACKAGE_CORRECTED"
        )

    elif (
        detection_mode ==
        "FULL_IMAGE_LABEL"
    ):

        processing_mode = (
            "FULL_IMAGE_LABEL"
        )

    else:

        processing_mode = (
            "FULL_IMAGE_PRESERVED"
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        "image":
            processed_image,

        "ocr_image":
            enhanced_pil,

        "ocr_variants":
            ocr_variants,

        "original_image":
            bgr_to_pil(
                original
            ),

        "ocr_source_image":
            bgr_to_pil(
                working_image
            ),

        "quality":
            quality,

        "label_detected":
            label_detected,

        "label_confidence":
            round(
                float(
                    label_confidence
                ),
                2
            ),

        "corners":
            rectangle,

        "rectangle_info":
            rectangle_info,

        "detection_mode":
            detection_mode,

        "perspective_applied":
            perspective_applied,

        "processing_mode":
            processing_mode,

        "orientation_angle":
            0.0,

        "deskew_angle":
            round(
                float(
                    deskew_angle
                ),
                3
            ),

        "scanner_version":
            SCANNER_VERSION
    }


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def preprocess_image(
    image: Image.Image
) -> Image.Image:
    """
    Compatibility function.

    Existing app.py code:

        processed_image =
            preprocess_image(image)

    continues to work.
    """

    result = process_scan(
        image
    )

    return result[
        "ocr_image"
    ]


# ============================================================
# SCAN SUMMARY
# ============================================================

def get_scan_summary(
    scan_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepare scan information for Streamlit.
    """

    quality = scan_result.get(
        "quality",
        {}
    )

    return {

        "quality_score":
            quality.get(
                "score",
                0
            ),

        "quality_status":
            quality.get(
                "status",
                "UNKNOWN"
            ),

        "brightness":
            quality.get(
                "brightness",
                0
            ),

        "contrast":
            quality.get(
                "contrast",
                0
            ),

        "sharpness":
            quality.get(
                "sharpness",
                0
            ),

        "glare":
            quality.get(
                "glare",
                0
            ),

        "label_detected":
            scan_result.get(
                "label_detected",
                False
            ),

        "label_confidence":
            scan_result.get(
                "label_confidence",
                0
            ),

        "detection_mode":
            scan_result.get(
                "detection_mode",
                "UNKNOWN"
            ),

        "perspective_applied":
            scan_result.get(
                "perspective_applied",
                False
            ),

        "processing_mode":
            scan_result.get(
                "processing_mode",
                "UNKNOWN"
            ),

        "deskew_angle":
            scan_result.get(
                "deskew_angle",
                0
            ),

        "scanner_version":
            scan_result.get(
                "scanner_version",
                SCANNER_VERSION
            ),

        "warnings":
            quality.get(
                "warnings",
                []
            ),

        "suggestions":
            quality.get(
                "suggestions",
                []
            )
    }


# ============================================================
# TEST / HEALTH CHECK
# ============================================================

def scanner_health_check() -> Dict[str, Any]:
    """
    Simple health check for debugging.
    """

    return {

        "status":
            "OK",

        "scanner_version":
            SCANNER_VERSION,

        "opencv_version":
            cv2.__version__,

        "numpy_version":
            np.__version__,

        "hough_parser":
            "SAFE",

        "full_frame_fallback":
            "ENABLED",

        "perspective_correction":
            "ENABLED",

        "improved_glare_detection":
            "ENABLED"
    }


# ============================================================
# END OF scanner.py
# ============================================================