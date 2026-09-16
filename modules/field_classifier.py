"""
============================================================
FIELD CLASSIFIER
============================================================

Purpose
-------
Determine what an OCR detection most likely represents.

Example:

    "MRP"
        -> MRP

    "Maximum Retail Price"
        -> MRP

    "399/-"
        -> possible MRP VALUE

    "Net Weight"
        -> NET QUANTITY

    "250 GM"
        -> possible NET QUANTITY VALUE

    "Batch No: NTZ26092"
        -> BATCH NUMBER

    "FSSAI Lic No. 12345678901234"
        -> FSSAI LICENSE

    "Manufactured By"
        -> MANUFACTURER

IMPORTANT
---------
This module does NOT assume a fixed package design.

It uses semantic + pattern + spatial evidence.

============================================================
"""

import math
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from .text_normalizer import (
    normalize_text,
    search_form,
    compact_text,
)


# ============================================================
# FIELD NAMES
# ============================================================

FIELD_PRODUCT_NAME = "product_name"
FIELD_MRP = "mrp"
FIELD_NET_QUANTITY = "net_quantity"
FIELD_MANUFACTURER = "manufacturer"
FIELD_ADDRESS = "address"

FIELD_BATCH = "batch_number"
FIELD_MFG_DATE = "manufacturing_date"
FIELD_EXPIRY_DATE = "expiry_date"

FIELD_FSSAI = "fssai_license"
FIELD_INGREDIENTS = "ingredients"
FIELD_COUNTRY = "country_of_origin"
FIELD_CUSTOMER_CARE = "customer_care"
FIELD_EMAIL = "email"

FIELD_UNKNOWN = "unknown"


ALL_FIELDS = [

    FIELD_PRODUCT_NAME,
    FIELD_MRP,
    FIELD_NET_QUANTITY,
    FIELD_MANUFACTURER,
    FIELD_ADDRESS,

    FIELD_BATCH,
    FIELD_MFG_DATE,
    FIELD_EXPIRY_DATE,

    FIELD_FSSAI,
    FIELD_INGREDIENTS,
    FIELD_COUNTRY,
    FIELD_CUSTOMER_CARE,
    FIELD_EMAIL,
]


# ============================================================
# FIELD KEYWORDS
# ============================================================

FIELD_KEYWORDS = {

    FIELD_MRP: [

        "mrp",
        "maximum retail price",
        "max retail price",
        "max. retail price",
        "retail price",
        "maximum retail",
    ],

    FIELD_NET_QUANTITY: [

        "net quantity",
        "net qty",
        "net weight",
        "net wt",
        "net content",
        "quantity",
        "weight",
        "contents",
        "content",
    ],

    FIELD_MANUFACTURER: [

        "manufactured by",
        "manufactured and marketed by",
        "manufactured & marketed by",
        "packed by",
        "packed and marketed by",
        "packed & marketed by",
        "marketed by",
        "mfd by",
        "mfg by",
        "packer",
        "manufacturer",
    ],

    FIELD_ADDRESS: [

        "address",
        "registered office",
        "corporate office",
        "manufacturing address",
        "factory address",
        "unit address",
    ],

    FIELD_BATCH: [

        "batch",
        "batch no",
        "batch number",
        "batch code",
        "lot no",
        "lot number",
        "lot code",
    ],

    FIELD_MFG_DATE: [

        "manufacturing date",
        "manufacture date",
        "mfg date",
        "mfd date",
        "date of manufacture",
        "date of mfg",
        "date of mfd",
    ],

    FIELD_EXPIRY_DATE: [

        "expiry",
        "expiry date",
        "expiration",
        "expiration date",
        "use by",
        "best before",
        "best before end",
    ],

    FIELD_FSSAI: [

        "fssai",
        "fssai license",
        "fssai licence",
        "fssai lic",
        "food safety license",
        "food safety licence",
    ],

    FIELD_INGREDIENTS: [

        "ingredients",
        "ingredient",
        "contains",
        "composition",
    ],

    FIELD_COUNTRY: [

        "country of origin",
        "country of origin:",
        "made in",
        "manufactured in",
        "product of",
    ],

    FIELD_CUSTOMER_CARE: [

        "customer care",
        "customer service",
        "consumer care",
        "helpline",
        "toll free",
        "contact us",
    ],

    FIELD_EMAIL: [

        "email",
        "e-mail",
    ],
}


# ============================================================
# HEADING PATTERNS
# ============================================================

FIELD_REGEX = {

    FIELD_MRP: [
        r"\bmrp\b",
        r"maximum\s+retail\s+price",
        r"max(?:imum)?\.?\s+retail\s+price",
    ],

    FIELD_NET_QUANTITY: [
        r"net\s+(?:quantity|qty|weight|wt|content)",
    ],

    FIELD_MANUFACTURER: [
        r"manufactured\s+by",
        r"manufactured\s+(?:and|&)\s+marketed\s+by",
        r"packed\s+by",
        r"packed\s+(?:and|&)\s+marketed\s+by",
        r"marketed\s+by",
        r"\bmfd\.?\s+by\b",
        r"\bmfg\.?\s+by\b",
    ],

    FIELD_ADDRESS: [
        r"\baddress\b",
        r"registered\s+office",
        r"corporate\s+office",
        r"manufacturing\s+address",
    ],

    FIELD_BATCH: [
        r"\bbatch\b",
        r"batch\s+(?:no|number|code)",
        r"lot\s+(?:no|number|code)",
    ],

    FIELD_MFG_DATE: [
        r"manufacturing\s+date",
        r"manufacture\s+date",
        r"mfg\.?\s+date",
        r"mfd\.?\s+date",
        r"date\s+of\s+manufacture",
    ],

    FIELD_EXPIRY_DATE: [
        r"\bexpiry\b",
        r"expiry\s+date",
        r"expiration\s+date",
        r"\buse\s+by\b",
        r"best\s+before",
    ],

    FIELD_FSSAI: [
        r"\bfssai\b",
        r"food\s+safety\s+(?:license|licence)",
    ],

    FIELD_INGREDIENTS: [
        r"\bingredients?\b",
        r"\bcontains\b",
        r"\bcomposition\b",
    ],

    FIELD_COUNTRY: [
        r"country\s+of\s+origin",
        r"\bmade\s+in\b",
        r"product\s+of",
    ],

    FIELD_CUSTOMER_CARE: [
        r"customer\s+care",
        r"customer\s+service",
        r"consumer\s+care",
        r"helpline",
        r"toll\s*free",
    ],

    FIELD_EMAIL: [
        r"\bemail\b",
        r"\be-mail\b",
        r"[\w.+-]+@[\w-]+\.[\w.-]+",
    ],
}


# ============================================================
# VALUE PATTERNS
# ============================================================

MRP_VALUE_REGEX = re.compile(
    r"(?:₹|rs\.?|inr)?\s*"
    r"\d{1,7}(?:[.,]\d{1,2})?"
    r"\s*(?:/-)?",
    re.IGNORECASE
)


QUANTITY_VALUE_REGEX = re.compile(
    r"\d+(?:\.\d+)?\s*"
    r"(?:kg|kgs|g|gm|gms|mg|ml|l|ltr|litre|liter)\b",
    re.IGNORECASE
)


DATE_REGEX = re.compile(
    r"\b"
    r"(?:"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|"
    r"\d{1,2}[/-]\d{4}"
    r"|"
    r"\d{4}[/-]\d{1,2}"
    r"|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[a-z]*\.?\s+\d{2,4}"
    r")"
    r"\b",
    re.IGNORECASE
)


EMAIL_REGEX = re.compile(
    r"\b"
    r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
    r"\b"
)


FSSAI_NUMBER_REGEX = re.compile(
    r"\b\d{10,14}\b"
)


PIN_REGEX = re.compile(
    r"\b[1-9]\d{5}\b"
)


PHONE_REGEX = re.compile(
    r"(?:\+91[\s-]?)?"
    r"[6-9]\d{9}\b"
)


# ============================================================
# BASIC GEOMETRY
# ============================================================

def box_center(
    box: Any
) -> Tuple[float, float]:

    try:

        points = [
            (
                float(point[0]),
                float(point[1])
            )
            for point in box
        ]

    except Exception:

        return (
            0.0,
            0.0
        )

    if not points:
        return (
            0.0,
            0.0
        )

    x = sum(
        point[0]
        for point in points
    ) / len(points)

    y = sum(
        point[1]
        for point in points
    ) / len(points)

    return (
        x,
        y
    )


def box_size(
    box: Any
) -> Tuple[float, float]:

    try:

        xs = [
            float(point[0])
            for point in box
        ]

        ys = [
            float(point[1])
            for point in box
        ]

    except Exception:

        return (
            0.0,
            0.0
        )

    if not xs or not ys:

        return (
            0.0,
            0.0
        )

    return (
        max(xs) - min(xs),
        max(ys) - min(ys)
    )


def distance_between_boxes(
    box_a: Any,
    box_b: Any
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


# ============================================================
# SIMILARITY
# ============================================================

def similarity(
    a: str,
    b: str
) -> float:

    a = compact_text(
        a
    )

    b = compact_text(
        b
    )

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


# ============================================================
# KEYWORD SCORE
# ============================================================

def keyword_score(
    text: str,
    field: str
) -> float:
    """
    Semantic keyword score.

    Exact heading match gets strongest score.
    """

    normalized = search_form(
        text
    )

    if not normalized:
        return 0.0

    score = 0.0

    keywords = FIELD_KEYWORDS.get(
        field,
        []
    )

    for keyword in keywords:

        key = search_form(
            keyword
        )

        if normalized == key:

            score = max(
                score,
                1.0
            )

        elif key in normalized:

            score = max(
                score,
                0.85
            )

        else:

            # Token-level fuzzy matching.
            current_similarity = similarity(
                normalized,
                key
            )

            if current_similarity >= 0.90:

                score = max(
                    score,
                    0.80
                )

            elif current_similarity >= 0.78:

                score = max(
                    score,
                    0.60
                )

    return score


# ============================================================
# REGEX SCORE
# ============================================================

def regex_score(
    text: str,
    field: str
) -> float:

    normalized = search_form(
        text
    )

    patterns = FIELD_REGEX.get(
        field,
        []
    )

    for pattern in patterns:

        try:

            if re.search(
                pattern,
                normalized,
                re.IGNORECASE
            ):

                return 1.0

        except re.error:
            continue

    return 0.0


# ============================================================
# VALUE FORMAT SCORE
# ============================================================

def value_format_score(
    text: str,
    field: str
) -> float:

    if not text:
        return 0.0

    # --------------------------------------------------------
    # MRP
    # --------------------------------------------------------

    if field == FIELD_MRP:

        if MRP_VALUE_REGEX.search(
            text
        ):

            return 0.85

    # --------------------------------------------------------
    # Quantity
    # --------------------------------------------------------

    if field == FIELD_NET_QUANTITY:

        if QUANTITY_VALUE_REGEX.search(
            text
        ):

            return 0.90

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    if field in (
        FIELD_MFG_DATE,
        FIELD_EXPIRY_DATE
    ):

        if DATE_REGEX.search(
            text
        ):

            return 0.75

    # --------------------------------------------------------
    # FSSAI
    # --------------------------------------------------------

    if field == FIELD_FSSAI:

        if (
            "fssai" in
            search_form(text)
        ):

            return 0.85

        if FSSAI_NUMBER_REGEX.search(
            text
        ):

            return 0.40

    # --------------------------------------------------------
    # Email
    # --------------------------------------------------------

    if field == FIELD_EMAIL:

        if EMAIL_REGEX.search(
            text
        ):

            return 1.0

    # --------------------------------------------------------
    # Address
    # --------------------------------------------------------

    if field == FIELD_ADDRESS:

        if PIN_REGEX.search(
            text
        ):

            return 0.90

    # --------------------------------------------------------
    # Customer care
    # --------------------------------------------------------

    if field == FIELD_CUSTOMER_CARE:

        if PHONE_REGEX.search(
            text
        ):

            return 0.70

    return 0.0


# ============================================================
# FIELD PRIORITY
# ============================================================

FIELD_PRIORS = {

    FIELD_MRP: 0.10,

    FIELD_NET_QUANTITY: 0.10,

    FIELD_MANUFACTURER: 0.08,

    FIELD_ADDRESS: 0.05,

    FIELD_BATCH: 0.04,

    FIELD_MFG_DATE: 0.04,

    FIELD_EXPIRY_DATE: 0.04,

    FIELD_FSSAI: 0.04,

    FIELD_INGREDIENTS: 0.03,

    FIELD_COUNTRY: 0.03,

    FIELD_CUSTOMER_CARE: 0.02,

    FIELD_EMAIL: 0.02,

    FIELD_PRODUCT_NAME: 0.02,
}


# ============================================================
# FIELD CLASSIFICATION
# ============================================================

def classify_text(
    text: str,
    detections: Optional[
        List[Dict[str, Any]]
    ] = None,
    index: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Classify one OCR text into possible fields.

    Returns ranked candidates.

    Example:

        [
            {
                "field": "mrp",
                "score": 0.94,
                "reasons": [...]
            }
        ]
    """

    text = normalize_text(
        text
    )

    if not text:

        return []

    candidates = []

    for field in ALL_FIELDS:

        key_score = keyword_score(
            text,
            field
        )

        pattern_score = regex_score(
            text,
            field
        )

        value_score = value_format_score(
            text,
            field
        )

        # ----------------------------------------------------
        # Combined score
        # ----------------------------------------------------

        score = (
            key_score * 0.50
            +
            pattern_score * 0.20
            +
            value_score * 0.20
            +
            FIELD_PRIORS.get(
                field,
                0.0
            )
            * 0.10
        )

        reasons = []

        if key_score > 0:

            reasons.append(
                "field keyword match"
            )

        if pattern_score > 0:

            reasons.append(
                "field pattern match"
            )

        if value_score > 0:

            reasons.append(
                "value format match"
            )

        if score >= 0.20:

            candidates.append(
                {
                    "field": field,
                    "score": round(
                        min(
                            score,
                            1.0
                        ),
                        4
                    ),
                    "reasons": reasons,
                }
            )

    candidates.sort(
        key=lambda item:
        item["score"],
        reverse=True
    )

    return candidates


# ============================================================
# CLASSIFY OCR DETECTIONS
# ============================================================

def classify_detections(
    detections: List[Dict[str, Any]],
    image_shape: Optional[
        Tuple[int, ...]
    ] = None
) -> List[Dict[str, Any]]:
    """
    Add field candidates to every OCR detection.

    IMPORTANT:
    We retain ALL OCR detections.

    We don't throw away ambiguous values such as:

        399/-
        250 GM
        12345678901234

    because later neighboring text can tell us their meaning.
    """

    result = []

    for index, detection in enumerate(
        detections
    ):

        if not isinstance(
            detection,
            dict
        ):
            continue

        text = normalize_text(
            detection.get(
                "text",
                ""
            )
        )

        if not text:
            continue

        candidates = classify_text(
            text,
            detections,
            index
        )

        item = dict(
            detection
        )

        item["text"] = text

        item["field_candidates"] = candidates

        if candidates:

            item["best_field"] = candidates[
                0
            ]["field"]

            item["field_score"] = candidates[
                0
            ]["score"]

        else:

            item["best_field"] = (
                FIELD_UNKNOWN
            )

            item["field_score"] = 0.0

        result.append(
            item
        )

    return result


# ============================================================
# FIND NEARBY DETECTIONS
# ============================================================

def nearby_detections(
    detection: Dict[str, Any],
    detections: List[Dict[str, Any]],
    max_distance_ratio: float = 0.12,
    image_shape: Optional[
        Tuple[int, ...]
    ] = None
) -> List[Dict[str, Any]]:
    """
    Find OCR detections near another detection.

    This is extremely important for labels.

    Example:

        MRP
        ₹399

    The value "₹399" may not contain the word MRP.

    Spatial relationship connects them.
    """

    if not image_shape:
        return []

    height, width = image_shape[:2]

    max_distance = (
        max(
            width,
            height
        )
        * max_distance_ratio
    )

    source_box = detection.get(
        "box"
    )

    if source_box is None:
        return []

    neighbors = []

    for other in detections:

        if other is detection:
            continue

        other_box = other.get(
            "box"
        )

        if other_box is None:
            continue

        distance = distance_between_boxes(
            source_box,
            other_box
        )

        if distance <= max_distance:

            neighbor = dict(
                other
            )

            neighbor["_distance"] = (
                distance
            )

            neighbors.append(
                neighbor
            )

    neighbors.sort(
        key=lambda item:
        item.get(
            "_distance",
            float("inf")
        )
    )

    return neighbors


# ============================================================
# VALUE -> HEADING RELATIONSHIP
# ============================================================

def relationship_score(
    value_detection: Dict[str, Any],
    heading_detection: Dict[str, Any],
    field: str,
    image_shape: Optional[
        Tuple[int, ...]
    ] = None
) -> float:
    """
    Determine whether a value belongs to a nearby heading.

    Strongest case:

        MRP
        ₹399

    or:

        Net Weight: 250 GM
    """

    if not image_shape:
        return 0.0

    value_box = value_detection.get(
        "box"
    )

    heading_box = heading_detection.get(
        "box"
    )

    if value_box is None or heading_box is None:
        return 0.0

    distance = distance_between_boxes(
        value_box,
        heading_box
    )

    height, width = image_shape[:2]

    max_distance = (
        max(
            width,
            height
        )
        * 0.15
    )

    if distance > max_distance:
        return 0.0

    distance_score = max(
        0.0,
        1.0
        -
        distance
        /
        max_distance
    )

    heading_text = heading_detection.get(
        "text",
        ""
    )

    semantic_score = max(
        keyword_score(
            heading_text,
            field
        ),
        regex_score(
            heading_text,
            field
        )
    )

    return (
        distance_score * 0.60
        +
        semantic_score * 0.40
    )


# ============================================================
# ASSIGN VALUE TO HEADING
# ============================================================

def find_field_value_candidates(
    detections: List[Dict[str, Any]],
    field: str,
    image_shape: Optional[
        Tuple[int, ...]
    ] = None
) -> List[Dict[str, Any]]:
    """
    Find values that could belong to a particular field.

    This is where the system starts understanding:

        MRP -> 399

        NET WEIGHT -> 250 GM

    rather than merely recognizing words.
    """

    if not detections:
        return []

    candidates = []

    for index, detection in enumerate(
        detections
    ):

        text = normalize_text(
            detection.get(
                "text",
                ""
            )
        )

        if not text:
            continue

        # ----------------------------------------------------
        # Direct value evidence.
        # ----------------------------------------------------

        format_score = value_format_score(
            text,
            field
        )

        direct_field_score = max(
            keyword_score(
                text,
                field
            ),
            regex_score(
                text,
                field
            )
        )

        # ----------------------------------------------------
        # Heading relationship.
        # ----------------------------------------------------

        relationship = 0.0

        if image_shape:

            neighbors = nearby_detections(
                detection,
                detections,
                image_shape=image_shape
            )

            for neighbor in neighbors:

                rel = relationship_score(
                    detection,
                    neighbor,
                    field,
                    image_shape
                )

                relationship = max(
                    relationship,
                    rel
                )

        # ----------------------------------------------------
        # Final candidate score.
        # ----------------------------------------------------

        score = (
            format_score * 0.40
            +
            direct_field_score * 0.20
            +
            relationship * 0.40
        )

        if score < 0.15:
            continue

        candidates.append(
            {
                "index": index,

                "text": text,

                "field": field,

                "score": round(
                    min(
                        score,
                        1.0
                    ),
                    4
                ),

                "format_score":
                    round(
                        format_score,
                        4
                    ),

                "relationship_score":
                    round(
                        relationship,
                        4
                    ),
            }
        )

    candidates.sort(
        key=lambda item:
        item["score"],
        reverse=True
    )

    return candidates


# ============================================================
# FULL FIELD MAP
# ============================================================

def build_field_map(
    detections: List[Dict[str, Any]],
    image_shape: Optional[
        Tuple[int, ...]
    ] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build candidate lists for every supported field.
    """

    classified = classify_detections(
        detections,
        image_shape
    )

    field_map = {
        field: []
        for field in ALL_FIELDS
    }

    # --------------------------------------------------------
    # Heading candidates.
    # --------------------------------------------------------

    for detection in classified:

        for candidate in detection.get(
            "field_candidates",
            []
        ):

            field = candidate[
                "field"
            ]

            if field not in field_map:
                continue

            field_map[
                field
            ].append(
                {
                    "type":
                        "heading_or_direct",

                    "text":
                        detection[
                            "text"
                        ],

                    "score":
                        candidate[
                            "score"
                        ],

                    "detection":
                        detection,
                }
            )

    # --------------------------------------------------------
    # Value candidates.
    # --------------------------------------------------------

    for field in ALL_FIELDS:

        value_candidates = (
            find_field_value_candidates(
                detections,
                field,
                image_shape
            )
        )

        for candidate in value_candidates:

            field_map[
                field
            ].append(
                {
                    "type":
                        "value",

                    "text":
                        candidate[
                            "text"
                        ],

                    "score":
                        candidate[
                            "score"
                        ],

                    "candidate":
                        candidate,
                }
            )

    # Highest score first.
    for field in field_map:

        field_map[
            field
        ].sort(
            key=lambda item:
            item["score"],
            reverse=True
        )

    return field_map


# ============================================================
# PRODUCT NAME HEURISTIC
# ============================================================

PRODUCT_NAME_IGNORE = [

    "nutrition facts",
    "ingredients",
    "mrp",
    "net weight",
    "net quantity",
    "manufactured by",
    "packed by",
    "marketed by",
    "batch",
    "fssai",
    "expiry",
    "best before",
    "customer care",
    "address",
    "email",
    "country of origin",
]


def looks_like_product_name(
    text: str
) -> float:
    """
    Estimate whether arbitrary OCR text is a product name.

    This is deliberately a heuristic, because product names
    have enormous visual and linguistic variation.
    """

    text = normalize_text(
        text
    )

    if len(text) < 3:
        return 0.0

    lowered = search_form(
        text
    )

    for ignored in PRODUCT_NAME_IGNORE:

        if ignored in lowered:

            return 0.0

    # Pure number is not a product name.
    if re.fullmatch(
        r"[\d\s.,:/()\-]+",
        text
    ):

        return 0.0

    letters = len(
        re.findall(
            r"[A-Za-z]",
            text
        )
    )

    numbers = len(
        re.findall(
            r"\d",
            text
        )
    )

    if letters == 0:
        return 0.0

    score = 0.30

    # Reasonable product-name length.
    if 4 <= len(text) <= 80:
        score += 0.20

    # Mostly alphabetic.
    if letters >= numbers:
        score += 0.20

    # Multi-word names are common.
    if len(
        text.split()
    ) >= 2:

        score += 0.15

    # Uppercase/prominent package text often indicates
    # product name, but is not mandatory.
    if text.isupper():
        score += 0.10

    return min(
        score,
        1.0
    )


# ============================================================
# ADD PRODUCT NAME CANDIDATES
# ============================================================

def add_product_name_candidates(
    field_map: Dict[str, List[Dict[str, Any]]],
    detections: List[Dict[str, Any]]
) -> None:

    for detection in detections:

        text = normalize_text(
            detection.get(
                "text",
                ""
            )
        )

        score = looks_like_product_name(
            text
        )

        if score <= 0:
            continue

        field_map[
            FIELD_PRODUCT_NAME
        ].append(
            {
                "type":
                    "product_name",

                "text":
                    text,

                "score":
                    round(
                        score,
                        4
                    ),

                "detection":
                    detection,
            }
        )

    field_map[
        FIELD_PRODUCT_NAME
    ].sort(
        key=lambda item:
        item["score"],
        reverse=True
    )


# ============================================================
# PUBLIC CLASSIFIER
# ============================================================

def classify_fields(
    detections: List[Dict[str, Any]],
    image_shape: Optional[
        Tuple[int, ...]
    ] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Main classifier API.

    Usage:

        field_map = classify_fields(
            detections,
            image.shape
        )

    Returns candidates for every field.
    """

    field_map = build_field_map(
        detections,
        image_shape
    )

    add_product_name_candidates(
        field_map,
        detections
    )

    return field_map


# ============================================================
# BEST FIELD
# ============================================================

def get_best_field(
    text: str
) -> Tuple[str, float]:

    candidates = classify_text(
        text
    )

    if not candidates:

        return (
            FIELD_UNKNOWN,
            0.0
        )

    best = candidates[0]

    return (
        best["field"],
        best["score"]
    )


# ============================================================
# PUBLIC EXPORT
# ============================================================

__all__ = [

    "FIELD_PRODUCT_NAME",
    "FIELD_MRP",
    "FIELD_NET_QUANTITY",
    "FIELD_MANUFACTURER",
    "FIELD_ADDRESS",

    "FIELD_BATCH",
    "FIELD_MFG_DATE",
    "FIELD_EXPIRY_DATE",

    "FIELD_FSSAI",
    "FIELD_INGREDIENTS",
    "FIELD_COUNTRY",
    "FIELD_CUSTOMER_CARE",
    "FIELD_EMAIL",

    "FIELD_UNKNOWN",

    "ALL_FIELDS",

    "classify_text",
    "classify_detections",
    "classify_fields",

    "build_field_map",

    "find_field_value_candidates",

    "nearby_detections",

    "get_best_field",
]