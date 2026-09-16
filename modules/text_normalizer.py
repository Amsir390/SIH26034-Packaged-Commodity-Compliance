"""
============================================================
TEXT NORMALIZER
============================================================

Purpose
-------
Clean and normalize OCR output before field classification.

This module is deliberately conservative.

It can repair common OCR mistakes such as:

    M.R.P.       -> MRP
    M R P        -> MRP
    Net Wt.      -> NET WEIGHT
    Net Qty      -> NET QUANTITY
    FSSAI Lic No -> FSSAI LICENSE NUMBER
    Mfd By       -> MANUFACTURED BY

It also normalizes:

    Rs / RS / R5 -> Rs
    kg / kgs     -> KG
    gm / gms     -> G
    litre / ltr  -> L

IMPORTANT
---------
Do NOT aggressively replace arbitrary letters.

For example:

    ABC123

could be a legitimate batch number.

============================================================
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# CONSTANTS
# ============================================================

OCR_CHARACTER_REPLACEMENTS = {
    # Common OCR confusion in isolated contexts.
    "₹": "₹",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "\u00a0": " ",
}


# ============================================================
# BASIC NORMALIZATION
# ============================================================

def unicode_normalize(
    text: str
) -> str:
    """
    Normalize Unicode characters.

    NFKC helps convert visually equivalent Unicode forms
    into a consistent representation.
    """

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(text)
    )

    for old, new in OCR_CHARACTER_REPLACEMENTS.items():
        text = text.replace(
            old,
            new
        )

    return text


def normalize_whitespace(
    text: str
) -> str:
    """
    Normalize spaces without destroying useful text.
    """

    if not text:
        return ""

    text = text.replace(
        "\r",
        "\n"
    )

    text = text.replace(
        "\t",
        " "
    )

    # Remove spaces before punctuation.
    text = re.sub(
        r"\s+([,:;.!?])",
        r"\1",
        text
    )

    # Normalize repeated spaces.
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Normalize repeated blank lines.
    text = re.sub(
        r"\n[ \t]*\n+",
        "\n",
        text
    )

    return text.strip()


# ============================================================
# OCR HEADING REPAIR
# ============================================================

def normalize_mrp_heading(
    text: str
) -> str:

    pattern = (
        r"\b"
        r"M\s*\.?\s*R\s*\.?\s*P"
        r"\s*\.?"
        r"\b"
    )

    return re.sub(
        pattern,
        "MRP",
        text,
        flags=re.IGNORECASE
    )


def normalize_net_quantity_heading(
    text: str
) -> str:

    replacements = [

        (
            r"\bNET\s+WT\.?\b",
            "NET WEIGHT"
        ),

        (
            r"\bNET\s+W[T7]\.?\b",
            "NET WEIGHT"
        ),

        (
            r"\bNET\s+QTY\.?\b",
            "NET QUANTITY"
        ),

        (
            r"\bNET\s+QUANT[I1]TY\b",
            "NET QUANTITY"
        ),

        (
            r"\bNET\s+CONTENT[S5]?\b",
            "NET CONTENT"
        ),
    ]

    for pattern, replacement in replacements:

        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE
        )

    return text


def normalize_manufacturer_heading(
    text: str
) -> str:

    replacements = [

        (
            r"\bMFD\.?\s*BY\b",
            "MANUFACTURED BY"
        ),

        (
            r"\bMFG\.?\s*BY\b",
            "MANUFACTURED BY"
        ),

        (
            r"\bMANUFACTURED\s*&\s*MARKETED\s*BY\b",
            "MANUFACTURED AND MARKETED BY"
        ),

        (
            r"\bMANUFACTURED\s+AND\s+MARKETED\s+BY\b",
            "MANUFACTURED AND MARKETED BY"
        ),

        (
            r"\bPACKED\s*&\s*MARKETED\s*BY\b",
            "PACKED AND MARKETED BY"
        ),

        (
            r"\bPACKED\s+AND\s+MARKETED\s+BY\b",
            "PACKED AND MARKETED BY"
        ),

        (
            r"\bPKD\.?\s*BY\b",
            "PACKED BY"
        ),

        (
            r"\bMKT\.?\s*BY\b",
            "MARKETED BY"
        ),
    ]

    for pattern, replacement in replacements:

        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE
        )

    return text


def normalize_date_headings(
    text: str
) -> str:

    replacements = [

        (
            r"\bMFG\.?\s*DATE\b",
            "MANUFACTURING DATE"
        ),

        (
            r"\bMFD\.?\s*DATE\b",
            "MANUFACTURING DATE"
        ),

        (
            r"\bMFG\.?\b",
            "MANUFACTURING"
        ),

        (
            r"\bEXP\.?\s*DATE\b",
            "EXPIRY DATE"
        ),

        (
            r"\bEXP\.?\b",
            "EXPIRY"
        ),

        (
            r"\bUSE\s*BY\b",
            "USE BY"
        ),

        (
            r"\bBEST\s*BEFORE\b",
            "BEST BEFORE"
        ),
    ]

    for pattern, replacement in replacements:

        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE
        )

    return text


def normalize_fssai_heading(
    text: str
) -> str:

    replacements = [

        (
            r"\bFSSAI\s+LIC\.?\s*NO\.?\b",
            "FSSAI LICENSE NUMBER"
        ),

        (
            r"\bFSSAI\s+LICENSE\s+NO\.?\b",
            "FSSAI LICENSE NUMBER"
        ),

        (
            r"\bFSSAI\s+LICENCE\s+NO\.?\b",
            "FSSAI LICENSE NUMBER"
        ),

        (
            r"\bFSSAI\s+NO\.?\b",
            "FSSAI NUMBER"
        ),
    ]

    for pattern, replacement in replacements:

        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE
        )

    return text


# ============================================================
# UNIT NORMALIZATION
# ============================================================

def normalize_units(
    text: str
) -> str:
    """
    Normalize measurement units.

    Examples:

        250 gm  -> 250 G
        1 kgs   -> 1 KG
        500 ml  -> 500 ML
        1 litre -> 1 L

    Only standalone units are changed.
    """

    replacements = [

        (
            r"(?<=\d)\s*KGS?\b",
            " KG"
        ),

        (
            r"(?<=\d)\s*KG\b",
            " KG"
        ),

        (
            r"(?<=\d)\s*GMS?\b",
            " G"
        ),

        (
            r"(?<=\d)\s*MG\b",
            " MG"
        ),

        (
            r"(?<=\d)\s*ML\b",
            " ML"
        ),

        (
            r"(?<=\d)\s*LTRS?\b",
            " L"
        ),

        (
            r"(?<=\d)\s*LIT(?:RE|ER)\b",
            " L"
        ),

        (
            r"(?<=\d)\s*L\b",
            " L"
        ),
    ]

    for pattern, replacement in replacements:

        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE
        )

    # Remove accidental double spaces introduced above.
    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


# ============================================================
# CURRENCY NORMALIZATION
# ============================================================

def normalize_currency(
    text: str
) -> str:

    # OCR often reads Rs as R5.
    text = re.sub(
        r"\bR5\.?\b",
        "Rs",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\bRS\.?\b",
        "Rs",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\bINR\b",
        "Rs",
        text,
        flags=re.IGNORECASE
    )

    # Remove unnecessary space after currency.
    text = re.sub(
        r"\bRs\s+",
        "Rs ",
        text
    )

    return text


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_date_separators(
    text: str
) -> str:
    """
    Normalize common date separators.

    12.08.2026
    12/08/2026
    12-08-2026

    are preserved semantically.
    """

    text = re.sub(
        r"(?<=\d)\s*[|]\s*(?=\d)",
        "/",
        text
    )

    return text


# ============================================================
# COMMON OCR ARTIFACTS
# ============================================================

def remove_ocr_artifacts(
    text: str
) -> str:

    if not text:
        return ""

    # Remove obvious zero-width characters.
    text = re.sub(
        r"[\u200b-\u200f\u202a-\u202e]",
        "",
        text
    )

    # Replace common OCR separator artifacts.
    text = text.replace(
        "¦",
        "|"
    )

    text = text.replace(
        "—",
        "-"
    )

    text = text.replace(
        "–",
        "-"
    )

    return text


# ============================================================
# FULL NORMALIZATION
# ============================================================

def normalize_text(
    text: Any
) -> str:
    """
    Main normalization function.

    This is the function that other modules should use.
    """

    if text is None:
        return ""

    text = str(
        text
    )

    text = unicode_normalize(
        text
    )

    text = remove_ocr_artifacts(
        text
    )

    text = normalize_whitespace(
        text
    )

    text = normalize_mrp_heading(
        text
    )

    text = normalize_net_quantity_heading(
        text
    )

    text = normalize_manufacturer_heading(
        text
    )

    text = normalize_date_headings(
        text
    )

    text = normalize_fssai_heading(
        text
    )

    text = normalize_currency(
        text
    )

    text = normalize_units(
        text
    )

    text = normalize_date_separators(
        text
    )

    return normalize_whitespace(
        text
    )


# ============================================================
# COMPACT FORM
# ============================================================

def compact_text(
    text: str
) -> str:
    """
    Lowercase alphanumeric comparison form.

    Example:

        "M.R.P. ₹399"

    becomes approximately:

        "mrp399"
    """

    text = normalize_text(
        text
    )

    return re.sub(
        r"[^a-zA-Z0-9]+",
        "",
        text
    ).lower()


# ============================================================
# NORMALIZED SEARCH FORM
# ============================================================

def search_form(
    text: str
) -> str:
    """
    Search-friendly lowercase form.
    """

    text = normalize_text(
        text
    )

    return re.sub(
        r"\s+",
        " ",
        text.lower()
    ).strip()


# ============================================================
# LINE NORMALIZATION
# ============================================================

def normalize_lines(
    text: str
) -> List[str]:
    """
    Normalize OCR output while preserving lines.
    """

    if not text:
        return []

    text = str(
        text
    ).replace(
        "\r",
        "\n"
    )

    result = []

    for raw_line in text.splitlines():

        line = normalize_text(
            raw_line
        )

        if line:
            result.append(
                line
            )

    return result


# ============================================================
# OCR DETECTION NORMALIZATION
# ============================================================

def normalize_detection(
    detection: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Normalize an OCR detection while preserving metadata.

    Expected input:

        {
            "text": "...",
            "confidence": 0.95,
            "box": [...]
        }
    """

    result = dict(
        detection
    )

    result["text"] = normalize_text(
        detection.get(
            "text",
            ""
        )
    )

    try:

        result["confidence"] = float(
            detection.get(
                "confidence",
                0.0
            )
        )

    except (
        TypeError,
        ValueError
    ):

        result["confidence"] = 0.0

    return result


def normalize_detections(
    detections: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Normalize a complete OCR detection list.
    """

    normalized = []

    for detection in detections:

        if not isinstance(
            detection,
            dict
        ):
            continue

        item = normalize_detection(
            detection
        )

        if item["text"]:
            normalized.append(
                item
            )

    return normalized


# ============================================================
# FIELD-SPECIFIC VALUE NORMALIZATION
# ============================================================

def normalize_mrp_value(
    value: str
) -> Optional[str]:
    """
    Normalize MRP value.

    Examples:

        ₹399       -> ₹399
        Rs. 399    -> ₹399
        399/-      -> ₹399
        INR 399    -> ₹399
    """

    if not value:
        return None

    value = normalize_text(
        value
    )

    match = re.search(
        r"(?<!\d)"
        r"(\d{1,7}(?:[.,]\d{1,2})?)"
        r"(?!\d)",
        value
    )

    if not match:
        return None

    number = match.group(
        1
    ).replace(
        ",",
        ""
    )

    try:

        amount = float(
            number
        )

    except ValueError:

        return None

    if amount <= 0:
        return None

    # Prevent dates/years from becoming MRP.
    if 1900 <= amount <= 2100:
        return None

    if amount.is_integer():

        formatted = str(
            int(amount)
        )

    else:

        formatted = (
            f"{amount:.2f}"
            .rstrip("0")
            .rstrip(".")
        )

    return f"₹{formatted}"


def normalize_quantity_value(
    value: str
) -> Optional[str]:
    """
    Normalize a quantity value.
    """

    if not value:
        return None

    value = normalize_text(
        value
    )

    match = re.search(
        r"(?<!\d)"
        r"(\d+(?:\.\d+)?)"
        r"\s*"
        r"(kg|kgs|g|gm|gms|mg|ml|l|ltr|litre|liter)"
        r"\b",
        value,
        flags=re.IGNORECASE
    )

    if not match:
        return None

    number = match.group(
        1
    )

    unit = match.group(
        2
    ).lower()

    unit_map = {

        "kg": "KG",
        "kgs": "KG",

        "g": "G",
        "gm": "G",
        "gms": "G",

        "mg": "MG",

        "ml": "ML",

        "l": "L",
        "ltr": "L",
        "litre": "L",
        "liter": "L",
    }

    return (
        f"{number} "
        f"{unit_map.get(unit, unit.upper())}"
    )


# ============================================================
# NORMALIZED TOKENIZATION
# ============================================================

def tokenize(
    text: str
) -> List[str]:

    text = normalize_text(
        text
    )

    if not text:
        return []

    return re.findall(
        r"[A-Za-z0-9₹./:%@+\-]+",
        text
    )


# ============================================================
# HEADING DETECTION
# ============================================================

def contains_any_phrase(
    text: str,
    phrases: List[str]
) -> bool:

    normalized = search_form(
        text
    )

    for phrase in phrases:

        phrase_normalized = search_form(
            phrase
        )

        if phrase_normalized in normalized:
            return True

    return False


# ============================================================
# PUBLIC EXPORT
# ============================================================

__all__ = [

    "normalize_text",
    "compact_text",
    "search_form",

    "normalize_lines",

    "normalize_detection",
    "normalize_detections",

    "normalize_mrp_value",
    "normalize_quantity_value",

    "tokenize",

    "contains_any_phrase",
]