import re


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean OCR text while preserving line structure.
    """

    if not text:
        return ""

    text = text.replace("\r", "\n")

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n", text)

    return text.strip()


def clean_value(value):
    """
    Clean an extracted field value.
    """

    if not value:
        return None

    value = value.strip()

    # Remove common OCR punctuation around values
    value = value.strip(" :-–—|")

    # Remove repeated spaces
    value = re.sub(r"\s+", " ", value)

    return value.strip() or None


def get_lines(text):
    """
    Convert OCR text into clean individual lines.
    """

    text = clean_text(text)

    lines = []

    for line in text.splitlines():

        line = clean_value(line)

        if line:
            lines.append(line)

    return lines


# ============================================================
# GENERIC HELPERS
# ============================================================

def is_heading(line, patterns):
    """
    Check whether an OCR line contains one of the
    supplied field headings.
    """

    if not line:
        return False

    for pattern in patterns:

        if re.search(
            pattern,
            line,
            re.IGNORECASE
        ):
            return True

    return False


def value_after_heading(line, patterns):
    """
    Extract a value when heading and value are on the
    same line.

    Example:

        MRP: Rs. 399

        Net Weight: 250 GM
    """

    if not line:
        return None

    for pattern in patterns:

        match = re.search(
            pattern,
            line,
            re.IGNORECASE
        )

        if match:

            value = line[match.end():]

            value = clean_value(value)

            if value:
                return value

    return None


def next_value(lines, index, max_lines=4):
    """
    Find a useful value after a heading.

    This handles labels such as:

        MRP:
        (399/-)

    and:

        Net Weight:
        250GM
    """

    for offset in range(1, max_lines + 1):

        position = index + offset

        if position >= len(lines):
            break

        candidate = clean_value(
            lines[position]
        )

        if not candidate:
            continue

        # Do not accidentally take another heading
        if is_heading(
            candidate,
            FIELD_HEADINGS
        ):
            continue

        return candidate

    return None


# ============================================================
# FIELD HEADINGS
# ============================================================

MRP_PATTERNS = [

    r"\bMRP\b",

    r"\bM\.?\s*R\.?\s*P\.?\b",

    r"\bMAXIMUM\s+RETAIL\s+PRICE\b",

    r"\bMAX\.?\s*RETAIL\s+PRICE\b",

]


QUANTITY_PATTERNS = [

    r"\bNET\s+QUANTITY\b",

    r"\bNET\s+QTY\b",

    r"\bNET\s+WEIGHT\b",

    r"\bNET\s+WT\.?\b",

    r"\bNET\s+WT\b",

]


MANUFACTURER_PATTERNS = [

    r"\bMANUFACTURER\b",

    r"\bMANUFACTURED\s+BY\b",

    r"\bMANUFACTURED\s*&\s*MARKETED\s+BY\b",

    r"\bMANUFACTURED\s+AND\s+MARKETED\s+BY\b",

    r"\bPACKED\s+BY\b",

    r"\bPACKED\s*&\s*MARKETED\s+BY\b",

    r"\bPACKED\s+AND\s+MARKETED\s+BY\b",

    r"\bMARKETED\s+BY\b",

    r"\bMFG\.?\s*BY\b",

    r"\bMFD\.?\s*BY\b",

]


ADDRESS_PATTERNS = [

    r"\bADDRESS\b",

    r"\bREGISTERED\s+OFFICE\b",

    r"\bCORPORATE\s+OFFICE\b",

    r"\bMANUFACTURING\s+ADDRESS\b",

]


FIELD_HEADINGS = (
    MRP_PATTERNS
    + QUANTITY_PATTERNS
    + MANUFACTURER_PATTERNS
    + ADDRESS_PATTERNS
)


# ============================================================
# MRP EXTRACTION
# ============================================================

def extract_mrp(lines):
    """
    Extract MRP from many common Indian package formats.

    Supports:

        MRP: ₹399
        MRP: Rs. 399
        MRP
        399/-

        MRP:
        (399/-)

        Maximum Retail Price
        399
    """

    # --------------------------------------------------------
    # First: search for MRP heading
    # --------------------------------------------------------

    for index, line in enumerate(lines):

        if not is_heading(
            line,
            MRP_PATTERNS
        ):
            continue

        # Try same line
        value = value_after_heading(
            line,
            MRP_PATTERNS
        )

        if value:

            amount = parse_mrp_amount(
                value
            )

            if amount:
                return amount

        # Try following lines
        for offset in range(1, 5):

            position = index + offset

            if position >= len(lines):
                break

            candidate = lines[position]

            amount = parse_mrp_amount(
                candidate
            )

            if amount:
                return amount

    # --------------------------------------------------------
    # Fallback: search entire OCR text for MRP-like pattern
    # --------------------------------------------------------

    joined = " ".join(lines)

    match = re.search(
        r"(?:MRP|M\.?\s*R\.?\s*P\.?)"
        r"\s*[:\-]?\s*"
        r"(?:₹|RS\.?|INR)?"
        r"\s*"
        r"[\(\[]?"
        r"\s*"
        r"(\d+(?:\.\d{1,2})?)",
        joined,
        re.IGNORECASE
    )

    if match:

        return f"₹{match.group(1)}"

    return None


def parse_mrp_amount(text):
    """
    Extract a monetary amount from an OCR line.

    Examples:

        (399/-)
        Rs. 399
        ₹399
        399/-
        399.00
    """

    if not text:
        return None

    # Normalize OCR variations
    normalized = text

    normalized = re.sub(
        r"\bR[S5]\.?\b",
        "Rs",
        normalized,
        flags=re.IGNORECASE
    )

    normalized = normalized.replace(
        "₹",
        "₹"
    )

    # Look for a number
    match = re.search(
        r"(?<!\d)"
        r"(\d{1,7}(?:\.\d{1,2})?)"
        r"(?!\d)",
        normalized
    )

    if not match:
        return None

    amount = match.group(1)

    try:

        number = float(amount)

        if number <= 0:
            return None

        # Avoid treating years as MRP
        if 1900 <= number <= 2100:
            return None

    except ValueError:
        return None

    return f"₹{amount}"


# ============================================================
# NET QUANTITY EXTRACTION
# ============================================================

def extract_net_quantity(lines):
    """
    Extract net quantity / net weight.

    Supports:

        Net Weight: 250 GM

        Net Weight
        250GM

        Net Quantity
        500 g

        Net Wt.
        1 kg
    """

    # --------------------------------------------------------
    # Search around quantity heading
    # --------------------------------------------------------

    for index, line in enumerate(lines):

        if not is_heading(
            line,
            QUANTITY_PATTERNS
        ):
            continue

        # Same line
        value = value_after_heading(
            line,
            QUANTITY_PATTERNS
        )

        quantity = parse_quantity(
            value
        )

        if quantity:
            return quantity

        # Following lines
        for offset in range(1, 5):

            position = index + offset

            if position >= len(lines):
                break

            candidate = lines[position]

            quantity = parse_quantity(
                candidate
            )

            if quantity:
                return quantity

    # --------------------------------------------------------
    # Fallback: search whole OCR text
    # --------------------------------------------------------

    joined = " ".join(lines)

    match = re.search(
        r"\b(?:NET\s+WEIGHT|NET\s+QUANTITY|NET\s+QTY|NET\s+WT\.?)"
        r"\s*[:\-]?\s*"
        r"(\d+(?:\.\d+)?)"
        r"\s*"
        r"(kg|kgs|g|gm|gms|mg|ml|l|ltr|litre|liter)\b",
        joined,
        re.IGNORECASE
    )

    if match:

        return normalize_quantity(
            match.group(1),
            match.group(2)
        )

    return None


def parse_quantity(text):
    """
    Parse a quantity from a single OCR line.
    """

    if not text:
        return None

    match = re.search(
        r"(?<!\d)"
        r"(\d+(?:\.\d+)?)"
        r"\s*"
        r"(kg|kgs|g|gm|gms|mg|ml|l|ltr|litre|liter)"
        r"\b",
        text,
        re.IGNORECASE
    )

    if not match:
        return None

    return normalize_quantity(
        match.group(1),
        match.group(2)
    )


def normalize_quantity(number, unit):
    """
    Normalize quantity units.
    """

    unit = unit.lower()

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

    normalized_unit = unit_map.get(
        unit,
        unit.upper()
    )

    return f"{number} {normalized_unit}"


# ============================================================
# MANUFACTURER EXTRACTION
# ============================================================

def extract_manufacturer(lines):
    """
    Extract manufacturer / packer / marketer.

    Important:

    Indian labels may use:

        Manufactured By
        Manufactured & Marketed By
        Packed By
        Packed & Marketed By
        Marketed By
        Mfd. By
        Mfg. By

    Example:

        Packed & Marketed By:
        Hindvedic Wellness Private Limited
    """

    for index, line in enumerate(lines):

        if not is_heading(
            line,
            MANUFACTURER_PATTERNS
        ):
            continue

        # ----------------------------------------------------
        # Same-line extraction
        # ----------------------------------------------------

        value = value_after_heading(
            line,
            MANUFACTURER_PATTERNS
        )

        if value:

            value = clean_manufacturer(
                value
            )

            if value:
                return value

        # ----------------------------------------------------
        # Following lines
        # ----------------------------------------------------

        candidates = []

        for offset in range(1, 5):

            position = index + offset

            if position >= len(lines):
                break

            candidate = lines[position]

            if is_heading(
                candidate,
                FIELD_HEADINGS
            ):
                break

            # Stop at obvious address labels
            if re.search(
                r"\b(?:address|customer\s+care|email|phone|mobile)\b",
                candidate,
                re.IGNORECASE
            ):
                break

            candidates.append(
                candidate
            )

            # Usually the company name appears in the
            # first one or two lines.
            if len(candidates) >= 2:
                break

        if candidates:

            manufacturer = " ".join(
                candidates
            )

            manufacturer = clean_manufacturer(
                manufacturer
            )

            if manufacturer:
                return manufacturer

    return None


def clean_manufacturer(value):
    """
    Clean manufacturer/company name.
    """

    if not value:
        return None

    value = clean_value(
        value
    )

    # Remove trailing punctuation
    value = value.rstrip(
        ".,:;-"
    )

    return value or None


# ============================================================
# ADDRESS EXTRACTION
# ============================================================

def extract_address(lines):
    """
    Extract multi-line address.

    Example:

        Address:
        C-2, F.F, Okhla Industrial Area,
        Phase-1, Delhi - 110020

    Also supports address immediately following
    manufacturer information.
    """

    # --------------------------------------------------------
    # Explicit Address heading
    # --------------------------------------------------------

    for index, line in enumerate(lines):

        if is_heading(
            line,
            ADDRESS_PATTERNS
        ):

            value = value_after_heading(
                line,
                ADDRESS_PATTERNS
            )

            if value:
                return clean_address(
                    value
                )

            candidates = []

            for offset in range(1, 6):

                position = index + offset

                if position >= len(lines):
                    break

                candidate = lines[position]

                if is_heading(
                    candidate,
                    FIELD_HEADINGS
                ):
                    break

                if re.search(
                    r"\b(?:customer\s+care|email|phone|mobile)\b",
                    candidate,
                    re.IGNORECASE
                ):
                    break

                candidates.append(
                    candidate
                )

            if candidates:

                return clean_address(
                    " ".join(candidates)
                )

    # --------------------------------------------------------
    # Detect address after manufacturer
    # --------------------------------------------------------

    manufacturer_index = None

    for index, line in enumerate(lines):

        if is_heading(
            line,
            MANUFACTURER_PATTERNS
        ):

            manufacturer_index = index

            break

    if manufacturer_index is not None:

        start = manufacturer_index + 1

        # Skip company name
        if start < len(lines):
            start += 1

        address_parts = []

        for index in range(
            start,
            min(
                start + 6,
                len(lines)
            )
        ):

            line = lines[index]

            if re.search(
                r"\b(?:customer\s+care|email|phone|mobile)\b",
                line,
                re.IGNORECASE
            ):
                break

            if is_heading(
                line,
                FIELD_HEADINGS
            ):
                break

            # Address usually contains one of these
            if looks_like_address(line):

                address_parts.append(
                    line
                )

        if address_parts:

            return clean_address(
                " ".join(address_parts)
            )

    # --------------------------------------------------------
    # Final fallback: detect Indian PIN code
    # --------------------------------------------------------

    for index, line in enumerate(lines):

        if re.search(
            r"\b\d{6}\b",
            line
        ):

            address_parts = []

            start = max(
                0,
                index - 3
            )

            end = min(
                len(lines),
                index + 1
            )

            for position in range(
                start,
                end
            ):

                candidate = lines[position]

                if looks_like_address(
                    candidate
                ):
                    address_parts.append(
                        candidate
                    )

            if address_parts:

                return clean_address(
                    " ".join(address_parts)
                )

    return None


def looks_like_address(text):
    """
    Determine whether an OCR line looks like
    an address.
    """

    if not text:
        return False

    address_words = [

        "road",
        "rd",
        "street",
        "st.",
        "lane",
        "ln",
        "industrial",
        "area",
        "phase",
        "sector",
        "plot",
        "floor",
        "building",
        "nagar",
        "market",
        "delhi",
        "mumbai",
        "kolkata",
        "chennai",
        "bengaluru",
        "bangalore",
        "hyderabad",
        "india",

    ]

    lowered = text.lower()

    # Indian PIN code
    if re.search(
        r"\b\d{6}\b",
        text
    ):
        return True

    # Address keywords
    for word in address_words:

        if word in lowered:
            return True

    # Typical address format:
    # C-2, F.F, Something
    if re.search(
        r"\d+\s*[-,/]",
        text
    ):
        return True

    return False


def clean_address(value):
    """
    Normalize address formatting.
    """

    if not value:
        return None

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    value = value.strip(
        " :-–—|"
    )

    return value or None


# ============================================================
# PRODUCT NAME EXTRACTION
# ============================================================

def extract_product_name(lines):
    """
    Extract product name intelligently.

    Do NOT simply use lines[0], because OCR may return:

        Nutrition Facts

    or:

        At

    before the actual product name.

    For the supplied product this should identify:

        SUNFLOWER SEEDS
    """

    if not lines:
        return None

    # --------------------------------------------------------
    # Strong product-name candidates
    # --------------------------------------------------------

    ignored_patterns = [

        r"^nutrition\s+facts?$",

        r"^suggested\s+serving",

        r"^amount\s+per",

        r"^ingredients?$",

        r"^country\s+of\s+origin",

        r"^storage",

        r"^storage\s+information",

        r"^packed",

        r"^manufactured",

        r"^marketed",

        r"^customer\s+care",

        r"^email",

        r"^like\s+us",

        r"^net\s+weight",

        r"^net\s+quantity",

        r"^mrp",

        r"^usp",

        r"^batch",

        r"^date\s+of",

        r"^use\s+by",

        r"^best\s+before",

    ]

    candidates = []

    for index, line in enumerate(lines):

        cleaned = clean_value(
            line
        )

        if not cleaned:
            continue

        # Ignore obvious metadata
        ignored = False

        for pattern in ignored_patterns:

            if re.search(
                pattern,
                cleaned,
                re.IGNORECASE
            ):

                ignored = True
                break

        if ignored:
            continue

        # Avoid very short OCR fragments
        if len(cleaned) < 3:
            continue

        # Avoid lines that are mainly numbers
        if re.fullmatch(
            r"[\d\s.,:/()\-]+",
            cleaned
        ):
            continue

        candidates.append(
            (index, cleaned)
        )

    if not candidates:
        return None

    # --------------------------------------------------------
    # Prefer early prominent text
    # --------------------------------------------------------

    for index, candidate in candidates[:10]:

        upper = candidate.upper()

        # Product names are often uppercase/prominent.
        if (
            len(candidate) >= 5
            and re.search(
                r"[A-Za-z]",
                candidate
            )
        ):

            # Avoid generic company/product words
            if upper not in {
                "NUTRITION",
                "NUTRITION FACTS",
                "SUGGESTED USES",
                "PROMISE",
                "INGREDIENTS",
            }:

                return candidate

    return candidates[0][1]


# ============================================================
# MAIN EXTRACTION FUNCTION
# ============================================================

def extract_product_details(text):
    """
    Advanced packaged-commodity information extractor.

    Handles:

        Product name
        MRP
        Net quantity / net weight
        Manufacturer
        Address

    The extractor supports both:

        Heading: Value

    and:

        Heading
        Value

    It is specifically designed for OCR output where
    label information may appear on separate lines.
    """

    data = {

        "product_name": None,

        "mrp": None,

        "net_quantity": None,

        "manufacturer": None,

        "address": None,

    }

    # --------------------------------------------------------
    # Validate OCR
    # --------------------------------------------------------

    if not text:

        return data

    # --------------------------------------------------------
    # Prepare OCR lines
    # --------------------------------------------------------

    lines = get_lines(
        text
    )

    if not lines:

        return data

    # --------------------------------------------------------
    # Extract fields
    # --------------------------------------------------------

    data["mrp"] = extract_mrp(
        lines
    )

    data["net_quantity"] = extract_net_quantity(
        lines
    )

    data["manufacturer"] = extract_manufacturer(
        lines
    )

    data["address"] = extract_address(
        lines
    )

    data["product_name"] = extract_product_name(
        lines
    )

    return data