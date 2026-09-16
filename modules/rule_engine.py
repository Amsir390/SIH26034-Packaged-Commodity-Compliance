import json
import re
from pathlib import Path


# ============================================================
# LOAD RULE DATABASE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RULE_FILE = BASE_DIR / "rules" / "packaged_commodity_rules.json"


def load_rules():
    """Load packaged commodity rules from JSON."""

    try:
        with open(RULE_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Rule database not found: {RULE_FILE}"
        )

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON rule database: {error}"
        )


# Load once
RULE_DATABASE = load_rules()
RULES = RULE_DATABASE.get("rules", {})


# ============================================================
# HELPERS
# ============================================================

def is_present(value):
    """
    Check whether an extracted field actually contains useful data.
    """

    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    return True


def normalize_text(text):
    """Normalize OCR text for searching."""

    if not text:
        return ""

    text = str(text).lower()

    # Normalize common OCR variations
    replacements = {
        "₹": " rs ",
        "—": "-",
        "–": "-",
        "|": " ",
        "\n": " "
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# FIELD SEARCH
# ============================================================

def field_found(product_data, field):
    """
    Determine whether an extracted field exists.
    """

    value = product_data.get(field)

    return is_present(value)


# ============================================================
# PATTERN SEARCH
# ============================================================

def pattern_found(text, patterns):
    """
    Search OCR text for any supplied pattern.
    """

    if not text or not patterns:
        return False

    text = normalize_text(text)

    for pattern in patterns:

        pattern = normalize_text(pattern)

        if pattern and pattern in text:
            return True

    return False


# ============================================================
# APPLICABILITY
# ============================================================

def check_applicability(rule, product_data):
    """
    Determine whether a rule applies to the current package.

    This is intentionally conservative.

    If applicability cannot be confidently determined,
    return REVIEW instead of falsely declaring compliance.
    """

    applicability = rule.get("applicability", "")

    # Rules that normally apply to retail packages
    if applicability == "retail_package":

        package_type = str(
            product_data.get("package_type", "retail")
        ).lower()

        if package_type == "wholesale":
            return False

        return True

    # Imported products
    if applicability == "imported_package":

        imported = product_data.get("imported")

        if imported is True:
            return True

        if str(
            product_data.get("country_of_origin", "")
        ).strip():

            return True

        return False

    # Wholesale
    if applicability == "wholesale_package":

        package_type = str(
            product_data.get("package_type", "")
        ).lower()

        return package_type == "wholesale"

    # Physical measurement cannot be confirmed from OCR
    if applicability == "physical_quantity_verification":

        return True

    # Registration needs external verification
    if applicability == "registered_entity":

        return True

    # If we don't know, allow the rule engine to review it
    if "where" in applicability or "as_applicable" in applicability:
        return True

    return True


# ============================================================
# SINGLE RULE CHECK
# ============================================================

def evaluate_rule(
    rule_id,
    rule,
    product_data,
    ocr_text
):
    """
    Evaluate one compliance rule.
    """

    field = rule.get("field")
    name = rule.get("name", rule_id)

    severity = rule.get(
        "severity",
        "MEDIUM"
    )

    # --------------------------------------------------------
    # Applicability
    # --------------------------------------------------------

    applies = check_applicability(
        rule,
        product_data
    )

    if not applies:

        return {
            "rule_id": rule_id,
            "field": field,
            "name": name,
            "status": "NOT_APPLICABLE",
            "severity": severity,
            "rule_reference": rule.get(
                "rule_reference",
                ""
            ),
            "message": "This requirement does not apply to this package."
        }

    # --------------------------------------------------------
    # Physical verification
    # --------------------------------------------------------

    if rule.get("verification_method") == "physical_measurement":

        return {
            "rule_id": rule_id,
            "field": field,
            "name": name,
            "status": "REVIEW",
            "severity": severity,
            "rule_reference": rule.get(
                "rule_reference",
                ""
            ),
            "message": (
                "Physical measurement is required. "
                "This cannot be verified from an image alone."
            )
        }

    # --------------------------------------------------------
    # External verification
    # --------------------------------------------------------

    if rule.get("verification_method") == "external_database_or_document":

        return {
            "rule_id": rule_id,
            "field": field,
            "name": name,
            "status": "REVIEW",
            "severity": severity,
            "rule_reference": rule.get(
                "rule_reference",
                ""
            ),
            "message": (
                "External registration verification is required."
            )
        }

    # --------------------------------------------------------
    # Direct field check
    # --------------------------------------------------------

    if field_found(product_data, field):

        return {
            "rule_id": rule_id,
            "field": field,
            "name": name,
            "status": "PASS",
            "severity": severity,
            "rule_reference": rule.get(
                "rule_reference",
                ""
            ),
            "message": (
                f"{name} detected."
            )
        }

    # --------------------------------------------------------
    # OCR pattern check
    # --------------------------------------------------------

    patterns = rule.get(
        "detection_patterns",
        []
    )

    if pattern_found(
        ocr_text,
        patterns
    ):

        return {
            "rule_id": rule_id,
            "field": field,
            "name": name,
            "status": "PASS",
            "severity": severity,
            "rule_reference": rule.get(
                "rule_reference",
                ""
            ),
            "message": (
                f"{name} detected in label text."
            )
        }

    # --------------------------------------------------------
    # Not detected
    # --------------------------------------------------------

    if rule.get("required", False):

        return {
            "rule_id": rule_id,
            "field": field,
            "name": name,
            "status": "FAIL",
            "severity": severity,
            "rule_reference": rule.get(
                "rule_reference",
                ""
            ),
            "message": rule.get(
                "failure_message",
                f"{name} was not detected."
            )
        }

    # Optional rule
    return {
        "rule_id": rule_id,
        "field": field,
        "name": name,
        "status": "REVIEW",
        "severity": severity,
        "rule_reference": rule.get(
            "rule_reference",
            ""
        ),
        "message": (
            f"{name} could not be automatically verified."
        )
    }


# ============================================================
# OVERALL COMPLIANCE
# ============================================================

def calculate_overall_status(checks):
    """
    Calculate final compliance status.

    NON_COMPLIANT:
        At least one critical/high requirement failed.

    REVIEW:
        No critical/high failure, but something requires
        human/external verification.

    COMPLIANT:
        All applicable automated checks pass.
    """

    for check in checks:

        if check["status"] == "FAIL":

            if check["severity"] in [
                "CRITICAL",
                "HIGH"
            ]:

                return "NON_COMPLIANT"

    for check in checks:

        if check["status"] == "REVIEW":

            return "REVIEW"

    return "COMPLIANT"


# ============================================================
# MAIN COMPLIANCE FUNCTION
# ============================================================

def check_compliance(
    product_data,
    ocr_text=""
):
    """
    Main compliance engine.

    Parameters
    ----------
    product_data : dict
        Information extracted by AI/OCR.

    ocr_text : str
        Raw OCR text from the package.

    Returns
    -------
    dict
        Complete compliance report.
    """

    if product_data is None:
        product_data = {}

    checks = []

    # --------------------------------------------------------
    # Evaluate every rule
    # --------------------------------------------------------

    for rule_id, rule in RULES.items():

        result = evaluate_rule(
            rule_id,
            rule,
            product_data,
            ocr_text
        )

        checks.append(result)

    # --------------------------------------------------------
    # Overall status
    # --------------------------------------------------------

    overall_status = calculate_overall_status(
        checks
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    passed = sum(
        1
        for check in checks
        if check["status"] == "PASS"
    )

    failed = sum(
        1
        for check in checks
        if check["status"] == "FAIL"
    )

    review = sum(
        1
        for check in checks
        if check["status"] == "REVIEW"
    )

    not_applicable = sum(
        1
        for check in checks
        if check["status"] == "NOT_APPLICABLE"
    )

    return {
        "overall_status": overall_status,

        "total_checks": len(checks),

        "passed": passed,

        "failed": failed,

        "review": review,

        "not_applicable": not_applicable,

        "checks": checks,

        "product_data": product_data
    }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    test_product = {
        "product_name": "Example Biscuits",
        "mrp": "₹50",
        "net_quantity": "100 g",
        "manufacturer": "Example Foods Pvt Ltd",
        "manufacturer_address": "Kolkata, India",
        "manufacture_date": "08/2026",
        "best_before_use_by": "Best Before 6 Months",
        "consumer_care_phone": "1800123456"
    }

    test_ocr = """
    EXAMPLE BISCUITS
    NET WEIGHT 100 g
    MRP ₹50 INCLUSIVE OF ALL TAXES
    MANUFACTURED BY EXAMPLE FOODS PVT LTD
    KOLKATA INDIA
    MFD 08/2026
    BEST BEFORE 6 MONTHS
    CONSUMER CARE 1800123456
    """

    result = check_compliance(
        test_product,
        test_ocr
    )

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )