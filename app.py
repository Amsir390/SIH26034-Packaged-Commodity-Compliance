import json
import traceback
from typing import Any, Dict, List

import streamlit as st
from PIL import Image

# ============================================================
# PROJECT MODULES
# ============================================================

from modules.scanner import process_scan
from modules.ocr import extract_text, extract_text_with_details
from modules.ai_extraction import extract_product_details
from modules.rule_engine import check_compliance
from modules.report import generate_report

from database.database import create_tables, save_scan


# ============================================================
# OPTIONAL ADVANCED MODULES
# ============================================================
#
# These modules were added to your project:
#
#   text_normalizer.py
#   field_classifier.py
#   confidence.py
#
# Because their exact public function names can differ while
# developing the project, this app loads them safely.
#
# The application WILL NOT crash if one optional helper has
# a different function name.
# ============================================================

try:
    from modules import text_normalizer
except Exception:
    text_normalizer = None

try:
    from modules import field_classifier
except Exception:
    field_classifier = None

try:
    from modules import confidence as confidence_module
except Exception:
    confidence_module = None


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Packaged Commodity Compliance",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

try:
    create_tables()
except Exception as error:
    st.warning(
        f"Database initialization warning: {error}"
    )


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .field-card {
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 12px;
    }

    .confidence-high {
        border-left: 5px solid #21c55d;
    }

    .confidence-medium {
        border-left: 5px solid #f59e0b;
    }

    .confidence-low {
        border-left: 5px solid #ef4444;
    }

    .small-text {
        font-size: 13px;
        opacity: 0.7;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📦 Packaged Commodity Compliance</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'AI-assisted packaged commodity label compliance checker'
    '</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Scan or upload a packaged commodity label to analyze "
    "image quality, recognize text, understand label fields, "
    "and check compliance."
)

st.divider()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_string(value: Any) -> str:
    """
    Safely convert any value to displayable text.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    return str(value).strip()


def display_value(value: Any) -> str:
    """
    UI display helper.
    """

    value = safe_string(value)

    if not value:
        return "Not found"

    return value


def normalize_ocr_for_pipeline(text: str) -> str:
    """
    Send OCR text through the new text_normalizer module
    whenever a compatible function exists.

    Falls back to the original OCR text.
    """

    if not text:
        return ""

    if text_normalizer is None:
        return text

    candidate_functions = [
        "normalize_text",
        "normalize_ocr_text",
        "clean_text",
        "preprocess_text",
        "normalize",
    ]

    for function_name in candidate_functions:

        function = getattr(
            text_normalizer,
            function_name,
            None,
        )

        if not callable(function):
            continue

        try:

            result = function(text)

            if isinstance(result, str):
                return result

        except Exception:
            continue

    return text


def run_field_classifier(
    text: str,
) -> Dict[str, Any]:
    """
    Try to use field_classifier.py without making app.py
    dependent on one exact function name.

    The classifier is an additional semantic layer.

    It can potentially identify:

        PRODUCT_NAME
        MRP
        NET_QUANTITY
        MANUFACTURER
        ADDRESS
        BATCH_NUMBER
        MFG_DATE
        EXPIRY_DATE
        FSSAI
        CUSTOMER_CARE
        EMAIL
        PHONE
        INGREDIENTS
        etc.

    If the classifier does not expose a compatible function,
    an empty result is returned and ai_extraction.py remains
    the main extraction engine.
    """

    result = {
        "fields": {},
        "candidates": [],
        "raw": None,
    }

    if not text:
        return result

    if field_classifier is None:
        return result

    candidate_functions = [
        "classify_fields",
        "classify_text",
        "classify",
        "extract_fields",
        "classify_candidates",
        "get_field_candidates",
    ]

    for function_name in candidate_functions:

        function = getattr(
            field_classifier,
            function_name,
            None,
        )

        if not callable(function):
            continue

        try:

            output = function(text)

            if output is None:
                continue

            result["raw"] = output

            if isinstance(output, dict):

                # Common format:
                #
                # {
                #     "mrp": ...,
                #     "net_quantity": ...
                # }
                #
                # OR:
                #
                # {
                #     "fields": {...},
                #     "candidates": [...]
                # }

                if isinstance(
                    output.get("fields"),
                    dict,
                ):
                    result["fields"] = output["fields"]

                elif isinstance(
                    output.get("classified_fields"),
                    dict,
                ):
                    result["fields"] = output[
                        "classified_fields"
                    ]

                else:
                    result["fields"] = output

                if isinstance(
                    output.get("candidates"),
                    list,
                ):
                    result["candidates"] = output[
                        "candidates"
                    ]

                return result

            if isinstance(output, list):

                result["candidates"] = output

                return result

        except Exception:
            continue

    return result


def get_confidence_summary(
    classifier_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Use confidence.py if possible.

    This supports confidence.py versions exposing:

        summarize_confidence
        rank_candidates_by_confidence
        confidence_from_classifier_candidate
    """

    output = {
        "summary": None,
        "ranked": [],
    }

    if confidence_module is None:
        return output

    candidates = classifier_result.get(
        "candidates",
        [],
    )

    # --------------------------------------------------------
    # Rank candidates
    # --------------------------------------------------------

    rank_function = getattr(
        confidence_module,
        "rank_candidates_by_confidence",
        None,
    )

    if callable(rank_function) and candidates:

        try:

            ranked = rank_function(
                candidates
            )

            if isinstance(
                ranked,
                list,
            ):
                output["ranked"] = ranked

        except Exception:
            pass

    # --------------------------------------------------------
    # Confidence summary
    # --------------------------------------------------------

    summary_function = getattr(
        confidence_module,
        "summarize_confidence",
        None,
    )

    if callable(summary_function):

        try:

            summary = summary_function(
                candidates
            )

            if isinstance(
                summary,
                dict,
            ):
                output["summary"] = summary

        except Exception:
            pass

    return output


def merge_extraction_results(
    ai_data: Dict[str, Any],
    classifier_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Combine AI extraction and classifier output.

    IMPORTANT:

    AI extraction remains the primary source.

    The classifier is used as a supporting semantic layer.

    This prevents a weak classifier result from destroying a
    stronger AI extraction result.
    """

    if not isinstance(
        ai_data,
        dict,
    ):
        ai_data = {}

    fields = classifier_result.get(
        "fields",
        {},
    )

    if not isinstance(
        fields,
        dict,
    ):
        fields = {}

    final_data = dict(
        ai_data
    )

    # --------------------------------------------------------
    # Standard field aliases
    # --------------------------------------------------------

    aliases = {

        "product_name": [
            "product_name",
            "product",
            "name",
            "productname",
        ],

        "mrp": [
            "mrp",
            "maximum_retail_price",
            "maximum retail price",
            "retail_price",
        ],

        "net_quantity": [
            "net_quantity",
            "net quantity",
            "quantity",
            "net_weight",
            "net weight",
            "weight",
        ],

        "manufacturer": [
            "manufacturer",
            "manufactured_by",
            "manufactured by",
            "mfg_by",
            "mfg by",
            "packer",
            "packed_by",
            "packed by",
            "marketer",
            "marketed_by",
            "marketed by",
        ],

        "address": [
            "address",
            "manufacturer_address",
            "manufacturing_address",
            "registered_office",
        ],
    }

    # --------------------------------------------------------
    # Only use classifier value when AI value is missing.
    # --------------------------------------------------------

    for standard_field, possible_names in aliases.items():

        current_value = final_data.get(
            standard_field
        )

        if current_value not in (
            None,
            "",
            "Not found",
            "not found",
        ):
            continue

        for name in possible_names:

            for actual_key, value in fields.items():

                normalized_actual = (
                    str(actual_key)
                    .strip()
                    .lower()
                )

                normalized_name = (
                    str(name)
                    .strip()
                    .lower()
                )

                if normalized_actual != normalized_name:
                    continue

                if value in (
                    None,
                    "",
                ):
                    continue

                final_data[
                    standard_field
                ] = value

                break

            if final_data.get(
                standard_field
            ):
                break

    return final_data


def calculate_extraction_completeness(
    product_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate how many important fields were successfully
    extracted.

    This is NOT legal compliance.

    It only measures extraction completeness.
    """

    fields = [
        "product_name",
        "mrp",
        "net_quantity",
        "manufacturer",
        "address",
    ]

    found = 0

    missing = []

    for field in fields:

        value = product_data.get(
            field
        )

        if value not in (
            None,
            "",
            "Not found",
        ):
            found += 1
        else:
            missing.append(
                field
            )

    total = len(
        fields
    )

    percentage = (
        found / total * 100
        if total
        else 0
    )

    return {
        "found": found,
        "total": total,
        "percentage": round(
            percentage,
            1,
        ),
        "missing": missing,
    }


def get_status_icon(status: str) -> str:

    status = safe_string(
        status
    ).upper()

    if status in (
        "PASS",
        "COMPLIANT",
    ):
        return "✅"

    if status in (
        "FAIL",
        "NON-COMPLIANT",
    ):
        return "❌"

    if status in (
        "REVIEW",
        "NEEDS REVIEW",
    ):
        return "⚠️"

    if status in (
        "NOT_APPLICABLE",
        "N/A",
    ):
        return "⚪"

    return "ℹ️"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ System"
    )

    st.write(
        "AI-assisted packaged commodity "
        "compliance analysis."
    )

    st.divider()

    st.write(
        "### Pipeline"
    )

    st.write(
        "📷 Image"
    )

    st.write(
        "↓"
    )

    st.write(
        "🔲 Label / Scan Analysis"
    )

    st.write(
        "↓"
    )

    st.write(
        "🔤 OCR"
    )

    st.write(
        "↓"
    )

    st.write(
        "🧹 Text Normalization"
    )

    st.write(
        "↓"
    )

    st.write(
        "🧠 Field Classification"
    )

    st.write(
        "↓"
    )

    st.write(
        "🤖 Information Extraction"
    )

    st.write(
        "↓"
    )

    st.write(
        "⚖️ Rule Engine"
    )

    st.write(
        "↓"
    )

    st.write(
        "📄 Compliance Report"
    )

    st.divider()

    st.caption(
        "SIH26034 — Packaged Commodity Compliance"
    )


# ============================================================
# INPUT METHOD
# ============================================================

st.subheader(
    "📷 Product Scanner"
)

input_mode = st.radio(
    "Choose input method",
    [
        "📷 Camera Scanner",
        "📁 Upload Image",
    ],
    horizontal=True,
)

uploaded_file = None


# ============================================================
# CAMERA
# ============================================================

if input_mode == "📷 Camera Scanner":

    uploaded_file = st.camera_input(
        "Point the camera at the product label"
    )


# ============================================================
# UPLOAD
# ============================================================

else:

    uploaded_file = st.file_uploader(
        "Choose a product or label image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )


# ============================================================
# MAIN PROCESSING
# ============================================================

if uploaded_file is not None:

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    try:

        original_image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception as error:

        st.error(
            f"Unable to read image: {error}"
        )

        st.stop()

    # ========================================================
    # DISPLAY ORIGINAL
    # ========================================================

    st.subheader(
        "🖼️ Original Image"
    )

    st.image(
        original_image,
        caption="Uploaded / Captured Product",
        width="stretch",
    )

    st.divider()

    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    analyze = st.button(
        "🔍 Analyze Product",
        type="primary",
        width="stretch",
    )

    if analyze:

        # ====================================================
        # STEP 1 — SCANNER
        # ====================================================

        st.header(
            "1️⃣ Image & Label Analysis"
        )

        with st.spinner(
            "🔍 Analyzing image quality and label..."
        ):

            try:

                scan_result = process_scan(
                    original_image
                )

                if not isinstance(
                    scan_result,
                    dict,
                ):
                    scan_result = {
                        "image": original_image,
                        "ocr_image": original_image,
                        "quality": {},
                        "label_detected": False,
                        "label_confidence": 0,
                    }

            except Exception as error:

                st.error(
                    f"Scanner error: {error}"
                )

                st.code(
                    traceback.format_exc()
                )

                st.stop()

        # ====================================================
        # QUALITY
        # ====================================================

        quality = scan_result.get(
            "quality",
            {},
        )

        if not isinstance(
            quality,
            dict,
        ):
            quality = {}

        quality_score = quality.get(
            "score",
            quality.get(
                "quality_score",
                0,
            ),
        )

        label_confidence = scan_result.get(
            "label_confidence",
            0,
        )

        label_detected = scan_result.get(
            "label_detected",
            False,
        )

        sharpness = quality.get(
            "sharpness",
            quality.get(
                "blur_score",
                0,
            ),
        )

        glare = quality.get(
            "glare",
            quality.get(
                "glare_ratio",
                0,
            ),
        )

        col1, col2, col3, col4 = st.columns(
            4
        )

        with col1:

            st.metric(
                "Image Quality",
                f"{quality_score}/100",
            )

        with col2:

            st.metric(
                "Label Detection",
                f"{label_confidence}%",
            )

        with col3:

            try:

                st.metric(
                    "Sharpness",
                    f"{float(sharpness):.0f}",
                )

            except Exception:

                st.metric(
                    "Sharpness",
                    display_value(
                        sharpness
                    ),
                )

        with col4:

            try:

                st.metric(
                    "Glare",
                    f"{float(glare):.1f}%",
                )

            except Exception:

                st.metric(
                    "Glare",
                    display_value(
                        glare
                    ),
                )

        # ====================================================
        # QUALITY STATUS
        # ====================================================

        quality_status = safe_string(
            quality.get(
                "status",
                "UNKNOWN",
            )
        ).upper()

        if quality_status == "GOOD":

            st.success(
                "🟢 Image quality is good for processing."
            )

        elif quality_status in (
            "FAIR",
            "ACCEPTABLE",
        ):

            st.warning(
                "🟡 Image quality is acceptable, "
                "but some fields may require verification."
            )

        elif quality_status == "POOR":

            st.error(
                "🔴 Image quality is poor. "
                "Consider capturing another image."
            )

        # ====================================================
        # WARNINGS
        # ====================================================

        warnings = quality.get(
            "warnings",
            [],
        )

        if warnings:

            with st.expander(
                "⚠️ Image Quality Warnings"
            ):

                for warning in warnings:

                    st.write(
                        f"• {warning}"
                    )

        # ====================================================
        # SUGGESTIONS
        # ====================================================

        suggestions = quality.get(
            "suggestions",
            [],
        )

        if suggestions:

            with st.expander(
                "💡 Improve Scan"
            ):

                for suggestion in suggestions:

                    st.write(
                        f"• {suggestion}"
                    )

        # ====================================================
        # LABEL DETECTION
        # ====================================================

        if label_detected:

            st.success(
                f"🔲 Label/product region detected "
                f"with {label_confidence}% confidence."
            )

        else:

            st.warning(
                "⚠️ A clear label boundary was not confidently "
                "detected. The complete image will still be "
                "processed by OCR."
            )

        # ====================================================
        # PROCESSED IMAGE
        # ====================================================

        processed_image = scan_result.get(
            "image",
            original_image,
        )

        st.subheader(
            "✨ Processed Label"
        )

        if processed_image is not None:

            st.image(
                processed_image,
                caption="Detected / Corrected Label",
                width="stretch",
            )

        # ====================================================
        # OCR IMAGE
        # ====================================================

        ocr_image = scan_result.get(
            "ocr_image",
            processed_image,
        )

        if ocr_image is None:

            ocr_image = original_image

        # ====================================================
        # STEP 2 — OCR
        # ====================================================

        st.divider()

        st.header(
            "2️⃣ Text Recognition"
        )

        with st.spinner(
            "🔤 Reading all visible text..."
        ):

            try:

                # Use detailed OCR when available.
                try:

                    ocr_details = (
                        extract_text_with_details(
                            ocr_image
                        )
                    )

                    if isinstance(
                        ocr_details,
                        dict,
                    ):

                        extracted_text = (
                            ocr_details.get(
                                "text",
                                "",
                            )
                        )

                    else:

                        extracted_text = (
                            extract_text(
                                ocr_image
                            )
                        )

                except Exception:

                    extracted_text = (
                        extract_text(
                            ocr_image
                        )
                    )

            except Exception as error:

                st.error(
                    f"OCR error: {error}"
                )

                st.code(
                    traceback.format_exc()
                )

                st.stop()

        extracted_text = safe_string(
            extracted_text
        )

        # ====================================================
        # NORMALIZE OCR
        # ====================================================

        normalized_text = (
            normalize_ocr_for_pipeline(
                extracted_text
            )
        )

        # ====================================================
        # OCR RESULT
        # ====================================================

        with st.expander(
            "🔤 View Extracted OCR Text",
            expanded=True,
        ):

            if extracted_text:

                st.text(
                    extracted_text
                )

            else:

                st.warning(
                    "No readable text was detected."
                )

        # ====================================================
        # NORMALIZED RESULT
        # ====================================================

        if normalized_text and (
            normalized_text
            != extracted_text
        ):

            with st.expander(
                "🧹 View Normalized Text"
            ):

                st.text(
                    normalized_text
                )

        # ====================================================
        # OCR STATUS
        # ====================================================

        if extracted_text:

            st.success(
                "✅ OCR successfully extracted text."
            )

        else:

            st.error(
                "❌ OCR could not extract readable text."
            )

            st.stop()

        # ====================================================
        # STEP 3 — FIELD CLASSIFICATION
        # ====================================================

        st.divider()

        st.header(
            "3️⃣ Intelligent Field Classification"
        )

        st.info(
            "The system does not assume a fixed label layout. "
            "It analyzes the meaning of the detected text and "
            "looks for relationships between headings, values, "
            "units, dates, prices and company information."
        )

        with st.spinner(
            "🧠 Understanding which text belongs to which field..."
        ):

            classifier_result = (
                run_field_classifier(
                    normalized_text
                )
            )

        classifier_fields = classifier_result.get(
            "fields",
            {},
        )

        classifier_candidates = (
            classifier_result.get(
                "candidates",
                [],
            )
        )

        if classifier_fields:

            st.success(
                f"🧠 Classified {len(classifier_fields)} field(s)."
            )

            with st.expander(
                "🧠 View Classified Fields",
                expanded=False,
            ):

                st.json(
                    classifier_fields
                )

        elif classifier_candidates:

            st.success(
                f"🧠 Generated {len(classifier_candidates)} "
                "field candidate(s)."
            )

            with st.expander(
                "🧠 View Field Candidates",
                expanded=False,
            ):

                st.json(
                    classifier_candidates
                )

        else:

            st.warning(
                "The field classifier did not return structured "
                "candidates. The advanced AI extraction layer "
                "will continue processing the OCR text."
            )

        # ====================================================
        # CONFIDENCE
        # ====================================================

        confidence_result = (
            get_confidence_summary(
                classifier_result
            )
        )

        ranked_candidates = (
            confidence_result.get(
                "ranked",
                [],
            )
        )

        confidence_summary = (
            confidence_result.get(
                "summary"
            )
        )

        if ranked_candidates:

            with st.expander(
                "📈 Field Confidence Ranking"
            ):

                st.json(
                    ranked_candidates
                )

        if confidence_summary:

            with st.expander(
                "📊 Confidence Summary"
            ):

                st.json(
                    confidence_summary
                )

        # ====================================================
        # STEP 4 — AI EXTRACTION
        # ====================================================

        st.divider()

        st.header(
            "4️⃣ Product Information Extraction"
        )

        with st.spinner(
            "🤖 Understanding product information..."
        ):

            try:

                ai_product_data = (
                    extract_product_details(
                        normalized_text
                    )
                )

                if not isinstance(
                    ai_product_data,
                    dict,
                ):

                    ai_product_data = {}

            except Exception as error:

                st.error(
                    f"AI extraction error: {error}"
                )

                st.code(
                    traceback.format_exc()
                )

                st.stop()

        # ====================================================
        # MERGE CLASSIFIER + AI
        # ====================================================

        product_data = (
            merge_extraction_results(
                ai_product_data,
                classifier_result,
            )
        )

        # ====================================================
        # EXTRACTION COMPLETENESS
        # ====================================================

        completeness = (
            calculate_extraction_completeness(
                product_data
            )
        )

        st.subheader(
            "📋 Extracted Product Information"
        )

        st.metric(
            "Important Fields Found",
            f"{completeness['found']}/"
            f"{completeness['total']}",
            f"{completeness['percentage']}%",
        )

        if completeness["missing"]:

            st.caption(
                "Missing/uncertain fields: "
                + ", ".join(
                    completeness["missing"]
                )
            )

        # ====================================================
        # FIELD DISPLAY
        # ====================================================

        col1, col2 = st.columns(
            2
        )

        # ----------------------------------------------------
        # LEFT
        # ----------------------------------------------------

        with col1:

            st.write(
                "**Product Name:**"
            )

            st.info(
                display_value(
                    product_data.get(
                        "product_name"
                    )
                )
            )

            st.write(
                "**MRP:**"
            )

            mrp_value = display_value(
                product_data.get(
                    "mrp"
                )
            )

            st.info(
                mrp_value
            )

            st.write(
                "**Net Quantity:**"
            )

            st.info(
                display_value(
                    product_data.get(
                        "net_quantity"
                    )
                )
            )

        # ----------------------------------------------------
        # RIGHT
        # ----------------------------------------------------

        with col2:

            st.write(
                "**Manufacturer / Packer / Marketer:**"
            )

            st.info(
                display_value(
                    product_data.get(
                        "manufacturer"
                    )
                )
            )

            st.write(
                "**Address:**"
            )

            st.info(
                display_value(
                    product_data.get(
                        "address"
                    )
                )
            )

        # ====================================================
        # ALL EXTRACTION DATA
        # ====================================================

        with st.expander(
            "🤖 View Complete Extracted Data"
        ):

            st.json(
                product_data
            )

        # ====================================================
        # STEP 5 — COMPLIANCE
        # ====================================================

        st.divider()

        st.header(
            "5️⃣ Legal Metrology Compliance Check"
        )

        with st.spinner(
            "⚖️ Checking applicable packaged commodity rules..."
        ):

            try:

                compliance_result = (
                    check_compliance(
                        product_data,
                        normalized_text,
                    )
                )

                if not isinstance(
                    compliance_result,
                    dict,
                ):

                    compliance_result = {}

            except Exception as error:

                st.error(
                    f"Compliance engine error: {error}"
                )

                st.code(
                    traceback.format_exc()
                )

                st.stop()

        # ====================================================
        # COMPLIANCE STATUS
        # ====================================================

        overall_status = safe_string(
            compliance_result.get(
                "overall_status",
                "UNKNOWN",
            )
        ).upper()

        st.subheader(
            "⚖️ Compliance Result"
        )

        if overall_status == "COMPLIANT":

            st.success(
                "🟢 COMPLIANT"
            )

        elif overall_status in (
            "NON-COMPLIANT",
            "NON_COMPLIANT",
        ):

            st.error(
                "🔴 NON-COMPLIANT"
            )

        else:

            st.warning(
                "🟡 NEEDS REVIEW"
            )

        # ====================================================
        # STATISTICS
        # ====================================================

        checks = compliance_result.get(
            "checks",
            [],
        )

        if not isinstance(
            checks,
            list,
        ):
            checks = []

        failed_count = sum(
            1
            for check in checks
            if isinstance(
                check,
                dict,
            )
            and check.get(
                "status"
            ) == "FAIL"
        )

        review_count = sum(
            1
            for check in checks
            if isinstance(
                check,
                dict,
            )
            and check.get(
                "status"
            ) == "REVIEW"
        )

        passed_count = sum(
            1
            for check in checks
            if isinstance(
                check,
                dict,
            )
            and check.get(
                "status"
            ) == "PASS"
        )

        not_applicable_count = sum(
            1
            for check in checks
            if isinstance(
                check,
                dict,
            )
            and check.get(
                "status"
            ) == "NOT_APPLICABLE"
        )

        if failed_count:

            st.error(
                f"❌ {failed_count} requirement(s) failed."
            )

        elif review_count:

            st.warning(
                f"⚠️ {review_count} requirement(s) "
                "require review."
            )

        else:

            st.success(
                "✅ All automatically verifiable "
                "requirements passed."
            )

        # ====================================================
        # DETAILED CHECKS
        # ====================================================

        st.subheader(
            "📊 Detailed Compliance Checks"
        )

        if not checks:

            st.warning(
                "No compliance checks were returned."
            )

        else:

            for index, check in enumerate(
                checks,
                start=1,
            ):

                if not isinstance(
                    check,
                    dict,
                ):
                    continue

                field = check.get(
                    "field",
                    "Unknown",
                )

                status = safe_string(
                    check.get(
                        "status",
                        "UNKNOWN",
                    )
                ).upper()

                message = safe_string(
                    check.get(
                        "message",
                        "",
                    )
                )

                rule_reference = safe_string(
                    check.get(
                        "rule_reference",
                        "",
                    )
                )

                icon = get_status_icon(
                    status
                )

                title = (
                    f"{icon} {field}"
                )

                if rule_reference:

                    title += (
                        f" — {rule_reference}"
                    )

                with st.expander(
                    f"{index}. {title}"
                ):

                    st.write(
                        message
                    )

                    if status == "PASS":

                        st.success(
                            "PASS"
                        )

                    elif status == "FAIL":

                        st.error(
                            "FAIL"
                        )

                    elif status == "REVIEW":

                        st.warning(
                            "REVIEW"
                        )

                    elif status == "NOT_APPLICABLE":

                        st.info(
                            "NOT APPLICABLE"
                        )

                    else:

                        st.info(
                            status
                        )

                    # Show supporting information if
                    # the rule engine provides it.

                    if "value" in check:

                        st.write(
                            "**Detected value:**"
                        )

                        st.code(
                            str(
                                check.get(
                                    "value"
                                )
                            )
                        )

                    if "expected" in check:

                        st.write(
                            "**Expected:**"
                        )

                        st.code(
                            str(
                                check.get(
                                    "expected"
                                )
                            )
                        )

        # ====================================================
        # STEP 6 — REPORT
        # ====================================================

        st.divider()

        st.header(
            "6️⃣ Compliance Report"
        )

        try:

            report = generate_report(
                compliance_result
            )

            if not isinstance(
                report,
                dict,
            ):

                report = {}

        except Exception as error:

            st.warning(
                f"Report generation warning: {error}"
            )

            report = {}

        # ====================================================
        # REPORT METRICS
        # ====================================================

        total_checks = compliance_result.get(
            "total_checks",
            report.get(
                "total_checks",
                len(checks),
            ),
        )

        passed_checks = compliance_result.get(
            "passed",
            report.get(
                "passed",
                passed_count,
            ),
        )

        failed_checks = compliance_result.get(
            "failed",
            report.get(
                "failed",
                failed_count,
            ),
        )

        review_checks = compliance_result.get(
            "review",
            review_count,
        )

        not_applicable_checks = (
            compliance_result.get(
                "not_applicable",
                not_applicable_count,
            )
        )

        col1, col2, col3, col4, col5 = (
            st.columns(5)
        )

        with col1:

            st.metric(
                "Total Checks",
                total_checks,
            )

        with col2:

            st.metric(
                "Passed",
                passed_checks,
            )

        with col3:

            st.metric(
                "Failed",
                failed_checks,
            )

        with col4:

            st.metric(
                "Review",
                review_checks,
            )

        with col5:

            st.metric(
                "Not Applicable",
                not_applicable_checks,
            )

        # ====================================================
        # RAW COMPLIANCE JSON
        # ====================================================

        with st.expander(
            "⚖️ View Complete Compliance Data"
        ):

            st.json(
                compliance_result
            )

        # ====================================================
        # SAVE SCAN
        # ====================================================

        st.divider()

        st.subheader(
            "💾 Scan History"
        )

        try:

            save_scan(
                product_data,
                compliance_result,
            )

            st.success(
                "💾 Scan saved to history."
            )

        except Exception as error:

            st.warning(
                f"Scan could not be saved: {error}"
            )

        # ====================================================
        # FINAL SUMMARY
        # ====================================================

        st.divider()

        st.header(
            "📌 Final Scan Summary"
        )

        summary_col1, summary_col2 = (
            st.columns(2)
        )

        # ----------------------------------------------------
        # SCANNER SUMMARY
        # ----------------------------------------------------

        with summary_col1:

            st.write(
                "### 🔍 Scanner"
            )

            if label_detected:

                st.write(
                    f"✅ Label detected "
                    f"({label_confidence}%)"
                )

            else:

                st.write(
                    "⚠️ Label boundary was not "
                    "confidently detected."
                )

            st.write(
                f"Image quality: "
                f"**{quality_score}/100**"
            )

            st.write(
                f"OCR detections/text available: "
                f"**{'Yes' if extracted_text else 'No'}**"
            )

            st.write(
                f"Important fields extracted: "
                f"**{completeness['found']}/"
                f"{completeness['total']}**"
            )

        # ----------------------------------------------------
        # COMPLIANCE SUMMARY
        # ----------------------------------------------------

        with summary_col2:

            st.write(
                "### ⚖️ Compliance"
            )

            if overall_status == "COMPLIANT":

                st.write(
                    "🟢 **COMPLIANT**"
                )

            elif overall_status in (
                "NON-COMPLIANT",
                "NON_COMPLIANT",
            ):

                st.write(
                    "🔴 **NON-COMPLIANT**"
                )

            else:

                st.write(
                    "🟡 **NEEDS REVIEW**"
                )

            st.write(
                f"Passed: **{passed_checks}**"
            )

            st.write(
                f"Failed: **{failed_checks}**"
            )

            st.write(
                f"Review: **{review_checks}**"
            )

            st.write(
                f"Not applicable: "
                f"**{not_applicable_checks}**"
            )

        # ====================================================
        # DOWNLOAD OCR TEXT
        # ====================================================

        st.divider()

        st.subheader(
            "📥 Export"
        )

        col1, col2 = st.columns(
            2
        )

        with col1:

            st.download_button(
                label="📄 Download OCR Text",
                data=normalized_text,
                file_name="ocr_text.txt",
                mime="text/plain",
                width="stretch",
            )

        with col2:

            try:

                json_data = json.dumps(
                    {
                        "product_data": product_data,
                        "compliance": compliance_result,
                        "ocr_text": normalized_text,
                    },
                    indent=4,
                    ensure_ascii=False,
                    default=str,
                )

                st.download_button(
                    label="📦 Download Analysis JSON",
                    data=json_data,
                    file_name="compliance_analysis.json",
                    mime="application/json",
                    width="stretch",
                )

            except Exception:

                pass

        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.divider()

        st.caption(
            "⚠️ This system is an AI-assisted screening tool. "
            "Its output should be verified against the applicable "
            "Legal Metrology requirements before legal or "
            "regulatory action."
        )