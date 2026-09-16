
import streamlit as st
from PIL import Image

from modules.scanner import process_scan
from modules.ocr import extract_text_with_details
from modules.ai_extraction import extract_product_details
from modules.rule_engine import check_compliance
from modules.report import generate_report

try:
    from modules.compliance_enhancements import (
        normalize_text,
        classify_fields,
        build_field_confidence,
        missing_declarations,
        draw_ocr_evidence,
        merge_texts,
        status_for_confidence,
    )
except ImportError:
    st.error(
        "Missing modules/compliance_enhancements.py. "
        "Copy the supplied feature module into your modules folder."
    )
    st.stop()

st.set_page_config(
    page_title="Packaged Commodity Compliance",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Packaged Commodity Compliance")
st.caption(
    "AI-assisted packaged commodity label screening and "
    "Legal Metrology rule checking."
)

st.info(
    "The system provides screening support. "
    "Final regulatory verification remains with the authorized inspector."
)

st.divider()

# ============================================================
# INPUT
# ============================================================

st.subheader("📷 Product Scanner")

input_mode = st.radio(
    "Choose input method",
    ["📷 Camera Scanner", "📁 Upload Image(s)"],
    horizontal=True,
)

uploaded_files = []

if input_mode == "📷 Camera Scanner":
    camera_file = st.camera_input("Point the camera at the product label")
    if camera_file is not None:
        uploaded_files = [camera_file]
else:
    uploaded_files = st.file_uploader(
        "Upload one or more images of the package",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="For better coverage, upload front, back or side views of the same package.",
    )

if uploaded_files:
    images = []
    for file in uploaded_files:
        try:
            images.append(Image.open(file).convert("RGB"))
        except Exception as exc:
            st.warning(f"Could not read {getattr(file, 'name', 'image')}: {exc}")

    if not images:
        st.stop()

    st.subheader("🖼️ Input Images")

    cols = st.columns(min(3, len(images)))
    for i, image in enumerate(images):
        with cols[i % len(cols)]:
            st.image(image, caption=f"Package view {i + 1}", width="stretch")

    st.divider()

    if st.button("🔍 Analyze Package", type="primary", width="stretch"):
        all_texts = []
        all_detections = []
        scan_results = []

        # ========================================================
        # STEP 1 — SCANNER + IMAGE QUALITY
        # ========================================================
        st.subheader("📊 1. Scan & Image Quality")

        for index, image in enumerate(images):
            with st.spinner(f"Analyzing package view {index + 1}..."):
                try:
                    result = process_scan(image)
                    scan_results.append(result)
                except Exception as exc:
                    st.error(f"Scanner error on image {index + 1}: {exc}")
                    continue

        if not scan_results:
            st.stop()

        for index, result in enumerate(scan_results):
            quality = result.get("quality", {})
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("Quality", f"{quality.get('score', 0)}/100")
            with c2:
                st.metric("Label Detection", f"{result.get('label_confidence', 0)}%")
            with c3:
                st.metric("Sharpness", f"{quality.get('blur_score', 0):.0f}")
            with c4:
                st.metric("Glare", f"{quality.get('glare_ratio', 0):.1f}%")

            status = quality.get("status", "UNKNOWN")
            if status == "GOOD":
                st.success(f"View {index + 1}: image quality is good for processing.")
            elif status == "ACCEPTABLE":
                st.warning(f"View {index + 1}: acceptable quality; verify uncertain results.")
            else:
                st.error(f"View {index + 1}: poor image quality; recapture is recommended.")

            if quality.get("warnings"):
                with st.expander(f"⚠️ Quality warnings — view {index + 1}"):
                    for warning in quality["warnings"]:
                        st.write("•", warning)

            processed = result.get("image")
            if processed is not None:
                st.image(
                    processed,
                    caption=f"Processed package view {index + 1}",
                    width="stretch",
                )

        # ========================================================
        # STEP 2 — OCR WITH DETAILS
        # ========================================================
        st.divider()
        st.subheader("🔤 2. OCR & Evidence")

        for index, result in enumerate(scan_results):
            ocr_image = result.get("ocr_image") or result.get("image")
            if ocr_image is None:
                continue

            with st.spinner(f"Reading text from package view {index + 1}..."):
                try:
                    ocr_result = extract_text_with_details(ocr_image)
                except Exception as exc:
                    st.error(f"OCR error on image {index + 1}: {exc}")
                    continue

            all_texts.append(ocr_result.get("text", ""))
            all_detections.extend(ocr_result.get("detections", []))

            avg_conf = ocr_result.get("average_confidence_percent", 0)
            st.metric(f"View {index + 1} OCR confidence", f"{avg_conf}%")

            evidence = draw_ocr_evidence(
                ocr_image,
                ocr_result.get("detections", []),
            )
            st.image(
                evidence,
                caption=f"OCR evidence — view {index + 1}",
                width="stretch",
            )

            with st.expander(f"View OCR text — package view {index + 1}"):
                st.text(ocr_result.get("text", "") or "No readable text detected.")

        # ========================================================
        # STEP 3 — NORMALIZATION
        # ========================================================
        st.divider()
        st.subheader("🧹 3. Text Normalization")

        combined_text = merge_texts(all_texts)
        combined_text = normalize_text(combined_text)

        st.text_area(
            "Normalized label text",
            combined_text,
            height=220,
        )

        # ========================================================
        # STEP 4 — SEMANTIC FIELD CLASSIFICATION
        # ========================================================
        st.divider()
        st.subheader("🧠 4. Semantic Field Classification")

        classified = classify_fields(combined_text)

        if classified:
            rows = []
            for field, item in classified.items():
                rows.append(
                    {
                        "Field": field.replace("_", " ").title(),
                        "Candidate": item.get("text", ""),
                        "Semantic": round(item.get("semantic_score", 0) * 100, 1),
                        "Pattern": round(item.get("pattern_score", 0) * 100, 1),
                        "Context": round(item.get("context_score", 0) * 100, 1),
                    }
                )
            st.dataframe(rows, width="stretch")
        else:
            st.warning("No semantic field candidates were identified.")

        # ========================================================
        # STEP 5 — CONFIDENCE
        # ========================================================
        st.divider()
        st.subheader("📈 5. Field Confidence & Review")

        avg_ocr = (
            sum(
                r.get("average_confidence", 0.0)
                for r in []
            )
            if False
            else 0.0
        )

        # Use the OCR detections available across all views.
        if all_detections:
            avg_ocr = sum(
                float(d.get("confidence", 0.0))
                for d in all_detections
            ) / len(all_detections)

        field_scores = build_field_confidence(classified, avg_ocr)

        if field_scores:
            rows = []
            for field, item in field_scores.items():
                conf = item["confidence_percent"]
                rows.append(
                    {
                        "Field": field.replace("_", " ").title(),
                        "Value/Candidate": item.get("text", ""),
                        "Confidence": f"{conf:.1f}%",
                        "Status": status_for_confidence(conf),
                    }
                )
            st.dataframe(rows, width="stretch")
        else:
            st.info("No field confidence scores are available.")

        # ========================================================
        # STEP 6 — MISSING DECLARATION SCREENING
        # ========================================================
        st.divider()
        st.subheader("🔎 6. Missing Declaration Screening")

        imported_product = st.checkbox(
            "Treat product as imported for this screening",
            value=False,
        )

        missing = missing_declarations(
            classified,
            imported_product=imported_product,
        )

        if missing:
            st.warning("Potentially missing / not detected declarations:")
            for item in missing:
                st.write("•", item)
        else:
            st.success("All fields in the configured screening list were detected.")

        # ========================================================
        # STEP 7 — EXISTING STRUCTURED EXTRACTION
        # ========================================================
        st.divider()
        st.subheader("📋 7. Structured Product Information")

        with st.spinner("Structuring the extracted label information..."):
            try:
                product_data = extract_product_details(combined_text)
            except Exception as exc:
                st.error(f"AI extraction error: {exc}")
                st.stop()

        left, right = st.columns(2)
        with left:
            st.write("**Product Name**")
            st.info(product_data.get("product_name") or "Not found")

            st.write("**MRP**")
            st.info(product_data.get("mrp") or "Not found")

            st.write("**Net Quantity**")
            st.info(product_data.get("net_quantity") or "Not found")

        with right:
            st.write("**Manufacturer / Packer**")
            st.info(product_data.get("manufacturer") or "Not found")

            st.write("**Address**")
            st.info(product_data.get("address") or "Not found")

        with st.expander("View complete extracted data"):
            st.json(product_data)

        # ========================================================
        # STEP 8 — EXISTING RULE ENGINE
        # ========================================================
        st.divider()
        st.subheader("⚖️ 8. Legal Metrology Compliance Screening")

        with st.spinner("Checking configured compliance rules..."):
            try:
                compliance_result = check_compliance(
                    product_data,
                    combined_text,
                )
            except Exception as exc:
                st.error(f"Compliance engine error: {exc}")
                st.stop()

        overall = compliance_result.get("overall_status", "UNKNOWN")

        if overall == "COMPLIANT":
            st.success("🟢 COMPLIANT")
        elif overall == "NON-COMPLIANT":
            st.error("🔴 NON-COMPLIANT")
        else:
            st.warning("🟡 NEEDS REVIEW")

        checks = compliance_result.get("checks", [])

        for check in checks:
            field = check.get("field", "Unknown")
            status = check.get("status", "UNKNOWN")
            message = check.get("message", "")
            reference = check.get("rule_reference", "")

            label = field + (f" [{reference}]" if reference else "")

            if status == "PASS":
                st.success(f"✅ {label} — {message}")
            elif status == "FAIL":
                st.error(f"❌ {label} — {message}")
            elif status == "NOT_APPLICABLE":
                st.info(f"⚪ {label} — {message}")
            else:
                st.warning(f"⚠️ {label} — {message}")

        # ========================================================
        # STEP 9 — REPORT
        # ========================================================
        st.divider()
        st.subheader("📄 9. Compliance Report")

        try:
            report = generate_report(compliance_result)
        except Exception:
            report = {}

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Checks", compliance_result.get("total_checks", report.get("total_checks", len(checks))))
        with c2:
            st.metric("Passed", compliance_result.get("passed", report.get("passed", 0)))
        with c3:
            st.metric("Failed", compliance_result.get("failed", report.get("failed", 0)))
        with c4:
            st.metric("Review", compliance_result.get("review", 0))

        # ========================================================
        # FINAL VERIFICATION
        # ========================================================
        st.divider()
        st.subheader("👤 10. Final Verification")

        st.warning(
            "Review flagged fields, OCR evidence and rule results before "
            "taking any regulatory action."
        )

        if field_scores:
            review_fields = [
                field.replace("_", " ").title()
                for field, item in field_scores.items()
                if item.get("review_required")
            ]
            if review_fields:
                st.write("**Fields recommended for review:**")
                for field in review_fields:
                    st.write("•", field)

        st.caption(
            "This application is an AI-assisted screening and decision-support "
            "tool. It does not replace verification by an authorized person."
        )
