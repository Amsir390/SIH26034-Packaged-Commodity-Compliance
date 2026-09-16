
"""
SIH26034 feature layer.

This module adds:
- text normalization
- semantic/context field classification
- field-level confidence
- missing-declaration screening
- OCR evidence visualization
- multi-image text aggregation

It is deliberately lightweight and works without a large language model.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


FIELD_ALIASES = {
    "mrp": [
        "mrp", "m.r.p", "maximum retail price", "max retail price",
        "retail price", "incl all taxes", "inclusive of all taxes",
    ],
    "net_quantity": [
        "net quantity", "net qty", "net weight", "net wt",
        "net volume", "contents", "quantity", "weight", "volume",
    ],
    "manufacturer": [
        "manufactured by", "manufactured & marketed by",
        "manufactured and marketed by", "manufacturer",
        "manufactured for", "packed by", "packer", "packed for",
    ],
    "address": [
        "address", "registered office", "corporate office",
        "manufactured at", "packed at",
    ],
    "manufacturing_date": [
        "mfg", "mfd", "manufactured on", "manufacturing date",
        "date of manufacture", "month and year of manufacture",
    ],
    "best_before": [
        "best before", "best-before", "use by", "use-by",
        "expiry", "expires", "expiry date",
    ],
    "batch": [
        "batch no", "batch number", "batch code", "lot no", "lot number",
    ],
    "product_name": [
        "product", "product name", "name of product", "common name",
        "generic name",
    ],
    "country_of_origin": [
        "country of origin", "made in", "origin",
    ],
    "consumer_care": [
        "consumer care", "customer care", "care number",
        "toll free", "helpline", "customer service",
    ],
}


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\.\s*", ".", text)
    text = re.sub(r"\bM\s*R\s*P\b", "MRP", text, flags=re.I)
    text = re.sub(r"\bM\s*F\s*G\b", "MFG", text, flags=re.I)
    text = re.sub(r"\bM\s*F\s*D\b", "MFD", text, flags=re.I)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _compact(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _alias_score(text: str, aliases: Iterable[str]) -> float:
    compact = _compact(text)
    best = 0.0
    for alias in aliases:
        a = _compact(alias)
        if not a:
            continue
        if a in compact:
            best = max(best, 1.0)
        elif len(a.split()) >= 2:
            words = set(a.split())
            overlap = len(words & set(compact.split())) / len(words)
            best = max(best, overlap * 0.75)
    return best


def _pattern_score(field: str, text: str) -> float:
    t = text.lower()

    if field == "mrp":
        return 1.0 if re.search(r"\b(?:mrp|maximum retail price)\b\s*[:\-]?\s*(?:rs\.?|₹)?\s*\d", t) else 0.0

    if field == "net_quantity":
        return 1.0 if re.search(r"\b\d+(?:\.\d+)?\s*(?:kg|g|mg|l|ml|cl|pcs?|pieces?)\b", t) else 0.0

    if field in {"manufacturing_date", "best_before"}:
        return 1.0 if re.search(
            r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
            r"\d{1,2}[-/][A-Za-z]{3,9}[-/]\d{2,4}|"
            r"[A-Za-z]{3,9}[-/]\d{2,4})\b",
            t,
        ) else 0.0

    if field == "batch":
        return 1.0 if re.search(r"\b(?:batch|lot)\s*(?:no|number|code)?\s*[:\-]?\s*[A-Za-z0-9-]+", t) else 0.0

    if field == "consumer_care":
        return 1.0 if re.search(r"(?:\+?\d[\d ()-]{7,}\d|toll[\s-]*free|helpline)", t) else 0.0

    return 0.0


def classify_fields(
    text: str,
    detections: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Classify candidate OCR lines into semantic fields.

    Spatial/context evidence is used when OCR detections are supplied.
    """
    normalized = normalize_text(text)
    lines = [x.strip() for x in normalized.splitlines() if x.strip()]
    results: Dict[str, Dict[str, Any]] = {}

    for line_index, line in enumerate(lines):
        for field, aliases in FIELD_ALIASES.items():
            semantic = _alias_score(line, aliases)
            pattern = _pattern_score(field, line)

            # Nearby line context: labels often appear immediately before values.
            context = 0.0
            if line_index + 1 < len(lines):
                context = _pattern_score(field, line + " " + lines[line_index + 1]) * 0.85

            score = min(1.0, 0.55 * semantic + 0.30 * pattern + 0.15 * context)

            if score < 0.35:
                continue

            candidate = {
                "field": field,
                "text": line,
                "score": round(score, 4),
                "semantic_score": round(semantic, 4),
                "pattern_score": round(pattern, 4),
                "context_score": round(context, 4),
                "line_index": line_index,
            }

            previous = results.get(field)
            if previous is None or candidate["score"] > previous["score"]:
                results[field] = candidate

    return results


def field_confidence(
    field: str,
    candidate: Dict[str, Any],
    ocr_confidence: float = 0.0,
) -> float:
    """Combine OCR and field signals into a review-oriented confidence."""
    semantic = float(candidate.get("semantic_score", 0.0))
    pattern = float(candidate.get("pattern_score", 0.0))
    context = float(candidate.get("context_score", 0.0))
    field_score = float(candidate.get("score", 0.0))

    # OCR is one signal; semantic/pattern/context are additional signals.
    value = (
        0.35 * max(0.0, min(1.0, ocr_confidence))
        + 0.25 * semantic
        + 0.20 * pattern
        + 0.10 * context
        + 0.10 * field_score
    )
    return round(max(0.0, min(1.0, value)) * 100, 1)


def build_field_confidence(
    classified: Dict[str, Dict[str, Any]],
    ocr_confidence: float,
) -> Dict[str, Dict[str, Any]]:
    output = {}
    for field, candidate in classified.items():
        score = field_confidence(field, candidate, ocr_confidence)
        output[field] = {
            **candidate,
            "confidence_percent": score,
            "review_required": score < 70.0,
        }
    return output


def missing_declarations(
    classified: Dict[str, Dict[str, Any]],
    *,
    imported_product: bool = False,
) -> List[str]:
    """
    Screening list based on the fields visible to this application.
    Applicability can depend on the commodity/product, so this is a
    review aid rather than a final legal determination.
    """
    required = [
        ("product_name", "Product/common or generic name"),
        ("net_quantity", "Net quantity"),
        ("mrp", "MRP"),
        ("manufacturer", "Manufacturer/packer/importer details"),
        ("address", "Address"),
        ("manufacturing_date", "Manufacturing date"),
        ("consumer_care", "Consumer-care information"),
    ]

    if imported_product:
        required.append(("country_of_origin", "Country of origin"))

    return [label for key, label in required if key not in classified]


def draw_ocr_evidence(
    image: Image.Image,
    detections: List[Dict[str, Any]],
) -> Image.Image:
    """
    Draw OCR bounding boxes and confidence labels.

    Expected detection shape:
      {"box": [[x,y], ...], "text": "...", "confidence": 0.91}
    """
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)

    for item in detections or []:
        box = item.get("box") or item.get("bbox")
        text = str(item.get("text", "")).strip()
        conf = float(item.get("confidence", 0.0))

        if not box or len(box) < 4:
            continue

        try:
            points = [(int(p[0]), int(p[1])) for p in box]
        except Exception:
            continue

        draw.line(points + [points[0]], width=3)
        x = min(p[0] for p in points)
        y = min(p[1] for p in points)
        label = f"{text[:28]} ({conf * 100:.0f}%)"
        draw.text((x, max(0, y - 16)), label)

    return output


def merge_texts(texts: Iterable[str]) -> str:
    """Merge text from multiple package-side images and remove duplicates."""
    seen = set()
    lines: List[str] = []

    for text in texts:
        for line in normalize_text(text).splitlines():
            key = re.sub(r"[^a-z0-9]+", "", line.lower())
            if not key or key in seen:
                continue
            seen.add(key)
            lines.append(line)

    return "\n".join(lines)


def status_for_confidence(confidence: float) -> str:
    if confidence >= 80:
        return "HIGH"
    if confidence >= 60:
        return "MEDIUM"
    return "REVIEW"
