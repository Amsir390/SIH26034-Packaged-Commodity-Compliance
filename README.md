
# SIH26034 Enhanced Features

## Added features

1. Image quality assessment
2. Advanced image preprocessing kept separate from OCR
3. Multi-image package input (front/back/side)
4. OCR bounding-box evidence visualization
5. Text normalization
6. Semantic/context field classification
7. Field-level confidence and review flags
8. Missing-declaration screening
9. Combined text from multiple package views
10. Human verification stage

## Files

- `modules/image_processing.py`
- `modules/compliance_enhancements.py`
- `app_enhanced.py`

## Integration

1. Copy `compliance_enhancements.py` into your existing `modules/` folder.
2. Replace your current `image_processing.py` with the supplied advanced version if you want the standalone preprocessing functions.
3. Keep your existing `scanner.py`, `ocr.py`, `ai_extraction.py`, `rule_engine.py` and `report.py`.
4. Test `app_enhanced.py` first. If it works with your current modules, rename it to `app.py`.

The enhanced app intentionally does not add a new database dependency.
