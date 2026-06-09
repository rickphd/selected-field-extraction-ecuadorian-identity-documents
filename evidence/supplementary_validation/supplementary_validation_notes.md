# Supplementary OCR and YOLO Evidence Package

This package reports module-level internal validation evidence for the OCR and YOLOv11n-OBB components.

## Scope

This package supports only module-level internal validation claims. It does not support independent held-out-test claims, leakage-free generalization claims, per-class detector robustness, field-level exact-match accuracy, full-record exact-match accuracy, or end-to-end information-extraction accuracy.

## Files

- `supp_fig_s1_ocr_val_cer_curves.pdf`
- `supp_fig_s2_yolo_validation_metrics.pdf`
- `supp_fig_s3_yolo_terminal_stability.pdf`
- `supplementary_validation_notes.md`
- `supplementary_validation_notes.pdf`

## Generated Tables

- Supplementary Table S1: OCR validation summary for complete OCR runs.
- Supplementary Table S2: YOLOv11n-OBB internal validation-history maxima and terminal stability.
- Supplementary Table S3: Validation evidence summary for OCR and YOLOv11n-OBB.

## Generated Figures

- Supplementary Figure S1: OCR validation CER trajectories.
- Supplementary Figure S2: YOLO validation metric trajectories over 141 epochs.
- Supplementary Figure S3: YOLO last-10-epoch terminal stability.

## Interpretive Note

The OCR validation results provide direct evidence for the OCR comparison under internal crop-level validation. Among the complete OCR runs, the Transformer-based model achieved the lowest validation CER, reaching 2.032055% at checkpoint 111 and ending at 2.045730% at checkpoint 143. This result supports a module-level OCR claim for cropped text regions only. It should not be interpreted as field-level exact-match accuracy, full-record exact-match accuracy, held-out-test performance, or end-to-end extraction accuracy.

The YOLOv11n-OBB validation results support the reported detector maxima: precision = 0.99499, recall = 0.98371, mAP50 = 0.99396, and mAP50-95 = 0.88233. The final 10 epochs show stable mAP values near the reported maxima, with mAP50 mean = 0.993421 and mAP50-95 mean = 0.881679. These values characterize the configured detector under internal validation and do not establish independent held-out-test performance, per-class robustness, or end-to-end text-extraction accuracy.
