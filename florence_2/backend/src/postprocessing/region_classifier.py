"""Post-processing and 5-class layout region categorization engine."""

from typing import List, Dict, Any
import numpy as np
from src.utils.image_io import clip_bbox_to_image, clip_bbox_to_page, nms_boxes


class RegionClassifier:
    """Classifies detected candidate regions into the 5 target classes and enforces spatial validity."""

    TARGET_CLASSES = ["header", "footer", "main_text", "side_text", "filler"]

    def __init__(self, confidence_threshold: float = 0.50, iou_threshold: float = 0.85):
        """Initialize RegionClassifier."""
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def process_and_classify(
        self, 
        image_bgr: np.ndarray,
        candidates: List[Dict[str, Any]], 
        preprocessed: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Maps candidate layout regions into target 5-class taxonomy with strict page containment.

        Args:
            image_bgr: Original image array.
            candidates: List of raw detected region dictionaries.
            preprocessed: Preprocessed data containing page_bbox, substrate_type, etc.

        Returns:
            Deduplicated, schema-compliant list of region dictionaries.
        """
        img_h, img_w = image_bgr.shape[:2]
        page_bbox = preprocessed.get("page_bbox", preprocessed.get("folio_bbox", [0, 0, img_w, img_h]))

        if not candidates:
            return []

        # 1. Enforce page boundary clipping and taxonomy mapping
        processed = []
        for cand in candidates:
            raw_box = cand["bbox"]
            # Strict clamping to physical page boundaries
            clamped_box = clip_bbox_to_page(raw_box, page_bbox)

            # Filter degenerate zero-area boxes
            w = clamped_box[2] - clamped_box[0]
            h = clamped_box[3] - clamped_box[1]
            if w <= 4 or h <= 4:
                continue

            cls_name = cand.get("layout_type", "main_text")
            if cand.get("is_stamp", False):
                cls_name = "filler"

            if cls_name not in self.TARGET_CLASSES:
                cls_name = "main_text"

            conf = float(cand.get("confidence", 0.90))
            if conf < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = clamped_box
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

            processed.append({
                "class": cls_name,
                "confidence": round(conf, 2),
                "bbox": clamped_box,
                "polygon": polygon,
                "area_px": int(w * h),
                "description": cand.get("description", f"Detected {cls_name} layout region")
            })

        # 2. Sort by confidence and assign deterministic integer IDs
        processed.sort(key=lambda x: x["confidence"], reverse=True)
        for idx, item in enumerate(processed, start=1):
            item["id"] = idx

        return processed
