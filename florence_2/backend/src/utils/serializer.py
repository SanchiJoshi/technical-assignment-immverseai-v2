"""JSON metadata serializer for manuscript layout region predictions."""

import json
import os
import time
from typing import List, Dict, Any, Optional


class LayoutSerializer:
    """Serializes region detection predictions into structured JSON output."""

    @staticmethod
    def format_output(
        image_name: str,
        image_shape: tuple,
        regions: List[Dict[str, Any]],
        substrate_type: str,
        page_bbox: List[int],
        processing_time_ms: float,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Formats layout predictions into a clean, standard JSON payload.

        Args:
            image_name: Base filename of the processed image.
            image_shape: Tuple of (height, width, channels).
            regions: List of detected region dictionaries.
            substrate_type: 'palm_leaf' or 'paper'.
            page_bbox: [x1, y1, x2, y2] physical page bounding box.
            processing_time_ms: Total latency in milliseconds.
            extra_metadata: Optional additional metadata fields.

        Returns:
            Dictionary matching the required submission metadata format.
        """
        h, w = image_shape[:2]
        
        # Calculate summary statistics
        class_counts: Dict[str, int] = {
            "header": 0,
            "footer": 0,
            "main_text": 0,
            "side_text": 0,
            "filler": 0
        }
        
        formatted_regions = []
        for idx, reg in enumerate(regions, 1):
            cls_name = reg["class"]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            
            x1, y1, x2, y2 = reg["bbox"]
            formatted_regions.append({
                "id": idx,
                "class": cls_name,
                "confidence": round(float(reg["confidence"]), 4),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "polygon": [
                    [int(x1), int(y1)],
                    [int(x2), int(y1)],
                    [int(x2), int(y2)],
                    [int(x1), int(y2)]
                ],
                "area_px": int((x2 - x1) * (y2 - y1)),
                "description": reg.get("description", "")
            })

        output_data = {
            "version": "1.0.0",
            "task": "manuscript_layout_region_detection",
            "image_metadata": {
                "file_name": image_name,
                "width": int(w),
                "height": int(h),
                "aspect_ratio": round(float(w / max(1, h)), 3),
                "substrate_type": substrate_type,
                "page_boundary": [int(v) for v in page_bbox]
            },
            "summary": {
                "total_regions_detected": len(formatted_regions),
                "class_distribution": class_counts
            },
            "regions": formatted_regions,
            "performance": {
                "processing_time_ms": round(processing_time_ms, 2),
                "device": extra_metadata.get("device", "cpu") if extra_metadata else "cpu"
            }
        }

        return output_data

    @staticmethod
    def save_json(data: Dict[str, Any], output_path: str) -> None:
        """Saves dictionary to formatted JSON file.

        Args:
            data: Data dictionary to export.
            output_path: Path to target .json file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
