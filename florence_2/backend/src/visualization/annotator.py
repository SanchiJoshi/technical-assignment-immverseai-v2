"""Visualizer module for generating publication-quality annotated layout maps."""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


class LayoutVisualizer:
    """Renders alpha-blended bounding boxes, class labels, and legends onto manuscript images."""

    COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
        "main_text": (50, 205, 50),     # Emerald Green
        "header":    (235, 140, 30),    # Dodger Blue
        "footer":    (211, 0, 148),     # Vivid Purple / Violet
        "side_text": (0, 140, 255),     # Amber Orange
        "filler":    (50, 50, 255)      # Crimson Red
    }

    def __init__(self, alpha: float = 0.18, font_scale: float = 0.40, line_thickness: int = 2):
        """Initialize LayoutVisualizer."""
        self.alpha = alpha
        self.font_scale = font_scale
        self.line_thickness = line_thickness

    def draw_legend(self, image: np.ndarray) -> np.ndarray:
        """Renders a compact, translucent legend box in top corner."""
        h, w = image.shape[:2]
        legend_items = [
            ("Main Text", "main_text"),
            ("Header", "header"),
            ("Footer", "footer"),
            ("Side Text", "side_text"),
            ("Filler", "filler")
        ]

        box_w = 110
        box_h = len(legend_items) * 16 + 10
        lx1 = w - box_w - 8
        ly1 = 8
        lx2 = w - 8
        ly2 = ly1 + box_h

        if lx1 < 0 or ly1 < 0:
            return image

        overlay = image.copy()
        cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)
        cv2.rectangle(image, (lx1, ly1), (lx2, ly2), (100, 100, 100), 1)

        for idx, (label, cls_key) in enumerate(legend_items):
            color = self.COLOR_MAP.get(cls_key, (200, 200, 200))
            item_y = ly1 + 13 + (idx * 16)
            cv2.rectangle(image, (lx1 + 6, item_y - 7), (lx1 + 16, item_y + 3), color, -1)
            cv2.rectangle(image, (lx1 + 6, item_y - 7), (lx1 + 16, item_y + 3), (255, 255, 255), 1)
            cv2.putText(
                image, 
                label, 
                (lx1 + 22, item_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.32, 
                (245, 245, 245), 
                1, 
                cv2.LINE_AA
            )

        return image

    def annotate(
        self, 
        image_bgr: np.ndarray, 
        regions: List[Dict[str, Any]], 
        include_legend: bool = True
    ) -> np.ndarray:
        """Draws color-coded bounding boxes, collision-free confidence badges, and labels."""
        canvas = image_bgr.copy()
        overlay = canvas.copy()
        img_h, img_w = canvas.shape[:2]

        # 1. Semi-transparent highlight masks
        for reg in regions:
            cls_name = reg["class"]
            color = self.COLOR_MAP.get(cls_name, (0, 255, 255))
            x1, y1, x2, y2 = reg["bbox"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        cv2.addWeighted(overlay, self.alpha, canvas, 1 - self.alpha, 0, canvas)

        # 2. Draw solid boundary outlines and collision-free label badges
        for reg in regions:
            cls_name = reg["class"]
            conf = reg.get("confidence", 1.0)
            color = self.COLOR_MAP.get(cls_name, (0, 255, 255))
            x1, y1, x2, y2 = reg["bbox"]

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, self.line_thickness)

            label_str = f"{cls_name}: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)

            # Smart placement based on class and spatial role
            if cls_name == "main_text":
                badge_x1 = max(0, x1 + 80)
                badge_y1 = min(img_h - th - 8, y1 + 4)
            elif cls_name == "header":
                badge_x1 = x1 + 4
                badge_y1 = min(img_h - th - 8, y1 + 3)
            elif cls_name == "footer":
                badge_x1 = max(0, x2 - tw - 16)
                badge_y1 = max(0, y2 - th - 6)
            elif cls_name == "side_text":
                badge_x1 = x1 + 2
                badge_y1 = min(img_h - th - 8, y1 + 4)
            else:
                # Filler / watermark stamps
                badge_x1 = max(0, x1 + 4)
                badge_y1 = max(0, y1 - th - 4) if (y1 - th - 4) >= 0 else y1 + 2

            badge_y2 = badge_y1 + th + 5
            badge_x2 = min(img_w, badge_x1 + tw + 8)

            cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), (20, 20, 20), -1)
            cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1)

            cv2.putText(
                canvas,
                label_str,
                (badge_x1 + 4, badge_y1 + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # 3. Render corner legend
        if include_legend:
            canvas = self.draw_legend(canvas)

        return canvas
