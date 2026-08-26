"""Substrate detection, physical page boundary extraction, and contrast normalization."""

from typing import Tuple, Dict, Any, List
import cv2
import numpy as np
from src.utils.image_io import clip_bbox_to_image


class FolioExtractor:
    """Isolates the physical manuscript folio/page from the digitization background."""

    def __init__(self, clahe_clip_limit: float = 2.0, clahe_grid_size: Tuple[int, int] = (8, 8)):
        """Initialize FolioExtractor."""
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)

    def detect_substrate_type(self, image_bgr: np.ndarray) -> str:
        """Determines substrate type based on aspect ratio and color variance."""
        h, w = image_bgr.shape[:2]
        aspect_ratio = w / float(h)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        sat_mean = np.mean(hsv[:, :, 1])

        if aspect_ratio >= 2.30 and sat_mean > 15:
            return "palm_leaf"
        return "paper"

    def extract_page_boundary(self, image_bgr: np.ndarray) -> List[int]:
        """Extracts the exact physical page/folio bounding box [fx1, fy1, fx2, fy2]."""
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0]
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        s_chan = hsv[:, :, 1]

        # Measure 4 corner background levels
        corn_size = max(6, min(25, int(min(h, w) * 0.05)))
        corners = [
            l_chan[0:corn_size, 0:corn_size],
            l_chan[0:corn_size, -corn_size:],
            l_chan[-corn_size:, 0:corn_size],
            l_chan[-corn_size:, -corn_size:]
        ]
        mean_corner = float(np.mean(corners))
        center_crop = l_chan[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]
        mean_center = float(np.mean(center_crop))

        raw_bbox = [0, 0, w, h]

        # Case 1: Dark Background (slate board, black book spine, dark scanning bed)
        if mean_corner < 120 and (mean_center - mean_corner) > 20:
            blurred = cv2.GaussianBlur(l_chan, (15, 15), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                largest = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(largest) > (w * h * 0.20):
                    x, y, bw, bh = cv2.boundingRect(largest)
                    margin_x = max(3, int(bw * 0.01))
                    margin_y = max(3, int(bh * 0.01))
                    raw_bbox = [
                        max(0, x + margin_x), 
                        max(0, y + margin_y), 
                        min(w, x + bw - margin_x), 
                        min(h, y + bh - margin_y)
                    ]
                    # Tight edge refinement
                    raw_bbox = self._refine_dark_background_edges(image_bgr, raw_bbox)

        # Case 2: Light Background (light desk / white scanner background with darker manuscript)
        elif mean_corner > 165 and mean_center < 165:
            blurred_s = cv2.GaussianBlur(s_chan, (11, 11), 0)
            _, s_th = cv2.threshold(blurred_s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            closed_s = cv2.morphologyEx(s_th, cv2.MORPH_CLOSE, k)
            cnts, _ = cv2.findContours(closed_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                largest_s = max(cnts, key=cv2.contourArea)
                if (w * h * 0.25) < cv2.contourArea(largest_s) < (w * h * 0.95):
                    x, y, bw, bh = cv2.boundingRect(largest_s)
                    margin_x = max(3, int(bw * 0.01))
                    margin_y = max(3, int(bh * 0.01))
                    raw_bbox = [
                        max(0, x + margin_x), 
                        max(0, y + margin_y), 
                        min(w, x + bw - margin_x), 
                        min(h, y + bh - margin_y)
                    ]

        return clip_bbox_to_image(raw_bbox, w, h)

    def _refine_dark_background_edges(self, img_bgr: np.ndarray, bbox: List[int]) -> List[int]:
        """Refines bounding box edges by trimming any residual dark background rows/columns."""
        x1, y1, x2, y2 = bbox
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return bbox

        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0]

        row_means = np.mean(l_chan, axis=1)
        col_means = np.mean(l_chan, axis=0)

        bg_val = 65
        valid_rows = np.where(row_means > bg_val)[0]
        valid_cols = np.where(col_means > bg_val)[0]

        if len(valid_rows) > 0 and len(valid_cols) > 0:
            new_x1 = x1 + int(valid_cols[0]) + 2
            new_x2 = x1 + int(valid_cols[-1]) - 2
            new_y1 = y1 + int(valid_rows[0]) + 2
            new_y2 = y1 + int(valid_rows[-1]) - 2
            return [new_x1, new_y1, max(new_x1 + 10, new_x2), max(new_y1 + 10, new_y2)]

        return bbox

    def enhance_contrast(self, image_bgr: np.ndarray) -> np.ndarray:
        """Applies CLAHE on the LAB L-channel."""
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        enhanced_l = self.clahe.apply(l_chan)
        merged = cv2.merge([enhanced_l, a_chan, b_chan])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def extract_ink_mask(self, image_bgr: np.ndarray, page_bbox: List[int]) -> np.ndarray:
        """Extracts binarized ink mask within the physical page boundary."""
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        ink_mask = np.zeros((h, w), dtype=np.uint8)

        px1, py1, px2, py2 = page_bbox
        page_crop = gray[py1:py2, px1:px2]
        if page_crop.size == 0:
            return ink_mask

        th = cv2.adaptiveThreshold(
            page_crop,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            6
        )
        ink_mask[py1:py2, px1:px2] = th
        return ink_mask

    def process(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """Runs the complete preprocessing and page isolation stage."""
        h, w = image_bgr.shape[:2]
        substrate_type = self.detect_substrate_type(image_bgr)
        page_bbox = self.extract_page_boundary(image_bgr)
        enhanced_bgr = self.enhance_contrast(image_bgr)
        ink_mask = self.extract_ink_mask(image_bgr, page_bbox)

        return {
            "substrate_type": substrate_type,
            "page_bbox": page_bbox,
            "folio_bbox": page_bbox,  # Alias for compatibility
            "enhanced_image": enhanced_bgr,
            "ink_mask": ink_mask,
            "image_dims": (w, h)
        }
