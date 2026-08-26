"""Image I/O, format conversion, and spatial geometry utilities."""

import os
from typing import Tuple, List, Optional
import cv2
import numpy as np


def load_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads an image from disk in both BGR and RGB formats safely.

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        Tuple of (bgr_image, rgb_image) as numpy ndarrays.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file exists but cannot be decoded as an image.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Handle paths with special characters/spaces safely via numpy buffer
    with open(image_path, "rb") as f:
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        bgr_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if bgr_image is None:
        raise ValueError(f"Failed to decode image from path: {image_path}")

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return bgr_image, rgb_image


def save_image(image_bgr: np.ndarray, output_path: str) -> bool:
    """Saves a BGR numpy image to disk, creating parent directories if needed.

    Args:
        image_bgr: Image array in BGR format.
        output_path: Destination path.

    Returns:
        True if saved successfully, False otherwise.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ext = os.path.splitext(output_path)[1]
    success, enc_img = cv2.imencode(ext if ext else ".png", image_bgr)
    if success:
        with open(output_path, "wb") as f:
            f.write(enc_img)
        return True
    return False


def clip_bbox_to_image(bbox: List[int], img_width: int, img_height: int) -> List[int]:
    """Ensures bounding box coordinates remain strictly within image dimensions.

    Args:
        bbox: [x1, y1, x2, y2].
        img_width: Image width (W).
        img_height: Image height (H).

    Returns:
        Clamped [x1, y1, x2, y2] with x1 <= x2 and y1 <= y2.
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), img_width))
    y1 = max(0, min(int(y1), img_height))
    x2 = max(0, min(int(x2), img_width))
    y2 = max(0, min(int(y2), img_height))

    # Ensure coordinate ordering
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    return [x1, y1, x2, y2]


def clip_bbox_to_page(bbox: List[int], page_bounds: List[int]) -> List[int]:
    """Ensures bounding box coordinates remain strictly within physical page bounds.

    Args:
        bbox: [x1, y1, x2, y2].
        page_bounds: [page_x1, page_y1, page_x2, page_y2].

    Returns:
        Clamped [x1, y1, x2, y2] strictly within [page_x1, page_y1, page_x2, page_y2].
    """
    px1, py1, px2, py2 = page_bounds
    x1, y1, x2, y2 = bbox

    x1 = max(px1, min(int(x1), px2))
    y1 = max(py1, min(int(y1), py2))
    x2 = max(px1, min(int(x2), px2))
    y2 = max(py1, min(int(y2), py2))

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    return [x1, y1, x2, y2]


def compute_iou(box_a: List[int], box_b: List[int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes.

    Args:
        box_a: [x1, y1, x2, y2].
        box_b: [x1, y1, x2, y2].

    Returns:
        IoU value in range [0.0, 1.0].
    """
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0

    return float(inter_area) / float(union_area)


def nms_boxes(
    boxes: List[List[int]], 
    scores: List[float], 
    iou_threshold: float = 0.50
) -> List[int]:
    """Applies Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes.

    Args:
        boxes: List of [x1, y1, x2, y2].
        scores: Corresponding confidence scores.
        iou_threshold: Overlap threshold above which suppressed.

    Returns:
        Indices of retained bounding boxes.
    """
    if not boxes:
        return []

    boxes_arr = np.array(boxes, dtype=np.float32)
    scores_arr = np.array(scores, dtype=np.float32)

    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 2]
    y2 = boxes_arr[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores_arr.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep
