"""Unit tests for utility functions and geometry operations."""

import os
import sys
import pytest
import numpy as np

# Ensure backend root is on sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(TEST_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.utils.image_io import clip_bbox_to_image, clip_bbox_to_page, compute_iou, nms_boxes


def test_clip_bbox_to_image():
    # Outside top-left
    bbox = [-10, -20, 100, 150]
    clamped = clip_bbox_to_image(bbox, 200, 200)
    assert clamped == [0, 0, 100, 150]

    # Outside bottom-right
    bbox = [50, 60, 250, 300]
    clamped = clip_bbox_to_image(bbox, 200, 200)
    assert clamped == [50, 60, 200, 200]

    # Inverted coordinates
    bbox = [100, 100, 50, 50]
    clamped = clip_bbox_to_image(bbox, 200, 200)
    assert clamped[0] <= clamped[2]
    assert clamped[1] <= clamped[3]


def test_clip_bbox_to_page():
    page_bbox = [50, 40, 400, 300]

    # Fully inside
    bbox = [100, 80, 250, 200]
    clamped = clip_bbox_to_page(bbox, page_bbox)
    assert clamped == [100, 80, 250, 200]

    # Spanning outside left and top
    bbox = [20, 10, 200, 150]
    clamped = clip_bbox_to_page(bbox, page_bbox)
    assert clamped == [50, 40, 200, 150]

    # Spanning outside right and bottom
    bbox = [300, 250, 450, 350]
    clamped = clip_bbox_to_page(bbox, page_bbox)
    assert clamped == [300, 250, 400, 300]


def test_compute_iou():
    box_a = [0, 0, 100, 100]
    box_b = [0, 0, 100, 100]
    assert compute_iou(box_a, box_b) == 1.0

    box_c = [200, 200, 300, 300]
    assert compute_iou(box_a, box_c) == 0.0

    box_d = [50, 0, 150, 100]
    iou = compute_iou(box_a, box_d)
    assert 0.30 < iou < 0.35


def test_nms_boxes():
    boxes = [
        [10, 10, 100, 100],
        [12, 12, 102, 102],  # Duplicate overlap
        [200, 200, 300, 300]  # Separate
    ]
    scores = [0.9, 0.8, 0.95]
    keep_indices = nms_boxes(boxes, scores, iou_threshold=0.5)
    assert 0 in keep_indices
    assert 2 in keep_indices
    assert 1 not in keep_indices
