"""Unit tests for Microsoft Florence-2 VLM Layout Detector."""

import os
import sys
import pytest
import numpy as np
import cv2

# Ensure backend root is on sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(TEST_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.detection.layout_detector import LayoutDetector


def test_florence_detector_initialization():
    detector = LayoutDetector(confidence_threshold=0.40, device="cpu")
    assert detector.model is not None
    assert detector.processor is not None
    assert detector.confidence_threshold == 0.40


def test_florence_detector_inference_on_synthetic_image():
    detector = LayoutDetector(confidence_threshold=0.30, device="cpu")
    # Synthetic manuscript image (400x800)
    image = np.ones((400, 800, 3), dtype=np.uint8) * 180
    cv2.rectangle(image, (50, 40), (750, 360), (220, 240, 245), -1)
    cv2.putText(image, "OM NAMO NARAYANAYA", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(image, "Sample Main Text Body Verse", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(image, "Colophon Line", (100, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)

    preprocessed = {
        "substrate_type": "paper",
        "page_bbox": [50, 40, 750, 360],
        "enhanced_image": image
    }

    candidates = detector.detect(image, preprocessed)
    assert isinstance(candidates, list)
    assert len(candidates) > 0

    valid_classes = {"header", "footer", "main_text", "side_text", "filler"}
    for cand in candidates:
        assert cand["layout_type"] in valid_classes
        assert len(cand["bbox"]) == 4
        x1, y1, x2, y2 = cand["bbox"]
        assert 50 <= x1 <= 750
        assert 40 <= y1 <= 360
        assert 50 <= x2 <= 750
        assert 40 <= y2 <= 360
