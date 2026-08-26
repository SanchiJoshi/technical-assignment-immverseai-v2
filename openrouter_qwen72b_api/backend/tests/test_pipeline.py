"""Unit tests for the end-to-end manuscript layout detection pipeline."""

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

from src.pipeline import ManuscriptLayoutPipeline


@pytest.fixture
def sample_image(tmp_path):
    """Generates a synthetic palm-leaf style manuscript test image."""
    img = np.ones((200, 500, 3), dtype=np.uint8) * 80  # slate background
    # Add leaf center
    img[20:180, 40:460] = (160, 200, 210)  # light substrate
    # Add simulated text lines
    for y in range(40, 160, 20):
        cv2.line(img, (60, y), (440, y), (30, 30, 30), 2)

    test_path = str(tmp_path / "test_synthetic.png")
    cv2.imwrite(test_path, img)
    return test_path


def test_pipeline_execution(sample_image, tmp_path):
    pipeline = ManuscriptLayoutPipeline()
    out_img = str(tmp_path / "out_annotated.png")
    out_json = str(tmp_path / "out_preds.json")

    res = pipeline.process_image(
        image_path=sample_image,
        save_annotated_path=out_img,
        save_json_path=out_json
    )

    assert "version" in res
    assert "image_metadata" in res
    assert "summary" in res
    assert "regions" in res
    assert len(res["regions"]) > 0

    # Validate regions and 5-class constraint
    valid_classes = {"header", "footer", "main_text", "side_text", "filler"}
    for reg in res["regions"]:
        assert reg["class"] in valid_classes
        assert len(reg["bbox"]) == 4
        x1, y1, x2, y2 = reg["bbox"]
        assert 0 <= x1 <= 500
        assert 0 <= y1 <= 200
        assert 0 <= x2 <= 500
        assert 0 <= y2 <= 200
        assert x1 <= x2
        assert y1 <= y2

    assert os.path.exists(out_img)
    assert os.path.exists(out_json)
