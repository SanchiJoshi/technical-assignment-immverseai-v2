"""End-to-End Pipeline Orchestrator for Historical Manuscript Layout Region Detection."""

import time
import os
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from src.utils.image_io import load_image, save_image
from src.utils.serializer import LayoutSerializer
from src.preprocessing.folio_extractor import FolioExtractor
from src.detection.layout_detector import LayoutDetector
from src.postprocessing.region_classifier import RegionClassifier
from src.visualization.annotator import LayoutVisualizer


class ManuscriptLayoutPipeline:
    """End-to-End inference engine for historical manuscript layout analysis powered by Microsoft Florence-2 VLM."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        confidence_threshold: float = 0.40,
        device: str = "cpu"
    ):
        """Initialize the pipeline with all sub-modules.

        Args:
            weights_path: Optional path or model ID to Florence-2 weights.
            confidence_threshold: Confidence cutoff for region predictions.
            device: 'cuda', 'cpu', or 'mps'.
        """
        self.confidence_threshold = confidence_threshold
        self.device = device

        # Initialize modular pipeline components
        self.preprocessor = FolioExtractor()
        self.detector = LayoutDetector(
            weights_path=weights_path,
            confidence_threshold=confidence_threshold,
            device=device
        )
        self.classifier = RegionClassifier(confidence_threshold=confidence_threshold)
        self.visualizer = LayoutVisualizer()
        self.serializer = LayoutSerializer()

    def process_image(
        self, 
        image_path: str, 
        save_annotated_path: Optional[str] = None,
        save_json_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs the complete inference workflow on a single manuscript image.

        Args:
            image_path: Path to the input image.
            save_annotated_path: Optional output path for the annotated image.
            save_json_path: Optional output path for the JSON metadata.

        Returns:
            Dictionary containing structured JSON output data.
        """
        start_time = time.perf_counter()

        # 1. Load image safely
        bgr_image, rgb_image = load_image(image_path)
        img_name = os.path.basename(image_path)
        h, w = bgr_image.shape[:2]

        # 2. Stage 1: Preprocessing & Substrate Isolation
        preprocessed = self.preprocessor.process(bgr_image)
        substrate_type = preprocessed["substrate_type"]
        page_bbox = preprocessed.get("page_bbox", [0, 0, w, h])

        # 3. Stage 2: YOLO Candidate Region Detection
        candidates = self.detector.detect(bgr_image, preprocessed)

        # 4. Stage 3: 5-Class Classification & Spatial Constraint Snapping
        final_regions = self.classifier.process_and_classify(
            bgr_image, candidates, preprocessed
        )

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        # 5. Stage 4: JSON Formatting & Metadata Serialization
        results_json = self.serializer.format_output(
            image_name=img_name,
            image_shape=bgr_image.shape,
            regions=final_regions,
            substrate_type=substrate_type,
            page_bbox=page_bbox,
            processing_time_ms=total_latency_ms,
            extra_metadata={"device": self.device, "model": "YOLOv8-Manuscript"}
        )

        # 6. Save JSON if path provided
        if save_json_path:
            self.serializer.save_json(results_json, save_json_path)

        # 7. Generate & Save Annotated Visualization if path provided
        if save_annotated_path:
            annotated_bgr = self.visualizer.annotate(
                bgr_image, final_regions, include_legend=True
            )
            save_image(annotated_bgr, save_annotated_path)

        return results_json

    def process_batch(
        self,
        input_paths: List[str],
        output_dir: str
    ) -> List[Dict[str, Any]]:
        """Processes a list of image paths in batch mode.

        Args:
            input_paths: List of file paths to process.
            output_dir: Destination folder for results.

        Returns:
            List of JSON metadata dictionaries.
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []

        for p in input_paths:
            base_name = os.path.splitext(os.path.basename(p))[0]
            out_img = os.path.join(output_dir, f"{base_name}_annotated.png")
            out_json = os.path.join(output_dir, f"{base_name}_predictions.json")

            res = self.process_image(
                image_path=p,
                save_annotated_path=out_img,
                save_json_path=out_json
            )
            results.append(res)

        return results
