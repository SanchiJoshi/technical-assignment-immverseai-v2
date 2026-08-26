"""Service layer wrapping ManuscriptLayoutPipeline with background thread execution."""

import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List

from src.pipeline import ManuscriptLayoutPipeline
from src.utils.image_io import load_image, save_image
from .job_manager import job_manager, JobManager


class PipelineService:
    """Production service for executing manuscript layout analysis synchronously and asynchronously."""

    def __init__(self, max_workers: int = 4, confidence_threshold: float = 0.50, device: str = "cpu"):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.pipeline = ManuscriptLayoutPipeline(
            confidence_threshold=confidence_threshold,
            device=device
        )
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ManuscriptWorker")

    def process_sync(
        self,
        image_path: str,
        save_annotated_path: Optional[str] = None,
        save_json_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes the pipeline synchronously on a single image.

        Args:
            image_path: Path to the manuscript image.
            save_annotated_path: Optional path to save visual overlay.
            save_json_path: Optional path to save JSON predictions.

        Returns:
            Dictionary containing prediction payload.
        """
        return self.pipeline.process_image(
            image_path=image_path,
            save_annotated_path=save_annotated_path,
            save_json_path=save_json_path
        )

    def submit_async_job(
        self,
        image_path: str,
        filename: str,
        output_dir: str,
        conf: Optional[float] = None
    ) -> str:
        """Enqueues an asynchronous manuscript processing job on a worker thread.

        Args:
            image_path: Local path to the uploaded image.
            filename: Original file name.
            output_dir: Destination directory for artifacts.
            conf: Optional confidence cutoff override.

        Returns:
            job_id string for status polling.
        """
        job_id = job_manager.create_job(filename=filename)
        self.executor.submit(
            self._execute_job_worker,
            job_id,
            image_path,
            filename,
            output_dir,
            conf
        )
        return job_id

    def _execute_job_worker(
        self,
        job_id: str,
        image_path: str,
        filename: str,
        output_dir: str,
        conf: Optional[float] = None
    ) -> None:
        """Worker function running in a background thread with granular stage updates."""
        try:
            start_time = time.perf_counter()
            base_name = os.path.splitext(filename)[0]
            os.makedirs(output_dir, exist_ok=True)
            out_img_path = os.path.join(output_dir, f"{base_name}_annotated.png")
            out_json_path = os.path.join(output_dir, f"{base_name}_predictions.json")

            # 1. Load Image
            job_manager.update_stage(job_id, stage="IMAGE_LOADING", progress_pct=10)
            bgr_image, _ = load_image(image_path)
            h, w = bgr_image.shape[:2]

            # 2. Stage 1: Preprocessing & Substrate Isolation
            job_manager.update_stage(job_id, stage="PREPROCESSING", progress_pct=30)
            preprocessed = self.pipeline.preprocessor.process(bgr_image)
            substrate_type = preprocessed["substrate_type"]
            page_bbox = preprocessed.get("page_bbox", preprocessed.get("folio_bbox", [0, 0, w, h]))

            # 3. Stage 2: Layout Region Detection
            job_manager.update_stage(job_id, stage="DETECTION", progress_pct=55)
            candidates = self.pipeline.detector.detect(bgr_image, preprocessed)

            # 4. Stage 3: Classification & Clamping
            job_manager.update_stage(job_id, stage="CLASSIFICATION", progress_pct=75)
            classifier = self.pipeline.classifier
            if conf is not None:
                classifier.confidence_threshold = conf
            final_regions = classifier.process_and_classify(bgr_image, candidates, preprocessed)

            # 5. Stage 4: Visualization & Serialization
            job_manager.update_stage(job_id, stage="SERIALIZATION", progress_pct=90)
            total_latency_ms = (time.perf_counter() - start_time) * 1000.0

            results_json = self.pipeline.serializer.format_output(
                image_name=filename,
                image_shape=bgr_image.shape,
                regions=final_regions,
                substrate_type=substrate_type,
                page_bbox=page_bbox,
                processing_time_ms=total_latency_ms,
                extra_metadata={"device": self.device, "job_id": job_id}
            )

            # Save artifacts
            self.pipeline.serializer.save_json(results_json, out_json_path)
            annotated_bgr = self.pipeline.visualizer.annotate(bgr_image, final_regions, include_legend=True)
            save_image(annotated_bgr, out_img_path)

            results_json["annotated_image_filename"] = f"{base_name}_annotated.png"
            results_json["json_filename"] = f"{base_name}_predictions.json"
            if "uploads" in image_path:
                results_json["original_image_url"] = f"/api/images/uploads/{os.path.basename(image_path)}"
            else:
                results_json["original_image_url"] = f"/api/images/samples/{filename}"

            # 6. Complete Job
            job_manager.complete_job(job_id, results_json)

        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            job_manager.fail_job(job_id, err_msg)


# Global singleton instance
pipeline_service = PipelineService()
