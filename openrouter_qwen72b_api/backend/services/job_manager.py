"""Thread-safe Job Manager for tracking asynchronous pipeline execution."""

import time
import uuid
import threading
from enum import Enum
from typing import Dict, Any, Optional, List


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobManager:
    """In-memory thread-safe state store for asynchronous pipeline jobs."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self, filename: str, extra_meta: Optional[Dict[str, Any]] = None) -> str:
        """Initializes a new job entry.

        Args:
            filename: Original image filename.
            extra_meta: Optional dictionary of metadata.

        Returns:
            Unique job_id string.
        """
        job_id = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "filename": filename,
                "status": JobStatus.PENDING.value,
                "stage": "QUEUED",
                "progress_pct": 0,
                "created_at": now,
                "updated_at": now,
                "result": None,
                "error": None,
                "meta": extra_meta or {}
            }
        return job_id

    def update_stage(self, job_id: str, stage: str, progress_pct: int) -> None:
        """Updates the ongoing stage and progress percentage of a job.

        Args:
            job_id: ID of the job to update.
            stage: Name of the current pipeline stage.
            progress_pct: Progress percentage (0-100).
        """
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = JobStatus.RUNNING.value
                self._jobs[job_id]["stage"] = stage
                self._jobs[job_id]["progress_pct"] = min(100, max(0, progress_pct))
                self._jobs[job_id]["updated_at"] = time.time()

    def complete_job(self, job_id: str, result_data: Dict[str, Any]) -> None:
        """Marks a job as successfully completed with final prediction data.

        Args:
            job_id: Target job ID.
            result_data: Output prediction payload.
        """
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = JobStatus.COMPLETED.value
                self._jobs[job_id]["stage"] = "COMPLETED"
                self._jobs[job_id]["progress_pct"] = 100
                self._jobs[job_id]["result"] = result_data
                self._jobs[job_id]["updated_at"] = time.time()

    def fail_job(self, job_id: str, error_message: str) -> None:
        """Marks a job as failed with an error description.

        Args:
            job_id: Target job ID.
            error_message: Error string or traceback.
        """
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = JobStatus.FAILED.value
                self._jobs[job_id]["stage"] = "FAILED"
                self._jobs[job_id]["error"] = error_message
                self._jobs[job_id]["updated_at"] = time.time()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a job by its ID.

        Args:
            job_id: Job ID string.

        Returns:
            Job dictionary or None if not found.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def list_jobs(self) -> List[Dict[str, Any]]:
        """Returns all tracked jobs sorted by creation time descending."""
        with self._lock:
            return sorted(
                [dict(j) for j in self._jobs.values()],
                key=lambda x: x["created_at"],
                reverse=True
            )


# Global singleton instance
job_manager = JobManager()
