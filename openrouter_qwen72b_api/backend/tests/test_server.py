"""Integration tests for Flask API routes and Pipeline Service."""

import os
import sys
import json
import pytest

# Ensure backend root is on sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(TEST_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app import app
from services.job_manager import job_manager, JobStatus
from services.pipeline_service import pipeline_service


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "classes" in data
    assert len(data["classes"]) == 5


def test_get_samples(client):
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = res.get_json()
    assert "samples" in data
    assert len(data["samples"]) > 0


def test_synchronous_detection_on_sample(client):
    res = client.post("/api/detect", data={"sample_name": "pic_1.png", "conf": 0.50})
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "data" in data
    assert "regions" in data["data"]
    assert len(data["data"]["regions"]) > 0


def test_job_manager_lifecycle():
    job_id = job_manager.create_job("test_folio.png")
    assert job_id is not None
    job = job_manager.get_job(job_id)
    assert job["status"] == JobStatus.PENDING.value

    job_manager.update_stage(job_id, "PREPROCESSING", 30)
    job = job_manager.get_job(job_id)
    assert job["status"] == JobStatus.RUNNING.value
    assert job["stage"] == "PREPROCESSING"
    assert job["progress_pct"] == 30

    dummy_result = {"status": "ok", "regions": []}
    job_manager.complete_job(job_id, dummy_result)
    job = job_manager.get_job(job_id)
    assert job["status"] == JobStatus.COMPLETED.value
    assert job["progress_pct"] == 100
    assert job["result"] == dummy_result
