"""Production Flask Web Application & REST API Server for Historical Manuscript Layout Analysis."""

import os
import sys
import uuid
import json
import time
from typing import Dict, Any
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for

# Ensure backend root is on sys.path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.pipeline import ManuscriptLayoutPipeline
from src.utils.image_io import load_image, save_image
from services.job_manager import job_manager, JobStatus
from services.pipeline_service import pipeline_service

# Paths configuration
FRONTEND_TEMPLATES = os.path.abspath(os.path.join(BACKEND_DIR, "..", "frontend", "templates"))
FRONTEND_STATIC = os.path.abspath(os.path.join(BACKEND_DIR, "..", "frontend", "static"))
DATA_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "data", "test_images"))
RESULTS_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "results"))
UPLOADS_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "results", "uploads"))

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize Flask App
app = Flask(
    __name__,
    template_folder=FRONTEND_TEMPLATES,
    static_folder=FRONTEND_STATIC,
    static_url_path="/static"
)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB upload limit
app.config["RESULTS_DIR"] = RESULTS_DIR
app.config["DATA_DIR"] = DATA_DIR
app.config["UPLOADS_DIR"] = UPLOADS_DIR


@app.route("/")
def index():
    """Renders the single-page application dashboard."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Telemetry and healthcheck endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Historical Manuscript Layout Analysis Engine",
        "version": "1.0.0",
        "timestamp": time.time(),
        "device": pipeline_service.device,
        "classes": ["header", "footer", "main_text", "side_text", "filler"]
    })


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Lists available test manuscript samples with metadata."""
    samples = []
    if os.path.isdir(DATA_DIR):
        for f in sorted(os.listdir(DATA_DIR)):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                full_path = os.path.join(DATA_DIR, f)
                size_kb = round(os.path.getsize(full_path) / 1024.0, 1)
                is_palm_leaf = "pic" in f.lower() or "palm" in f.lower()
                samples.append({
                    "filename": f,
                    "size_kb": size_kb,
                    "preview_url": f"/api/images/samples/{f}",
                    "substrate_hint": "Palm-Leaf (Tala-patra)" if is_palm_leaf else "Handmade Paper (Kaghaz)"
                })
    return jsonify({"samples": samples})


@app.route("/api/images/<category>/<filename>", methods=["GET"])
def serve_image(category: str, filename: str):
    """Safely serves sample images, uploaded scans, and annotated outputs."""
    # Prevent directory traversal
    filename = os.path.basename(filename)
    if category == "samples":
        return send_from_directory(DATA_DIR, filename)
    elif category == "results":
        return send_from_directory(RESULTS_DIR, filename)
    elif category == "uploads":
        return send_from_directory(UPLOADS_DIR, filename)
    else:
        return jsonify({"error": "Invalid image category"}), 404


@app.route("/api/detect", methods=["POST"])
def detect_synchronous():
    """Synchronous layout detection endpoint. Accepts uploaded file or sample filename."""
    start_time = time.perf_counter()
    conf = float(request.form.get("conf", 0.50))
    sample_name = request.form.get("sample_name")

    image_path = None
    original_filename = None

    if "file" in request.files and request.files["file"].filename != "":
        file = request.files["file"]
        original_filename = file.filename
        safe_name = f"{uuid.uuid4().hex[:8]}_{os.path.basename(original_filename)}"
        image_path = os.path.join(UPLOADS_DIR, safe_name)
        file.save(image_path)
    elif sample_name:
        original_filename = os.path.basename(sample_name)
        image_path = os.path.join(DATA_DIR, original_filename)
        if not os.path.isfile(image_path):
            return jsonify({"error": f"Sample '{sample_name}' not found."}), 404
    else:
        return jsonify({"error": "No image file or sample name provided."}), 400

    try:
        base_name = os.path.splitext(original_filename)[0]
        out_img_name = f"{base_name}_annotated.png"
        out_json_name = f"{base_name}_predictions.json"
        out_img_path = os.path.join(RESULTS_DIR, out_img_name)
        out_json_path = os.path.join(RESULTS_DIR, out_json_name)

        # Execute layout pipeline
        pipeline_service.pipeline.classifier.confidence_threshold = conf
        results = pipeline_service.process_sync(
            image_path=image_path,
            save_annotated_path=out_img_path,
            save_json_path=out_json_path
        )

        results["annotated_image_url"] = f"/api/images/results/{out_img_name}"
        results["json_url"] = f"/api/results/{out_json_name}"
        if sample_name:
            results["original_image_url"] = f"/api/images/samples/{original_filename}"
        else:
            results["original_image_url"] = f"/api/images/uploads/{os.path.basename(image_path)}"

        return jsonify({
            "status": "success",
            "data": results
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def upload_async():
    """Asynchronous pipeline submission. Enqueues job and returns job_id."""
    conf = float(request.form.get("conf", 0.50))
    sample_name = request.form.get("sample_name")

    image_path = None
    original_filename = None

    if "file" in request.files and request.files["file"].filename != "":
        file = request.files["file"]
        original_filename = file.filename
        safe_name = f"{uuid.uuid4().hex[:8]}_{os.path.basename(original_filename)}"
        image_path = os.path.join(UPLOADS_DIR, safe_name)
        file.save(image_path)
    elif sample_name:
        original_filename = os.path.basename(sample_name)
        image_path = os.path.join(DATA_DIR, original_filename)
        if not os.path.isfile(image_path):
            return jsonify({"error": f"Sample '{sample_name}' not found."}), 404
    else:
        return jsonify({"error": "No image file or sample name provided."}), 400

    job_id = pipeline_service.submit_async_job(
        image_path=image_path,
        filename=original_filename,
        output_dir=RESULTS_DIR,
        conf=conf
    )

    return jsonify({
        "status": "enqueued",
        "job_id": job_id,
        "filename": original_filename,
        "polling_url": f"/api/status/{job_id}"
    })


@app.route("/api/status/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    """Polls status and stage of an asynchronous layout detection job."""
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    response = {
        "job_id": job["job_id"],
        "filename": job["filename"],
        "status": job["status"],
        "stage": job["stage"],
        "progress_pct": job["progress_pct"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"]
    }

    if job["status"] == JobStatus.COMPLETED.value:
        res = job["result"]
        base_name = os.path.splitext(job["filename"])[0]
        res["annotated_image_url"] = f"/api/images/results/{base_name}_annotated.png"
        res["json_url"] = f"/api/results/{base_name}_predictions.json"
        if os.path.isfile(os.path.join(DATA_DIR, job["filename"])):
            res["original_image_url"] = f"/api/images/samples/{job['filename']}"
        else:
            res["original_image_url"] = f"/api/images/uploads/{job['filename']}"
        response["result"] = res

    if job["status"] == JobStatus.FAILED.value:
        response["error"] = job.get("error")

    return jsonify(response)


@app.route("/api/results/<filename>", methods=["GET"])
def get_prediction_json(filename: str):
    """Returns saved prediction JSON file."""
    safe_name = os.path.basename(filename)
    json_path = os.path.join(RESULTS_DIR, safe_name)
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Prediction JSON not found"}), 404


def run_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """Launches the Flask application."""
    print("=" * 70)
    print("  MANUSCRIPT SPECIFIC LAYOUT REGION DETECTION - WEB SERVER")
    print(f"  Serving at: http://127.0.0.1:{port}")
    print("=" * 70)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    run_server(port=port)
