# Historical Manuscript Specific Layout Region Detection Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/PyTorch-YOLOv8-orange.svg)](https://ultralytics.com/)
[![Web Server](https://img.shields.io/badge/Flask-REST%20API-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A production-grade **Substrate-Aware Historical Manuscript Layout Region Detection Engine** developed for the **ImmverseAI Technical Assignment**. 

This system performs robust semantic object localization and spatial classification across diverse historical manuscript substrates—including **Palm-Leaf Folios (*Tala-patra*)** and **Handmade Paper (*Kaghaz*)**—combining deep learning backbones with spatial semantic post-processing and strict page boundary containment enforcement.

---

## Key Features

- **Substrate-Aware Pipeline**: Automatically distinguishes palm-leaf folios from paper manuscripts and adapts bounding box containment, margin spatial heuristics, and binding punch hole filtering.
- **YOLOv8 Deep Learning Backbone**: Leverages Ultralytics YOLO candidate detection trained on layout structures with fallback mechanisms.
- **5-Class Semantic Taxonomy**:
  - `header`: Top running headers, section titles, and chapter opening invocations.
  - `footer`: Bottom colophons, scribe signatures, and catchwords.
  - `main_text`: Central primary text blocks.
  - `side_text`: Lateral marginal annotations and marginal folio pagination.
  - `filler`: Central palm-leaf binding punch holes, digital archival watermarks, and collection stamps.
- **Strict Bounding Box Containment**: Ensures all localized layout regions are strictly clipped within the physical folio boundaries.
- **Interactive Single-Page Dashboard & REST API**: Features real-time sample selection, drag-and-drop scanning, asynchronous job queue polling, confidence threshold sliders, bounding box visual overlays, and JSON metadata downloads.
- **CLI Batch Processing**: Support for running directory-wide or single-file inference with timing telemetry and JSON summaries.
- **Extensive Test Coverage**: Unit and integration test suite (`pytest`) covering preprocessing, layout detection, service pipelines, and Flask REST endpoints.

---


## Quick Start Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/SanchiJoshi/technical-assignment-immverseai-v2.git
cd technical-assignment-immverseai-v2
```

---

### Step 2: Create & Activate Virtual Environment (`venv`)

#### On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### On Windows (Command Prompt / CMD):
```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

#### On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Dependencies

```bash
pip install -r backend/requirements.txt
```

---

### Step 4: Launch the Interactive Web Dashboard

Start the Flask web server:

```bash
python backend/app.py
```

Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## Command Line Interface (CLI) Usage

Process single images or entire directories of historical manuscript scans:

```bash
# Process all sample images in data folder
python backend/inference.py --input backend/data/test_images --output backend/results

# Process a specific image with custom confidence cutoff
python backend/inference.py --input backend/data/test_images/pic_1.png --output backend/results --conf 0.50
```

### CLI Arguments:
- `--input`, `-i`: Path to single manuscript image or directory of images. (Required)
- `--output`, `-o`: Directory to save output annotations and JSON results. (Default: `./results`)
- `--conf`, `-c`: Confidence threshold between `0.0` and `1.0`. (Default: `0.50`)
- `--device`, `-d`: Inference compute device (`cpu` or `cuda`). (Default: `cpu`)

---

## Running Automated Tests

Run the full automated test suite using `pytest`:

```bash
pytest backend/tests
```

---

## REST API Documentation

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/health` | `GET` | System telemetry, model version, compute device, and supported class list. |
| `/api/samples` | `GET` | List available manuscript sample scans with substrate hints and size metrics. |
| `/api/detect` | `POST` | Synchronous layout region detection. Accepts file upload or sample name. |
| `/api/upload` | `POST` | Asynchronous job submission. Returns `job_id` for background processing. |
| `/api/status/<job_id>` | `GET` | Poll job execution state, progress percentage, and final detection URLs. |
| `/api/results/<filename>` | `GET` | Retrieve JSON result file containing detected region bounding boxes & scores. |

---

## Target Class Taxonomy

| Class Label | Color Code | Spatial & Functional Description |
| :--- | :--- | :--- |
| **`header`** | Red (`#EF4444`) | Top running titles, chapter headers, or top opening invocations. |
| **`footer`** | Green (`#10B981`) | Bottom colophons, scribe signatures, catchwords, or folio footers. |
| **`main_text`** | Blue (`#3B82F6`) | Primary central manuscript body text block. |
| **`side_text`** | Purple (`#8B5CF6`) | Marginal commentaries, glosses, and lateral folio pagination. |
| **`filler`** | Amber (`#F59E0B`) | Archival stamps, digital watermarks, or central palm-leaf binding punch holes. |

---

## Submission Details

- **Candidate Name**: Sanchi Joshi
- **GitHub Repository**: [https://github.com/SanchiJoshi/technical-assignment-immverseai-v2.git](https://github.com/SanchiJoshi/technical-assignment-immverseai-v2.git)
- **Organization**: ImmverseAI
