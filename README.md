# Historical Manuscript Layout Analysis

This repository contains computer vision and vision-language model (VLM) pipelines for detecting, classifying, and annotating layout regions in historical Indian manuscripts (Palm-Leaf Tala-patra folios and Handmade Paper Kaghaz manuscripts).

The workspace includes three implementations:
1. Florence-2 VLM Pipeline (`florence_2`)
2. YOLO DocLayout Pipeline (`yolo_doclayout`)
3. OpenRouter Qwen 2.5 VL Pipeline (`openrouter_qwen72b_api`)

---

## Target Class Taxonomy

All implementations segment and categorize layout regions into five target classes:
- header: Top running headers, section titles, and chapter opening invocations.
- footer: Bottom colophons, scribe signatures, and catchwords.
- main_text: Central primary text blocks.
- side_text: Lateral marginal annotations and marginal folio pagination.
- filler: Central palm-leaf binding punch holes, digital archival watermarks, and collection stamps.

---

## Project Implementations

### 1. Florence-2 Pipeline (`florence_2`)
A substrate-aware pipeline powered by Microsoft Florence-2 VLM for semantic localization, contrast enhancement (CLAHE), and layout region classification. Features a Flask web dashboard and CLI batch inference tool.

### 2. YOLO DocLayout Pipeline (`yolo_doclayout`)
A layout detection engine built using Ultralytics YOLOv8 for candidate region detection combined with spatial post-processing heuristics and strict page boundary containment. Includes a Flask web app and CLI script.

### 3. OpenRouter Qwen 2.5 VL Pipeline (`openrouter_qwen72b_api`)
A cloud VLM layout analysis pipeline utilizing the OpenRouter API and Qwen 2.5 VL 72B Instruct model for detailed visual layout analysis with a web dashboard and CLI execution.

---

## Getting Started

### Step 1: Clone the Repository

```bash
git clone https://github.com/SanchiJoshi/technical-assignment-immverseai-v2.git
cd technical-assignment-immverseai-v2
```

### Step 2: Set Up Virtual Environment

On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Running the Pipelines

### 1. Running Florence-2 (`florence_2`)

Install dependencies:
```bash
pip install -r florence_2/backend/requirements.txt
```

Run Web Dashboard:
```bash
python florence_2/backend/app.py
```
Open browser at http://127.0.0.1:5000

Run CLI Inference:
```bash
python florence_2/backend/inference.py --input florence_2/backend/data/test_images --output florence_2/backend/results
```

---

### 2. Running YOLO DocLayout (`yolo_doclayout`)

Install dependencies:
```bash
pip install -r yolo_doclayout/backend/requirements.txt
```

Run Web Dashboard:
```bash
python yolo_doclayout/backend/app.py
```
Open browser at http://127.0.0.1:5000

Run CLI Inference:
```bash
python yolo_doclayout/backend/inference.py --input yolo_doclayout/backend/data/test_images --output yolo_doclayout/backend/results
```

---

### 3. Running OpenRouter Qwen 2.5 VL (`openrouter_qwen72b_api`)

Install dependencies:
```bash
pip install -r openrouter_qwen72b_api/backend/requirements.txt
```

Set API Key (optional if already configured in environment):
- Windows (PowerShell):
  ```powershell
  $env:OPENROUTER_API_KEY="your_api_key_here"
  ```
- Linux / macOS:
  ```bash
  export OPENROUTER_API_KEY="your_api_key_here"
  ```

Run Web Dashboard:
```bash
python openrouter_qwen72b_api/backend/app.py
```
Open browser at http://127.0.0.1:5000

Run CLI Inference:
```bash
python openrouter_qwen72b_api/backend/inference.py --image openrouter_qwen72b_api/backend/data/test_images/manuscript_sample1.jpg --output openrouter_qwen72b_api/backend/results
```

---

## Running Automated Tests

To execute tests for the Florence-2 or YOLO DocLayout pipelines:

```bash
pytest florence_2/backend/tests
pytest yolo_doclayout/backend/tests
```
