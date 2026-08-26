# Historical Manuscript Layout Region Detection Engine

A substrate-aware Computer Vision & Vision-Language Model (VLM) pipeline for historical Indian manuscript layout region detection and spatial classification. Powered by **Microsoft Florence-2 VLM** and **PyTorch**, this engine analyzes historical paper manuscripts (*Kaghaz*) and palm-leaf folios (*Tala-patra*) to automatically segment, classify, and annotate layout regions.

---

## 🌟 Overview

Historical manuscripts pose unique challenges for computer vision due to irregular substrates, aging, binding holes, marginal notes, and non-standard layouts. This project delivers an end-to-end, modular pipeline that:
1. **Isolates Physical Page Boundaries**: Separates the manuscript folio from scanning beds or dark digitizing backgrounds.
2. **Normalizes Contrast**: Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) on LAB color space to enhance faded ink.
3. **Detects & Categorizes Layout Regions**: Applies Microsoft Florence-2 VLM semantic localization mapped into a 5-class target taxonomy (`header`, `footer`, `main_text`, `side_text`, `filler`).
4. **Enforces Spatial Validity**: Restricts all detected region bounding boxes strictly inside physical folio boundaries.
5. **Provides Web & CLI Interfaces**: Includes a modern single-page web dashboard (Flask) and a command-line interface (CLI) for batch processing.

---

## 🏷️ Target Taxonomy (5 Classes)

| Class | Description |
| :--- | :--- |
| **`main_text`** | Primary central manuscript body text block. |
| **`header`** | Top running header, invocation line, or opening chapter title. |
| **`footer`** | Bottom colophon, scribe signature, or catchword. |
| **`side_text`** | Lateral margin annotations, commentary, or marginal pagination. |
| **`filler`** | Palm-leaf binding punch holes, stamps, seals, or archival watermarks. |

---

## 📁 Repository Structure

```
florence_2/
├── backend/
│   ├── app.py                  # Flask Web Server & REST API
│   ├── inference.py            # CLI Inference Script (single image or folder)
│   ├── requirements.txt        # Python package dependencies
│   ├── pytest.ini              # Pytest configuration
│   ├── data/
│   │   └── test_images/        # Sample historical manuscript images
│   ├── services/
│   │   ├── job_manager.py      # Async background job manager
│   │   └── pipeline_service.py # Core pipeline thread-safe wrapper
│   ├── src/
│   │   ├── pipeline.py         # End-to-End manuscript layout pipeline orchestrator
│   │   ├── detection/          # Florence-2 VLM layout detector module
│   │   ├── preprocessing/      # Page boundary extraction & contrast enhancement
│   │   ├── postprocessing/     # 5-class region classifier & boundary snapping
│   │   ├── visualization/      # Bounding box & polygon annotator
│   │   └── utils/              # Image I/O & JSON serialization helpers
│   └── tests/                  # Automated unit & integration tests
├── frontend/
│   ├── templates/
│   │   └── index.html          # Interactive Web Dashboard HTML
│   └── static/                 # Stylesheet & frontend JavaScript
└── .gitignore                  # Git ignore rules
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+** installed on your system
- **Git** installed
- Optional: CUDA-capable GPU for faster VLM inference (CPU execution is fully supported)

---

### Step 1: Clone the Repository

Clone this repository from GitHub to your local machine:

```bash
git clone https://github.com/YOUR_USERNAME/florence_2.git
cd florence_2
```

---

### Step 2: Create and Activate a Virtual Environment

It is recommended to set up an isolated Python virtual environment:

- **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```

- **On Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

---

### Step 3: Install Dependencies

Install all required Python packages using `pip`:

```bash
pip install -r backend/requirements.txt
```

*Note: Microsoft Florence-2 model weights will be automatically downloaded from Hugging Face (`microsoft/Florence-2-base`) upon the first run.*

---

## 💻 Running the Application

### Option A: Interactive Web Dashboard (Flask Web Server)

Launch the Flask web server to use the graphical web interface:

```bash
python backend/app.py
```

Once started, open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

**Web App Features:**
- Upload your own manuscript scans (.png, .jpg, .webp) or choose pre-loaded test folios.
- View live asynchronous processing stage progress.
- Toggle between original scans and annotated layout overlays.
- Inspect bounding box coordinates, confidence scores, area measurements, and class breakdowns.
- Download annotated output images and JSON prediction files.

---

### Option B: CLI Batch Inference Script

Process a single image or an entire folder of manuscript scans via command-line interface:

#### 1. Batch process a directory of manuscript scans:
```bash
python backend/inference.py --input backend/data/test_images --output backend/results
```

#### 2. Process a single manuscript image with custom confidence threshold:
```bash
python backend/inference.py --input backend/data/test_images/pic_1.png --output backend/results --conf 0.50
```

#### CLI Parameters:
- `--input`, `-i`: **(Required)** Path to input image file or folder of images.
- `--output`, `-o`: Destination directory to save outputs (Default: `./results`).
- `--conf`, `-c`: Confidence cutoff threshold between `0.0` and `1.0` (Default: `0.50`).
- `--device`, `-d`: Inference compute device (`cpu` or `cuda`, Default: `cpu`).

---

### Option C: Automated Testing

Run the test suite using `pytest` to verify pipeline components:

```bash
pytest backend/tests
```

---

## 📊 Output Data Format

The pipeline outputs structured JSON metadata alongside annotated visual overlays:

```json
{
  "version": "1.0.0",
  "task": "manuscript_layout_region_detection",
  "image_metadata": {
    "file_name": "pic_1.png",
    "width": 1280,
    "height": 720,
    "substrate_type": "palm_leaf",
    "page_boundary": [32, 18, 1248, 702]
  },
  "summary": {
    "total_regions_detected": 4,
    "class_distribution": {
      "header": 1,
      "footer": 1,
      "main_text": 1,
      "side_text": 0,
      "filler": 1
    }
  },
  "regions": [
    {
      "id": 1,
      "class": "main_text",
      "confidence": 0.96,
      "bbox": [120, 150, 1160, 580],
      "polygon": [[120, 150], [1160, 150], [1160, 580], [120, 580]],
      "area_px": 447200,
      "description": "Primary manuscript central text block"
    }
  ]
}
```

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
