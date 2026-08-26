# Historical Manuscript Layout Analysis Engine

A web application and API for detecting and classifying layout regions (headers, footers, main text, side annotations, and filler artifacts) in historical manuscript scans using OpenRouter's Qwen 2.5 VL 72B Instruct Vision-Language Model.

---

## How to Run Locally

### 1. Prerequisites
- Python 3.10 or higher installed

### 2. Setup Virtual Environment
Navigate to the project root directory, create a virtual environment, and activate it:

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Set API Key (Optional)
Set your OpenRouter API key in environment variables (if not using default):

**Windows (PowerShell):**
```powershell
$env:OPENROUTER_API_KEY="your_api_key_here"
```

**Linux / macOS:**
```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

### 5. Run the Application

#### Web Dashboard
To start the Flask web server:
```bash
python backend/app.py
```
Open your browser and visit: `http://127.0.0.1:5000`

#### CLI Script
To run inference on a manuscript image via command line:
```bash
python backend/inference.py --image backend/data/test_images/manuscript_sample1.jpg --output backend/results
```
