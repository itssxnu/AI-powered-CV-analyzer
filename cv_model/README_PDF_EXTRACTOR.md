# 📄 PDF CV Extractor (Module 1)

This module handles the complex transformation of unstructured PDF Resumes/CVs into a strict, machine-readable JSON format optimized for the Hospitality Sector.

## 🚀 Features

* **Smart Extraction Strategy:** Uses `PyPDF2` by default for true-text PDFs. If the file is scanned or an image-level PDF, it intelligently falls back to `Tesseract OCR` and `pdf2image`.
* **LLM Data Structuring:** Leverages the **Google Gemini 2.5 Flash** API via the `google-genai` SDK to meticulously parse the raw text.
* **Hospitality Focused Schema:** Uses strict `Pydantic` schemas. The LLM is heavily prompted to look for Hospitality nuances such as *Night Audit Experience*, *F&B Outlets*, *Guest relations*, etc.
* **JSON Output:** Fully normalizes unstructured resumes into a clean `cv2.json` format containing `meta`, `contact`, `profile`, `hospitality`, `skills`, `experience`, and `education` objects.

## ⚙️ Prerequisites

1. **Poppler** (Installed and in system PATH, e.g., `C:\poppler\poppler-24.08.0\Library\bin`)
2. **Tesseract-OCR** (Installed and in system PATH, e.g., `C:\Program Files\Tesseract-OCR\tesseract.exe`)
3. A Google Gemini API Key configured in a local `.env` file:
   ```ini
   GEMINI_API_KEY="AIzaSy..."
   ```

## 💻 Usage

To process a single CV or an entire directory of PDFs into JSON:

```bash
python pdf_extractor.py --in "data/raw/cv1.pdf" --outdir "data/parsed"
```

The parsed JSON output will be saved into the `--outdir` directory, ready to be digested by the **Skill Matching Engine (Module 2)**.
