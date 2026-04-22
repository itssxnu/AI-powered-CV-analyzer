from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from PyPDF2 import PdfReader
import pytesseract
from pdf2image import convert_from_path

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("WARNING: google-genai library is required for LLM extraction.")
    genai = None

# =========================
# CONFIG (edit these)
# =========================
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-24.08.0\Library\bin"
DEFAULT_OCR_LANG = "eng"

try:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
except Exception:
    pass

# =========================
# SCHEMA (PYDANTIC)
# =========================
class MetaSchema(BaseModel):
    source_file: Optional[str] = None
    parser: Optional[str] = None
    confidence: Optional[float] = None
    language_hint: Optional[str] = None
    page_count: Optional[int] = None

class ContactSchema(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None

class ProfileSchema(BaseModel):
    target_role: Optional[str] = None
    career_level: Optional[str] = Field(description="'entry', 'mid', or 'senior'")
    years_experience: Optional[int] = None
    career_summary: Optional[str] = None

class ShiftExperienceSchema(BaseModel):
    night_shift: Optional[bool] = None
    rotational_shifts: Optional[bool] = None
    weekend_holiday: Optional[bool] = None

class OperationalFlagsSchema(BaseModel):
    cash_handling: Optional[bool] = None
    banquet_experience: Optional[bool] = None
    complaint_handling: Optional[bool] = None
    night_audit_experience: Optional[bool] = None

class HospitalitySchema(BaseModel):
    brands_worked: Optional[List[str]] = Field(description="e.g. Cinnamon, Shangri-La, Hilton")
    hotel_types: Optional[List[str]] = None
    room_inventory_experience: Optional[int] = None
    departments_worked: Optional[List[str]] = Field(description="Front Office, Housekeeping, Culinary, F&B, HR, etc.")
    outlets_or_areas: Optional[List[str]] = None
    systems_tools: Optional[List[str]] = None
    pre_opening_experience: Optional[bool] = None
    shift_experience: Optional[ShiftExperienceSchema] = None
    operational_flags: Optional[OperationalFlagsSchema] = None
    compliance_certifications: Optional[List[str]] = Field(description="e.g. Food Safety, First Aid, HACCP")

class ExperienceSchema(BaseModel):
    company: Optional[str] = None
    role: Optional[str] = None
    duration: Optional[str] = None
    description: Optional[str] = None

class EducationSchema(BaseModel):
    institute: Optional[str] = None
    degree: Optional[str] = Field(description="Course Name, Degree, or GCE O/L or A/L")
    year: Optional[int] = None

class CertificationSchema(BaseModel):
    name: Optional[str] = None
    year: Optional[int] = None

class LanguageSchema(BaseModel):
    name: Optional[str] = None
    level: Optional[str] = None

class EligibilitySchema(BaseModel):
    nationality: Optional[str] = None
    passport_no: Optional[str] = None
    visa_status: Optional[str] = None
    visa_expiry: Optional[str] = None

class ReferenceSchema(BaseModel):
    available_on_request: Optional[bool] = None
    referees: Optional[List[str]] = None

class RawSchema(BaseModel):
    text: Optional[str] = None

class ResumeSchema(BaseModel):
    meta: Optional[MetaSchema] = None
    contact: Optional[ContactSchema] = None
    profile: Optional[ProfileSchema] = None
    hospitality: Optional[HospitalitySchema] = None
    skills: Optional[List[str]] = None
    experience: Optional[List[ExperienceSchema]] = None
    education: Optional[List[EducationSchema]] = None
    certifications: Optional[List[CertificationSchema]] = None
    languages: Optional[List[LanguageSchema]] = None
    work_eligibility: Optional[EligibilitySchema] = None
    references: Optional[ReferenceSchema] = None
    raw: Optional[RawSchema] = None

# =========================
# TEXT EXTRACTION (KEEP EXISTING LOGIC)
# =========================
def extract_text_pypdf(pdf_path: str) -> Tuple[str, int]:
    try:
        reader = PdfReader(pdf_path)
        parts = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(parts).strip(), len(reader.pages)
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return "", 0

def extract_text_ocr(pdf_path: str, dpi: int = 300, lang: str = DEFAULT_OCR_LANG) -> Tuple[str, int]:
    try:
        images = convert_from_path(pdf_path, dpi=dpi, poppler_path=POPPLER_PATH)
        parts = []
        for img in images:
            # Try to determine orientation and rotate if necessary
            try:
                osd = pytesseract.image_to_osd(img)
                rot_match = re.search(r'(?<=Rotate: )\d+', osd)
                if rot_match:
                    angle = int(rot_match.group(0))
                    if angle != 0:
                        img = img.rotate(-angle, expand=True)
            except Exception as e:
                pass
                
            parts.append(pytesseract.image_to_string(img, lang=lang))
        return "\n".join(parts).strip(), len(images)
    except Exception as e:
        print(f"OCR failed for {pdf_path}: {e}")
        return "", 0

def is_probably_scanned(text: str) -> bool:
    if not text: return True
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) < 200: return True
    alpha = sum(c.isalpha() for c in cleaned)
    if max(1, len(cleaned)) > 0 and (alpha / len(cleaned)) < 0.15: return True
    return False

def extract_text_smart(pdf_path: str, ocr_lang: str = DEFAULT_OCR_LANG) -> Tuple[str, str, float, int]:
    text, page_count = extract_text_pypdf(pdf_path)
    if is_probably_scanned(text):
        ocr_text, ocr_pages = extract_text_ocr(pdf_path, lang=ocr_lang)
        conf = min(1.0, max(0.2, len(ocr_text) / 6000))
        return ocr_text, "ocr", conf, ocr_pages
    conf = min(1.0, max(0.4, len(text) / 9000))
    return text, "pypdf", conf, page_count


# =========================
# LLM EXTRACTION
# =========================
SYSTEM_PROMPT = """You are an expert hospitality recruiter and resume data parser.
Your job is to read the raw text extracted from a parsed PDF Curriculum Vitae and meticulously map all the details to the exact JSON schema provided.
Do NOT hallucinate information. If a field is not present in the CV, leave it as null/None.

Specific Hospitality Guidelines:
- Departments: Try to map experience to core hospitality departments (Front Office, Housekeeping, Culinary, F&B, Human Resources, Engineering, Sales & Marketing, Finance).
- Shift Experience: Look for mentions of "Night Audit", "Rotational Shifts", "Night Shift", etc.
- Operational Flags: Note if they handle cash processing, large banquets, or explicit complaint handling.
- Compliance: Note if they have "Food Safety", "First Aid", "HACCP" or "Fire Safety" training based on their education and courses.
- Career Level: Categorize them strictly as 'entry', 'mid', or 'senior' based on total years of experience or previous roles. Unpaid internships are entry.
- Names & Contacts: Separate names, phones, and emails cleanly. Ignore Sri Lankan ID numbers (usually 12 digits or 9 digits followed by V). Phone numbers usually begin with 07 or +94.
- Education: Map "G.C.E. O/L" or "GCE Advanced Level" degrees clearly.
"""

def call_llm_parser(text: str, source_file: str, parser_used: str, conf: float, page_count: int) -> Dict[str, Any]:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("CRITICAL: GEMINI_API_KEY environment variable not set. LLM parsing will fail.")
        return {}
        
    client = genai.Client()
    
    # We use gemini-2.5-flash for very fast and cheap structured outputs
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ResumeSchema,
            temperature=0.1, # Extremely low temp for consistent extraction
        ),
    )
    
    # The response.text will be a JSON string that exactly matches ResumeSchema
    try:
        parsed_json = json.loads(response.text)
    except Exception as e:
        print(f"Failed to decode LLM JSON: {e}")
        return {}
        
    # Inject our extraction meta and raw fields
    if "meta" not in parsed_json or not parsed_json["meta"]:
        parsed_json["meta"] = {}
    parsed_json["meta"].update({
        "source_file": source_file,
        "parser": parser_used,
        "confidence": round(conf, 3),
        "page_count": page_count
    })
    
    if "raw" not in parsed_json or not parsed_json["raw"]:
        parsed_json["raw"] = {}
    parsed_json["raw"]["text"] = text
    
    return parsed_json


def parse_pdf(pdf_path: str, ocr_lang: str = DEFAULT_OCR_LANG) -> Dict[str, Any]:
    text, parser_used, conf, page_count = extract_text_smart(pdf_path, ocr_lang=ocr_lang)
    return call_llm_parser(text, Path(pdf_path).name, parser_used, conf, page_count)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--outdir", dest="outdir", default="data/parsed")
    args = ap.parse_args()

    inp = Path(args.inp)
    out_dir = Path(args.outdir)
    pdfs = sorted(inp.glob("*.pdf")) if inp.is_dir() else [inp]

    out_dir.mkdir(parents=True, exist_ok=True)
    
    for pdf in pdfs:
        print(f"Parsing: {pdf.name} (via LLM)...")
        res = parse_pdf(str(pdf))
        out_file = out_dir / f"{pdf.stem}.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(res, f, indent=4, ensure_ascii=False)
        print(f"  -> Saved to {out_file.resolve()}")

    print(f"\nSuccessfully parsed {len(pdfs)} PDF(s) into {out_dir.resolve()}")


if __name__ == "__main__":
    main()