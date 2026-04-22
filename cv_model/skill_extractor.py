import re

import json
import os

SYNONYMS_FILE = os.path.join(os.path.dirname(__file__), "hotel_synth_dataset", "synonyms.json")
CANONICAL_SKILLS = {}
try:
    with open(SYNONYMS_FILE, "r", encoding="utf-8") as f:
        CANONICAL_SKILLS = json.load(f)
except Exception as e:
    print(f"Warning: Could not load {SYNONYMS_FILE}: {e}")
    # Fallback minimal subset
    CANONICAL_SKILLS = {
        "communication": ["communication", "verbal communication", "written communication"],
        "leadership": ["leadership", "management", "mentoring", "team lead"],
        "teamwork": ["teamwork", "team work", "collaboration"],
        "customer service": ["customer service", "guest service", "service excellence"]
    }

# Reverse mapping for fast synonym lookup
SYNONYM_MAP = {}
for canonical, synonyms in CANONICAL_SKILLS.items():
    # Keep the canonical itself in the map too
    SYNONYM_MAP[canonical.lower()] = canonical
    for syn in synonyms:
        SYNONYM_MAP[syn.lower()] = canonical

def _extract_from_text(text: str, evidence_label: str) -> list[dict]:
    """Helper to find skills in raw text by matching canonical list + synonyms."""
    results = []
    text_lower = text.lower()
    
    # Very basic tokenization and phrase matching
    # In production, use spaCy or simple Aho-Corasick automaton for faster sub-string matching
    found_phrases = set()

    for surface_phrase, canonical in SYNONYM_MAP.items():
        # Match word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(surface_phrase) + r'\b'
        matches = list(re.finditer(pattern, text_lower))
        
        if matches:
            for match in matches:
                span_start = max(0, match.start() - 30)
                span_end = min(len(text_lower), match.end() + 30)
                snippet = text[span_start:span_end].replace('\n', ' ').strip()
                
                # Check if we already found this exact surface phrase to avoid dupes in this text block
                if surface_phrase not in found_phrases:
                    results.append({
                        "canonical_skills": canonical,
                        "surface_skills": text[match.start():match.end()], # Use original casing
                        "evidence": f"[{evidence_label}] snippet: '...{snippet}...'"
                    })
                    found_phrases.add(surface_phrase)
    
    return results

def clean_text(text: str) -> str:
    """Optionally clean filler words from text to improve accuracy."""
    fillers = r'\b(a|an|the|and|or|of|in|to|with)\b'
    cleaned = re.sub(fillers, ' ', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', cleaned).strip()

def extract_skills(cv_data: dict | str | list) -> list[dict]:
    """
    Extracts structured skill representations from CV Data or JD Data.
    cv_data can be a parsed JSON dictionary, a raw string text, or a list of sentences.

    Returns list of dicts:
    {
        "canonical_skills": "normalized string",
        "surface_skills": "original phrase found",
        "evidence": "location/snippet"
    }
    """
    extracted_skills = []
    
    if isinstance(cv_data, str):
        # Fallback to pure text extraction
        return _extract_from_text(clean_text(cv_data), "raw_text_scan")
        
    if isinstance(cv_data, list):
        # Handle JD lists
        full_text = " ".join([str(item) for item in cv_data])
        return _extract_from_text(clean_text(full_text), "list_scan")

    # If it's a dictionary (Parsed JSON)
    
    # 1. Check for a dedicated "skills" section (prefer structured)
    skills_section = cv_data.get("skills") or cv_data.get("Skills") or cv_data.get("SKILLS")
    if skills_section:
        if isinstance(skills_section, list):
            for skill_item in skills_section:
                if isinstance(skill_item, str):
                    canonical = SYNONYM_MAP.get(skill_item.lower(), skill_item.lower()) # Fallback to lowercase item
                    extracted_skills.append({
                        "canonical_skills": canonical,
                        "surface_skills": skill_item,
                        "evidence": "Found in structured 'skills' list"
                    })
                elif isinstance(skill_item, dict) and "name" in skill_item:
                    skill_name = skill_item["name"]
                    canonical = SYNONYM_MAP.get(skill_name.lower(), skill_name.lower())
                    extracted_skills.append({
                        "canonical_skills": canonical,
                        "surface_skills": skill_name,
                        "evidence": "Found in structured 'skills' object list"
                    })
        elif isinstance(skills_section, str):
             extracted_skills.extend(_extract_from_text(clean_text(skills_section), "skills_section"))

    # 2. If no structured skills found, fallback to scanning summary / experience / full text
    if not extracted_skills:
        # Construct a synthetic full text block from json values
        def _flatten_values(d):
            vals = []
            if isinstance(d, dict):
                for v in d.values():
                    vals.extend(_flatten_values(v))
            elif isinstance(d, list):
                for v in d:
                    vals.extend(_flatten_values(v))
            elif isinstance(d, str):
                vals.append(d)
            return vals
        
        full_text = " ".join(_flatten_values(cv_data))
        extracted_skills = _extract_from_text(clean_text(full_text), "fallback_json_scan")

    # Deduplicate canonical skills, keeping the best evidence (first found)
    seen_canonicals = set()
    final_skills = []
    for s in extracted_skills:
        canon = s["canonical_skills"]
        if canon not in seen_canonicals:
            final_skills.append(s)
            seen_canonicals.add(canon)
            
    return final_skills
