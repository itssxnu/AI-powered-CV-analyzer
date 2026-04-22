import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Interviewer")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
# Configurable model, defaulting to the requested NVIDIA Nemotron 3 Nano (Free)
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free")

def scrub_cv(cv_json: dict) -> dict:
    """Removes sensitive Personally Identifiable Information (PII) before sending to external LLM APIs."""
    scrubbed = json.loads(json.dumps(cv_json))  # Deep copy
    
    # Remove Contact Block completely
    if "contact" in scrubbed:
        # Keep location maybe for context, but destroy name, email, phone, linkedin
        scrubbed["contact"] = {"location": scrubbed["contact"].get("location", "Unknown")}
        
    # Remove candidate name from the top level if it leaked anywhere
    # Just in case, erase any top-level identifiable keys if they exist
    keys_to_remove = ["name", "email", "phone", "linkedin", "address"]
    for k in keys_to_remove:
        if k in scrubbed:
            del scrubbed[k]
            
    # Scrub names of companies if strict anonymity required? 
    # Usually company names are fine. We will just remove standard PII.
    return scrubbed

FALLBACK_QUESTIONS = [
    {
        "category": "Role Fit", 
        "focus": "General Experience", 
        "question": "Can you walk me through your most recent role and how it prepared you for this position?", 
        "ideal_answer": "Candidate provides a clear summary of recent relevant experience."
    },
    {
        "category": "Matched Skill", 
        "focus": "Core Competency", 
        "question": "What do you consider your strongest technical skill and how have you applied it recently?", 
        "ideal_answer": "Candidate gives a specific example demonstrating depth of knowledge."
    },
    {
        "category": "Missing Skill", 
        "focus": "Adaptability", 
        "question": "Tell me about a time you had to learn a completely new tool or skill on the job.", 
        "ideal_answer": "Candidate demonstrates quick learning and proactive mindset."
    }
]

def validate_and_clean_questions(questions: list) -> list:
    """Validates structure, deduplicates, and ensures exactly 3 questions are returned."""
    validated = []
    seen_texts = set()
    required_keys = {"category", "focus", "question", "ideal_answer"}
    
    if isinstance(questions, list):
        for q in questions:
            if not isinstance(q, dict):
                continue
                
            # 1. Validate Structure
            if not required_keys.issubset(q.keys()):
                continue
                
            # 2. Check for Duplicate Questions
            q_text = str(q.get("question", "")).strip().lower()
            if not q_text or q_text in seen_texts:
                continue
                
            seen_texts.add(q_text)
            validated.append(q)
            
            # 3. Validate Count (Truncate if > 3)
            if len(validated) == 3:
                break
                
    # 4. Validate Count & Pad with fallbacks if < 3
    for fq in FALLBACK_QUESTIONS:
        if len(validated) >= 3:
            break
        fq_text = fq["question"].lower()
        if fq_text not in seen_texts:
            validated.append(fq)
            seen_texts.add(fq_text)
            
    return validated[:3]

def generate_interview_questions(cv_json: dict, jd_data: dict, match_details: dict) -> list[dict]:
    """Generates personalized interview questions via OpenRouter."""
    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY is not set. Cannot generate questions.")
        return [{"error": "OPENROUTER_API_KEY environment variable is missing."}]

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    
    # 1. Prepare Safe Payload
    safe_cv = scrub_cv(cv_json)
    
    # 2. Extract Data Context
    job_title = jd_data.get("title", "the role")
    department = jd_data.get("department", "the department")
    
    # Details from the Match
    matched_req = [m.get("jd_skill", "") for m in match_details.get("matched_required", [])]
    missing_req = match_details.get("missing_required", [])
    matched_pref = [m.get("jd_skill", "") for m in match_details.get("matched_preferred", [])]
    
    # 3. Construct the Prompt
    prompt = f"""You are an expert HR Technical Recruiter interviewing a candidate for the position of '{job_title}' in the '{department}' department. 

Candidate's Anonymized CV:
{json.dumps(safe_cv, indent=2)}

Match Analysis against Job Requirements:
- Matched Required Skills: {', '.join(matched_req) if matched_req else 'None'}
- Missing Required Skills: {', '.join(missing_req) if missing_req else 'None'}
- Matched Preferred Skills: {', '.join(matched_pref) if matched_pref else 'None'}

Please generate exactly 3 highly personalized interview questions for this specific candidate:
1. One deep-dive Behavioral (STAR method) question focusing on one of their MATcHED required skills to verify their expertise.
2. One probing question focusing on one of their MISSING required skills to see if they possess adjacent/transferable experience or willingness to learn.
3. One general role-fit or situational question based on their past experience block.

IMPORTANT: You MUST return your response as a pure JSON array of objects. Do not include markdown formatting like ```json. Return exactly this format:
[
  {{
    "category": "Matched Skill",
    "focus": "Communication",
    "question": "Tell me about a time...",
    "ideal_answer": "Look for..."
  }},
  ...
]
"""
    
    logger.info(f"Connecting to OpenRouter using model: {OPENROUTER_MODEL}...")
    try:
        completion = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": "You are a highly structured HR Assistant API. You always output valid, parseable JSON arrays."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        # Strip potential markdown blocks if the LLM disobeys instructions
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        questions_array = json.loads(response_text)
        return validate_and_clean_questions(questions_array)
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response into JSON: {response_text}")
        return validate_and_clean_questions([])  # Gracefully fallback to the 3 guaranteed questions
    except Exception as e:
        logger.error(f"OpenRouter API call failed: {e}")
        return validate_and_clean_questions([])  # Gracefully fallback to the 3 guaranteed questions
