from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional, Any
from sqlalchemy.orm import Session
import json
import os
import tempfile

from database import get_db, JobDescriptionRef, MatchResult, CVParsed, init_db
from matcher import evaluate_match

app = FastAPI(title="Skill Matching Engine API")

# Auto-create any missing tables (cv_parsed, match_results, etc.) on startup
init_db()

class MatchRequest(BaseModel):
    cv_json: Optional[dict[str, Any]] = None
    cv_text: Optional[str] = None
    job_id: Optional[int] = Field(None, description="Optional job_id to match against.")
    job_data: Optional[dict[str, Any]] = Field(None, description="Optional raw JD data to match against without checking DB.")
    top_k: int = Field(5, description="Number of top matches to return")
    source_file: Optional[str] = Field("api_request", description="Source identifier to store in DB")

@app.post("/match")
def match_cv(request: MatchRequest, db: Session = Depends(get_db)):
    if not request.cv_json and not request.cv_text:
        raise HTTPException(status_code=400, detail="Must provide either cv_json or cv_text")
        
    cv_data_input = request.cv_json
    if not cv_data_input and request.cv_text:
        # Try to parse cv_text as JSON if it looks like it
        if request.cv_text.strip().startswith("{"):
            try:
                cv_data_input = json.loads(request.cv_text)
            except:
                cv_data_input = request.cv_text
        else:
            cv_data_input = request.cv_text
    
    # 1. Save parsed CV to DB
    candidate_name = None
    candidate_email = None
    
    if isinstance(cv_data_input, dict) and "contact" in cv_data_input:
        contact = cv_data_input["contact"]
        if isinstance(contact, dict):
            candidate_name = contact.get("name")
            candidate_email = contact.get("email")

    from classifier import classify_candidate
    
    # Use cv_data_input (already parsed dict if possible) for classification
    cv_json_payload = cv_data_input if isinstance(cv_data_input, dict) else {"raw_text": cv_data_input}
    classification_result = classify_candidate(cv_json_payload)
    career_level = classification_result.get("level", "Unknown")

    
    cv_record = CVParsed(
        source_file=request.source_file,
        candidate_name=candidate_name,
        candidate_email=candidate_email,
        career_level=career_level,
        parsed_json=cv_json_payload
    )
    db.add(cv_record)
    db.flush() # get cv_record.id
    
    # 2. Pre-compute CV Skills and Embeddings
    from skill_extractor import extract_skills
    from matcher import get_embeddings_batch
    
    cv_skills_extracted = extract_skills(cv_data_input)
    cv_canonical_skills = [skill["canonical_skills"] for skill in cv_skills_extracted]
    cv_embeddings = get_embeddings_batch(cv_canonical_skills, db)

    # 3. Fetch Job Descriptions
    jds = []
    if request.job_data:
        # Create a mock JD from the provided data
        class MockJD:
            def __init__(self, data):
                self.id = data.get("id", request.job_id or 0)
                self.title = data.get("title", "")
                self.department = data.get("department", "")
                self.min_years = data.get("min_years", 0)
                
                # Handle skills which might be passed as strings instead of JSON strings
                import json
                
                req_skills = data.get("required_skills", "[]")
                if isinstance(req_skills, list):
                    # Trim whitespace and filter empty strings from each skill
                    req_skills = [s.strip() for s in req_skills if str(s).strip()]
                    self.required_skills_json = json.dumps(req_skills)
                else:
                    self.required_skills_json = req_skills
                    
                pref_skills = data.get("preferred_skills", "[]")
                if isinstance(pref_skills, list):
                    pref_skills = [s.strip() for s in pref_skills if str(s).strip()]
                    self.preferred_skills_json = json.dumps(pref_skills)
                else:
                    self.preferred_skills_json = pref_skills

                    
        jds = [MockJD(request.job_data)]
    elif request.job_id:
        jds = db.query(JobDescriptionRef).filter(JobDescriptionRef.id == request.job_id).all()
        if not jds:
            raise HTTPException(status_code=404, detail=f"JobDescription with id {request.job_id} not found")
    else:
        jds = db.query(JobDescriptionRef).all()
        
    if not jds:
        return {"matches": []}
        
    # 4. Match against each JD
    matches = []
    for jd_model in jds:
        # Construct JD data matching what evaluate_match expects
        jd_data = {
            "title": jd_model.title,
            "department": jd_model.department,
            "min_years": jd_model.min_years,
            "required_skills_json": jd_model.required_skills_json,
            "preferred_skills_json": jd_model.preferred_skills_json
        }
        
        result_dict = evaluate_match(
            cv_data_input, 
            jd_data, 
            db,
            precomputed_cv_skills=cv_skills_extracted,
            precomputed_cv_embeddings=cv_embeddings
        )
        
        # Save or update MatchResult
        match_record = db.query(MatchResult).filter(
            MatchResult.cv_id == cv_record.id,
            MatchResult.job_id == jd_model.id
        ).first()
        
        if not match_record:
            match_record = MatchResult(
                cv_id=cv_record.id,
                job_id=jd_model.id
            )
            db.add(match_record)
            
        match_record.score = result_dict["score"]
        match_record.label = result_dict["label"]
        match_record.details_json = result_dict["details"]
        
        matches.append({
            "job_id": jd_model.id,
            "title": jd_model.title,
            "department": jd_model.department,
            "score": result_dict["score"],
            "label": result_dict["label"].value,
            "details": result_dict["details"]
        })
        
    db.commit()
    
    # Sort by score desc, take top_k
    matches.sort(key=lambda x: x["score"], reverse=True)
    top_matches = matches[:request.top_k]
    
    return {
        "cv_id": cv_record.id,
        "career_level": career_level,
        "career_yoe": classification_result.get("yoe", 0),
        "career_explanation": classification_result.get("explanation", ""),
        "matches": top_matches
    }

@app.post("/analyze-resume")
async def analyze_resume(resume: UploadFile = File(...)):
    """
    Accepts a PDF resume upload, parses it using the LLM extractor, 
    and returns the highly structured JSON data.
    """
    from pdf_extractor import parse_pdf

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await resume.read()
            tmp.write(content)
            tmp_path = tmp.name
            
        # Parse PDF using the existing module
        parsed_data = parse_pdf(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        return parsed_data
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Failed to parse resume: {str(e)}")

@app.post("/fairness-audit")
def fairness_audit(db: Session = Depends(get_db)):
    """
    Runs a real-time Fairness and Bias Audit across 12 protected characteristics
    by generating synthetic CV clones and evaluating them against the Ranker Model.
    Returns Demographic Parity and Disparate Impact statistics.
    """
    from bias_detector import audit_cv_fairness
    
    try:
        report = audit_cv_fairness(db)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fairness audit failed: {str(e)}")

class QuestionsRequest(BaseModel):
    cv_json: Optional[dict[str, Any]] = None
    cv_text: Optional[str] = None
    job_data: dict[str, Any]
    match_details: Optional[dict[str, Any]] = Field(default_factory=dict)

@app.post("/generate-questions")
def generate_questions(request: QuestionsRequest):
    """
    Generates customized interview questions by passing the candidate's CV
    and skill match details through OpenRouter's LLM API.
    """
    from interviewer import generate_interview_questions
    
    cv_json_payload = request.cv_json
    if not cv_json_payload and request.cv_text:
        import json
        try:
            if request.cv_text.strip().startswith("{"):
                cv_json_payload = json.loads(request.cv_text)
            else:
                cv_json_payload = {"raw_text": request.cv_text}
        except:
            cv_json_payload = {"raw_text": request.cv_text}
    
    jd_data = {
        "title": request.job_data.get("title", ""),
        "department": request.job_data.get("department", "")
    }
    
    try:
        questions = generate_interview_questions(
            cv_json=cv_json_payload,
            jd_data=jd_data,
            match_details=request.match_details
        )
        return {"questions": questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate questions: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
