import json
import logging
from typing import Optional
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import joblib
import pandas as pd

from database import MatchResult, MatchLabel, EmbeddingCache
from skill_extractor import extract_skills

logger = logging.getLogger(__name__)

EMBEDDING_MEMORY_CACHE = {}
DEBUG_MODE = os.environ.get("DEBUG_MODE", "True").lower() == "true"

class EmbeddingService:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            try:
                logger.info("Loading sentence-transformers model...")
                cls._model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load sentence-transformers model: {e}")
        return cls._model

# Initialize Random Forest Ranker Model
RANKER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "candidate_ranker.pkl")
try:
    logger.info("Loading Candidate Ranker Model...")
    ranker_model = joblib.load(RANKER_MODEL_PATH)
except Exception as e:
    logger.error(f"Failed to load ranker model: {e}")
    ranker_model = None

def get_embedding(text: str, db: Session) -> list[float]:
    """Get embedding from cache or generate it."""
    import hashlib
    text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    if text_hash in EMBEDDING_MEMORY_CACHE:
        return EMBEDDING_MEMORY_CACHE[text_hash]
    
    cached = db.query(EmbeddingCache).filter(EmbeddingCache.text_hash == text_hash).first()
    if cached:
        embedding = json.loads(cached.embedding_vector)
        EMBEDDING_MEMORY_CACHE[text_hash] = embedding
        return embedding
    
    model = EmbeddingService.get_model()
    if model is None:
        raise RuntimeError("Semantic model not loaded.")
        
    embedding = model.encode(text).tolist()
    
    new_cache = EmbeddingCache(
        text_hash=text_hash,
        phrase=text,
        embedding_vector=json.dumps(embedding)
    )
    db.add(new_cache)
    db.commit()
    
    EMBEDDING_MEMORY_CACHE[text_hash] = embedding
    
    if DEBUG_MODE:
        logger.info(f"DEBUG: Encoded '{text}', vector len: {len(embedding)}, first 5: {embedding[:5]}")
    return embedding

def get_embeddings_batch(texts: list[str], db: Session) -> list[list[float]]:
    """Batch embed multiple texts to optimize model usage."""
    if not texts:
        return []
        
    import hashlib
    results = [None] * len(texts)
    missing_indices = []
    missing_texts = []
    
    for i, text in enumerate(texts):
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if text_hash in EMBEDDING_MEMORY_CACHE:
            results[i] = EMBEDDING_MEMORY_CACHE[text_hash]
        else:
            cached = db.query(EmbeddingCache).filter(EmbeddingCache.text_hash == text_hash).first()
            if cached:
                emb = json.loads(cached.embedding_vector)
                EMBEDDING_MEMORY_CACHE[text_hash] = emb
                results[i] = emb
            else:
                missing_indices.append(i)
                missing_texts.append(text)
                
    if missing_indices:
        model = EmbeddingService.get_model()
        if model is None:
            raise RuntimeError("Semantic model not loaded.")
            
        embs = model.encode(missing_texts).tolist()  # batch processing
        
        for idx, text, emb in zip(missing_indices, missing_texts, embs):
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            new_cache = EmbeddingCache(
                text_hash=text_hash,
                phrase=text,
                embedding_vector=json.dumps(emb)
            )
            db.add(new_cache)
            EMBEDDING_MEMORY_CACHE[text_hash] = emb
            results[idx] = emb
            
            if DEBUG_MODE:
                logger.info(f"DEBUG: Batch-encoded '{text}', vector len: {len(emb)}, first 5: {emb[:5]}")
                
        db.commit()
    return results

def _match_skills(jd_skills: list[str], cv_skills_data: list[dict], db: Session, threshold: float = 0.75, partial_threshold: float = 0.25, cv_embeddings: list[list[float]] = None) -> tuple[int, list[dict], list[str], list[str]]:
    """Matches JD skills against CV skills, returns count, details, missing, partial."""
    if not jd_skills:
        return 0, [], [], []
        
    matched_count = 0
    matched_details = []
    missing_skills = []
    partial_skills = []
    
    # If not provided by precomputation, compute here
    if cv_embeddings is None:
        cv_phrases_to_embed = [skill["canonical_skills"] for skill in cv_skills_data]
        cv_embeddings = get_embeddings_batch(cv_phrases_to_embed, db)
        
    cv_phrases = cv_skills_data

    logger.info(f"Matching {len(jd_skills)} JD skills against {len(cv_phrases)} CV skills...")

    jd_skills_clean = [str(s).strip() for s in jd_skills if str(s).strip()]
    jd_embeddings = get_embeddings_batch(jd_skills_clean, db)

    for jd_skill, jd_emb in zip(jd_skills_clean, jd_embeddings):
        best_sim = 0.0
        best_cv_skill = None
        
        if cv_embeddings:
            cos_scores = util.cos_sim(jd_emb, cv_embeddings)[0]
            best_idx = np.argmax(cos_scores).item()
            best_sim = cos_scores[best_idx].item()
            
            if best_sim >= threshold:
                best_cv_skill = cv_phrases[best_idx]
        
        if best_sim >= threshold:
            matched_count += 1
            matched_details.append({
                "jd_skill": jd_skill,
                "best_cv_skill": best_cv_skill["surface_skills"],
                "canonical_matched": best_cv_skill["canonical_skills"],
                "evidence": best_cv_skill["evidence"],
                "similarity_score": round(best_sim, 3)
            })
            if DEBUG_MODE:
                logger.info(f"DEBUG Match: JD '{jd_skill}' vs CV '{best_cv_skill['canonical_skills']}' | score: {round(best_sim, 3)}")
        elif best_sim >= partial_threshold:
            partial_skills.append(jd_skill)
            if DEBUG_MODE:
                logger.info(f"DEBUG Partial Match: JD '{jd_skill}' top score was {round(best_sim, 3)}")
        else:
            missing_skills.append(jd_skill)
            if DEBUG_MODE:
                logger.info(f"DEBUG Miss: JD '{jd_skill}' top score was {round(best_sim, 3)}")
            
    return matched_count, matched_details, missing_skills, partial_skills

def evaluate_match(cv_data: dict, jd_data: dict, db: Session, precomputed_cv_skills: list[dict] = None, precomputed_cv_embeddings: list[list[float]] = None) -> dict:
    """Evaluates cv against jd."""
    # 1. Extract CV Skills
    if precomputed_cv_skills is not None:
        cv_skills_extracted = precomputed_cv_skills
    else:
        cv_skills_extracted = extract_skills(cv_data)
        
    if precomputed_cv_embeddings is not None:
        cv_emb_vectors = precomputed_cv_embeddings
    else:
        cv_canonical_skills = [skill["canonical_skills"] for skill in cv_skills_extracted]
        cv_emb_vectors = get_embeddings_batch(cv_canonical_skills, db)
    
    # 2. Extract JD Skills - Trust the literal HR input strings instead of running them through the synonym extractor 
    # which heavily drops words that aren't in the canonical vocabulary.
    req_skills, pref_skills = [], []
    
    if jd_data.get("required_skills_json"):
        try:
            req_raw = json.loads(jd_data["required_skills_json"])
            if isinstance(req_raw, list):
                req_skills = [str(s).strip() for s in req_raw if str(s).strip()]
            elif isinstance(req_raw, str):
                req_skills = [s.strip() for s in req_raw.split(',') if s.strip()]
        except: pass
        
    if jd_data.get("preferred_skills_json"):
        try:
            pref_raw = json.loads(jd_data["preferred_skills_json"])
            if isinstance(pref_raw, list):
                pref_skills = [str(s).strip() for s in pref_raw if str(s).strip()]
            elif isinstance(pref_raw, str):
                pref_skills = [s.strip() for s in pref_raw.split(',') if s.strip()]
        except: pass
        
    # 3. Match Skills
    # Thresholds adjusted to 0.60 and partial to 0.45 to prevent false positives.
    # A threshold of 0.32 or 0.25 is too low for sentence-transformers and causes 
    # completely unrelated skills (like "Software Engineering" and "Front Desk Operations")
    # to match due to background semantic noise.
    req_matched, req_details, req_missing, req_partial = _match_skills(
        req_skills, cv_skills_extracted, db, threshold=0.60, partial_threshold=0.45, cv_embeddings=cv_emb_vectors
    )
    pref_matched, pref_details, pref_missing, pref_partial = _match_skills(
        pref_skills, cv_skills_extracted, db, threshold=0.55, partial_threshold=0.40, cv_embeddings=cv_emb_vectors
    )
    
    # 4. Calculate Coverages
    req_total = len(req_skills) if req_skills else 0
    pref_total = len(pref_skills) if pref_skills else 0
    
    print(f"\n[{jd_data.get('title', 'Unknown JD')}] vs CV extracted skills ({len(cv_skills_extracted)}):")
    print(f"JD req skills: {req_skills} | matches: {req_matched}/{req_total}")
    print(f"JD pref skills: {pref_skills} | matches: {pref_matched}/{pref_total}")
    
    # Logic Fix: If no skills are defined, coverage should be 0 or a neutral value, not 1.0 (100%)
    # because it falsely inflates scores for jobs without data.
    req_coverage = (req_matched / req_total) if req_total > 0 else 0.0
    pref_coverage = (pref_matched / pref_total) if pref_total > 0 else 0.0
    
    # 5. Experience & Department Fit for ML Model
    cv_years = 0.0
    # Module 3 classifier integration to calculate actual YoE robustly
    from classifier import calculate_total_experience
    cv_years = calculate_total_experience(cv_data)
    
    jd_min_years = jd_data.get("min_years", 0) or 0
    
    exp_val = 0.0 # 0=none, 1=low, 2=meets, 3=exceeds
    if cv_years <= 0:
        exp_val = 0.0  # No experience at all — 'none'
    elif cv_years >= jd_min_years + 2:
        exp_val = 3.0  # Exceeds requirement
    elif cv_years >= jd_min_years and jd_min_years > 0:
        exp_val = 2.0  # Meets requirement
    elif cv_years >= jd_min_years:  # jd_min_years == 0, and cv_years > 0
        exp_val = 2.0  # Has some experience, no requirement set — treat as meets
    else:
        exp_val = 1.0  # Below requirement but has some experience
        
    dept_fit = 0.0
    jd_dept = str(jd_data.get("department", "")).lower()
    hosp = cv_data.get("hospitality", {})
    if not jd_dept:
        # No department on the JD — neutral, don't penalise
        dept_fit = 1.0
    elif hosp and "departments_worked" in hosp and hosp["departments_worked"]:
        depts = [str(d).lower() for d in hosp["departments_worked"]]
        if any(jd_dept in d or d in jd_dept for d in depts):
            dept_fit = 1.0
        # else: dept_fit stays 0.0 — department mismatch
    # else: no department info in CV, dept_fit stays 0.0 — unknown = penalise
        
    # 6. ML Model Ranker Score Calculation
    score = 0.0
    if ranker_model:
        features = pd.DataFrame([{
            "req_coverage": float(req_coverage),
            "pref_coverage": float(pref_coverage),
            "experience_fit": float(exp_val),
            "department_fit": float(dept_fit)
        }])
        print(f"Features -> req_cov: {req_coverage:.2f}, pref_cov: {pref_coverage:.2f}, exp_val: {exp_val}, dept_fit: {dept_fit}")
        score = float(ranker_model.predict(features)[0])
        print(f"ML Model Output Score: {round(score, 2)}\n")
    else:
        # Fallback to hardcoded heuristic (Improved)
        # Weights: Req Skills (60%), Pref Skills (20%), Experience (10%), Department (10%)
        # If req_total is 0, we rebalance the weights.
        if req_total == 0:
            score = (40 * pref_coverage) + (30 * (exp_val/3.0)) + (30 * dept_fit)
        else:
            score = (60 * req_coverage) + (20 * pref_coverage) + (10 * (exp_val/3.0)) + (10 * dept_fit)
        
        print(f"Heuristic Score: {round(score, 2)} (req_cov: {round(req_coverage, 2)}, exp: {exp_val})\n")
        
    score = max(0.0, min(100.0, score)) # Clamp 0-100
    
    # 7. Determine Label
    if score >= 75:
        label = MatchLabel.good
    elif score >= 50:
        label = MatchLabel.close
    else:
        label = MatchLabel.bad
        
    return {
        "score": round(score, 2),
        "label": label,
        "details": {
            "matched_required": req_details,
            "missing_required": req_missing,
            "partial_required": req_partial,
            "matched_preferred": pref_details,
            "summary_metrics": {
                "required_coverage": round(req_coverage, 2),
                "preferred_coverage": round(pref_coverage, 2),
                "experience_fit": round(exp_val, 2), # Exposing the internal ordinal var
                "department_fit": dept_fit,
                "ai_predicted_score": round(score, 2)
            }
        }
    }
