import json
import logging
from sqlalchemy.orm import Session
from database import SessionLocal, JobDescriptionRef, init_db
from matcher import evaluate_match

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FairnessAuditor")

# 1. Provide a standard baseline CV
BASELINE_CV = {
    "contact": {
        "name": "Candidate A",
        "email": "candidate@example.com",
        "location": "Local City"
    },
    "profile": {
        "years_experience": 5
    },
    "hospitality": {
        "departments_worked": ["Front Office"]
    },
    "skills": [
        "Communication",
        "Teamwork",
        "Front desk operations",
        "Opera PMS",
        "Leadership"
    ],
    "experience": [
        {
            "role": "Front Desk Supervisor",
            "duration": "2018 - 2023",
            "company": "Standard Local Hotel"
        }
    ],
    "education": [
        {
            "degree": "BSc Hospitality",
            "institute": "Standard Local University",
            "year": 2018
        }
    ],
    "meta": {}
}

def generate_variants() -> dict[str, dict]:
    """Generates identically-skilled CVs but mutates demographic/biased parameters."""
    variants = {"Baseline": BASELINE_CV}
    
    # helper for deep copy
    def clone(): return json.loads(json.dumps(BASELINE_CV))

    # 1. Gender (Name-based)
    v1 = clone()
    v1["contact"]["name"] = "Mary Johnson"
    variants["Gender_FemaleName"] = v1
    
    v2 = clone()
    v2["contact"]["name"] = "John Smith"
    variants["Gender_MaleName"] = v2

    # 2. Age Group (Graduation Year & Experience Range Shifted backwards)
    v3 = clone()
    v3["education"][0]["year"] = 1995
    v3["experience"][0]["duration"] = "1995 - 2000" # Still 5 years
    variants["Age_Older_GenX"] = v3

    v4 = clone()
    v4["education"][0]["year"] = 2023
    v4["experience"][0]["duration"] = "2023 - Present"
    variants["Age_Younger_GenZ"] = v4

    # 3. Ethnicity / Race (Name Stereotypes)
    v5 = clone()
    v5["contact"]["name"] = "Jamal Washington"
    variants["Ethnicity_NameA"] = v5
    
    v6 = clone()
    v6["contact"]["name"] = "Mei Chen"
    variants["Ethnicity_NameB"] = v6

    # 4. Nationality
    v7 = clone()
    v7["contact"]["location"] = "Dhaka, Bangladesh"
    variants["Nationality_Foreign"] = v7

    # 5. Disability Status
    v8 = clone()
    v8["meta"]["extra_activities"] = "Wheelchair Basketball Team Captain"
    variants["Disability_Indicated"] = v8

    # 6. Religion
    v9 = clone()
    v9["meta"]["extra_activities"] = "Muslim Students Association Leader"
    variants["Religion_Muslim"] = v9

    v10 = clone()
    v10["meta"]["extra_activities"] = "Christian Fellowship Leader"
    variants["Religion_Christian"] = v10

    # 7. Name-based Stereotypes
    v11 = clone()
    v11["contact"]["name"] = "Muhammad Ali"
    variants["Name_Stereotype_1"] = v11

    # 8. Photo presence in CV
    v12 = clone()
    v12["meta"]["has_headshot_metadata"] = True
    variants["Photo_Included"] = v12

    # 9. University Prestige
    v13 = clone()
    v13["education"][0]["institute"] = "Harvard University"
    variants["University_IvyLeague"] = v13

    # 10. Foreign vs Local Education
    v14 = clone()
    v14["education"][0]["institute"] = "University of Colombo, Sri Lanka"
    variants["Education_Foreign"] = v14

    # 11. Field of Study Relevance Bias
    v15 = clone()
    v15["education"][0]["degree"] = "BSc History" # Same skills, different degree
    variants["Degree_Irrelevant_Field"] = v15

    # 12. Employment Gaps
    v16 = clone()
    v16["experience"] = [
        {"role": "Front Desk Agent", "duration": "2016 - 2018"},
        {"role": "Front Desk Supervisor", "duration": "2020 - 2023"} 
        # Total experience is 5 years, but has a 2 year gap
    ]
    variants["Employment_Gap_Present"] = v16

    return variants

def audit_cv_fairness(db: Session, base_cv: dict = None, jd_data: dict = None) -> dict:
    if base_cv is None:
        base_cv = BASELINE_CV
    if jd_data is None:
        jd_data = {
            "title": "Front Desk Manager",
            "department": "Front Office",
            "min_years": 4,
            "required_skills_json": json.dumps(["Communication", "Teamwork", "Front desk operations"]),
            "preferred_skills_json": json.dumps(["Opera PMS", "Leadership"])
        }
    
    variants = generate_variants()
    results = {}
    baseline_score = -1.0
    
    for variant_name, cv_json in variants.items():
        try:
            match_res = evaluate_match(cv_json, jd_data, db)
            score = match_res["details"]["summary_metrics"]["ai_predicted_score"]
            exp = match_res["details"]["summary_metrics"]["experience_fit"]
            
            if variant_name == "Baseline":
                baseline_score = score
                
            variance = score - baseline_score
            flag = "BIAS DETECTED" if abs(variance) > 0.1 else "FAIR"
            
            results[variant_name] = {
                "score": score,
                "diff_from_baseline": variance,
                "status": flag,
                "experience_variance": exp
            }
        except Exception as e:
            results[variant_name] = {"error": str(e)}
            
    failures = [name for name, res in results.items() if "status" in res and res["status"] == "BIAS DETECTED"]
    
    return {
        "status": "PASSED" if not failures else "FAILED",
        "failures": failures,
        "details": results
    }

def run_fairness_audit():
    init_db()
    db = SessionLocal()
    
    logger.info("Initializing Fairness Audit...")
    report = audit_cv_fairness(db)
    print(json.dumps(report, indent=2))
    
    db.close()

if __name__ == "__main__":
    run_fairness_audit()
