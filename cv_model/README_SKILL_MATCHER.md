# 🧠 Skill Matching Engine (Module 2)

This module is responsible for bridging the gap between a parsed curriculum vitae JSON (from Module 1) and a structured Job Description located in the `'hotel_ai'` MySQL database. 

It accomplishes this through semantic AI skill interpretation, translating soft-skills and hard-skills into numerical evaluation vectors.

## 🚀 Features

* **Hospitality Native Extraction:** Automatically filters and standardizes raw CV lists using the `hotel_synth_dataset/synonyms.json` matrix. (e.g. mapping "Guest check-in" and "Check in/out" to the exact same Canonical root).
* **Machine Learning Matching Engine:** Uses `sentence-transformers/all-MiniLM-L6-v2` to mathematically compute the cosine similarity between the Job Requirement semantic vector and the CV semantic vector.
* **Intelligent Score Weights (0-100%):** Calculates highly explainable scores by considering exactly how many *required* skills were matched (`>= 0.70` similarity), *preferred* skills (`>= 0.65` similarity), and exact experience (`min_years`) fitment.
* **Persistent Tracking:** Saves the candidate's `cv_id` and name directly into the `'cv_parsed'` MySQL table, and establishes their score linkage in the `'match_results'` table.
* **Vector Caching:** Accelerates performance by computing inference on individual skills exactly once, storing the matrix output into `embedding_cache`.

## ⚙️ Prerequisites

1. Standard Python libraries: `fastapi`, `uvicorn`, `sqlalchemy`, `mysql-connector-python`.
2. Machine Learning core: `sentence-transformers` (PyTorch). 

Install dependencies via:
```bash
python -m pip install -r requirements.txt
```

## 💻 API Usage

This module exposes a REST API built with **FastAPI**. To run the engine:

```bash
uvicorn api:app --reload
```

### `POST /match`

Evaluates the incoming CV candidate against the active database.

**Form Request Parameters:**
- `cv_json` (Optional, Dict): The parsed output from Module 1. 
- `cv_text` (Optional, String): A raw text string if sending an unparsed CV.
- `job_id` (Optional, int): If supplied, it evaluates against that specific JD. If missing, it evaluates the CV against the entire library.
- `source_file` (Optional, String): Identifying filename/label.
- `top_k` (Optional, int): Default 5. Trims bulk evaluations to the best `k` fits.

**Example cURL Response snippet:**
```json
{
  "cv_id": 4,
  "matches": [
    {
      "job_id": 1,
      "title": "Front Desk Agent",
      "department": "Front Office",
      "score": 63.33,
      "label": "close",
      "details": {
        "matched_required": [ ... ],
        "missing_required": [ "Front desk operations" ],
        "summary_metrics": {
           "final_score": 63.33
        }
      }
    }
  ]
}
```
