# 🤖 Candidate Ranking Model (Module 4)

This module represents the predictive ML pipeline of the application. It replaces hardcoded, heuristic scoring equations with a fully-trained **Random Forest Regressor** to predict the final Job Fit Match Score (0-100) based on historical human-recruiter data.

## 🚀 Features

* **Data-Driven Predictions:** Instead of arbitrary math formulas (e.g. `70% required + 20% preferred`), the system uses supervised machine learning to predict exact match scores.
* **Random Forest Regressor:** A robust ensemble decision-tree built via `scikit-learn` that beautifully captures non-linear features (e.g. heavily penalizing a candidate if they completely mismatch the target department).
* **High Precision:** Achieved an outstanding **0.07 Mean Absolute Error (MAE)** upon testing the 6,000 synthetic `cv-to-job` pairs dataset (`hotel_synth_dataset/pairs.jsonl`).
* **Microsecond Inference:** The trained model is serialized via `joblib` into a binary `.pkl` pipeline, allowing `matcher.py` to instantly load it into memory and perform inferences at lightning speeds during API requests.
* **Feature Explainability:** Retains the exact same raw JSON format (listing matched and missing skills), but simply computes the ultimate `ai_predicted_score`.

## ⚙️ Prerequisites

This module necessitates supervised ML dependencies.

Install dependencies via:
```bash
python -m pip install -r requirements.txt
```
*(Ensure `scikit-learn`, `pandas`, and `joblib` are installed)*

## 🧠 Training & Re-Training the Model

If the underlying `hotel_synth_dataset/pairs.jsonl` dataset changes and you need the AI Engine to "learn" the new recruiter scoring habits, run the standalone training script:

```bash
python train_ranker.py
```

**Executing this script will:**
1. Parse all thousands of pairs, extracting numerical feature metrics (Required Coverage, Preferred Coverage, Ordinal Experience Fit, Boolean Department Fit).
2. Clean and format training/testing splits via `pandas`.
3. Train the Random Forest Regressor and evaluate accuracy (MAE and $R^2$ score).
4. Output calculated **Feature Importances** (e.g., demonstrating that `req_coverage` commands `93%` importance).
5. Overwrite the prevailing `candidate_ranker.pkl` file, which the core API instances will load upon next initialization. 
    
## 💻 Integration (Pipeline Flow)

This Module seamlessly inserts into the pre-existing End-to-End Pipeline:

1. **(Module 1 - CV Parser)** Extracts candidate text from `.pdf` to structured JSON.
2. **(Module 2 - AI Embeddings)** Computes Semantic Cosine Similarities to calculate raw numerical features (`req_coverage = 0.67`).
3. **(Module 3 - Classifier)** Parses specific Work Histories to classify explicit `Years of Experience` metric (`exp_fit = 2.0`).
4. **(Module 4 - ML Ranker)** Feeds the raw combination `[0.67, 0.33, 2.0, 1.0]` into the `candidate_ranker.pkl` file to predict score: **`67.05`**.
