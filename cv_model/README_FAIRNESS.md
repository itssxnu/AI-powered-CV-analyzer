# ⚖️ Bias Detection and Fairness Checking (Module 5)

This module operates as a dedicated auditing pipeline to prove the CV Matching and Candidate Ranking logic is impervious to demographic bias.

It specifically ensures compliance against 12 different operational axes of potential Machine Learning prejudice.

## 🚀 Features

* **Comprehensive Generation Engine:** Parses a generic baseline curriculum vitae and synthetically creates identical clones featuring precisely injected demographic permutations.
* **12 Protected Traits Monitored:**
   * `Gender` (Male vs Female names)
   * `Age Group` (Gen Z vs Gen X graduation years)
   * `Ethnicity / Race` (Culturally distinct names)
   * `Nationality` (Foreign vs Local addresses)
   * `Disability Status` (Explicit extracurricular disability markers)
   * `Religion` (Explicit extracurricular markers)
   * `Name-based Stereotypes`
   * `Photo Metadata Presence` 
   * `University Prestige` (Ivy League vs Local Tier)
   * `Foreign vs Local Education`
   * `Field of Study Relevance Bias`
   * `Employment Gaps`
* **Automated Audit Script (`bias_detector.py`):** Passes all 17 derived combinations through the `sentence-transformers` evaluator and the `Random Forest` regressor to establish the mathematical truth boundary.
* **API Integration:** This module is natively exposed through a `POST /fairness-audit` endpoint, allowing HR Administrators to hit a button and receive an instant JSON statistical breakdown determining `Demographic Parity` and `Disparate Impact` flags.

## ⚙️ Testing Fairness Locally

If you modify the Ranker or Model in the future, you must verify fairness. You can run the audit locally:

```bash
python bias_detector.py
```

*Example Output:*
```
Baseline                  | Score: 100.00 | Diff:   0.00 | FAIR 
Gender_FemaleName         | Score: 100.00 | Diff:   0.00 | FAIR 
Name_Stereotype_1         | Score: 100.00 | Diff:   0.00 | FAIR 
...
FAIRNESS AUDIT REPORT:
  PASSED: Demographic Parity achieved. Model prediction is perfectly blind to all 12 protected traits.
```

## 💻 API Endpoint

When running the fast API via `uvicorn api:app --reload`, you may fetch a fairness health check.

### `POST /fairness-audit`

Requires no request body. Generates the baseline and its clones inside the backend, maps them through the full prediction pipeline, and directly returns the report.

**Responses Snippet:**
```json
{
  "status": "PASSED",
  "failures": [],
  "details": {
    "Baseline": {
      "score": 100.0,
      "diff_from_baseline": 0.0,
      "status": "FAIR",
      "experience_variance": 2.0
    },
    "Age_Older_GenX": {
      "score": 100.0,
      "diff_from_baseline": 0.0,
      "status": "FAIR",
      "experience_variance": 2.0
    }
  }
}
```
