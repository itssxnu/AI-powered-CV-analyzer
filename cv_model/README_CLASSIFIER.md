# 📊 Experience and Role Suitability Classifier (Module 3)

This module systematically evaluates parsed CV data to determine a candidate's explicit **Years of Experience (YoE)** and assigns them an operational "Career Level" (`Junior`, `Mid`, or `Senior`).

It is designed to run automatically within the API pipeline directly after the PDF extraction phase, before semantic ranking.

## 🚀 Features

* **Dynamic Experience Calculation:** Calculates total years of experience mathematically by parsing varied date formats (e.g. "Jan 2018 - Present", "2 years", "2015-2019") from the structured 'experience' JSON array.
* **Skill Depth Analysis:** Scans the candidate's work history descriptions and `hospitality` tags for explicit flags indicating seniority, such as:
  * *Executive Leadership* (Director, VP, GM roles)
  * *Management/Supervisory* (Manager, Supervisor, Coordinator)
  * *Financial Responsibility* (Budgets, P&L)
  * *Enterprise Systems* (e.g. Opera PMS usage)
* **Categorical Labeling:** Evaluates the computed YoE and Leadership Flags against an internal ruleset to output a definitive label of `Junior`, `Mid`, or `Senior`.
* **Database Integration:** Saves the resulting `career_level` permanently to the candidates `cv_parsed` row in the MySQL database.
* **Traceable Explanations:** Outputs a plain-English explanation detailing precisely why the candidate was assigned their specific career level.

## ⚙️ Prerequisites

This module relies on native Python libraries (`re`, `datetime`, `json`) and does not require complex ML dependencies. It natively hooks into the `api.py` architecture.

## 💻 Standalone Usage

While typically invoked automatically via the `POST /match` endpoint, the logic can be tested locally using the dedicated test script:

```bash
python test_classifier.py
```

### Module Methods (`classifier.py`)

* **`calculate_total_experience(cv_json: dict) -> float`**
  Parses the provided JSON to determine the exact numeric total of years worked.
* **`evaluate_skill_depth(cv_json: dict) -> dict`**
  Scans experience blocks and role titles, returning dictionaries containing boolean management flags.
* **`classify_candidate(cv_json: dict) -> dict`**
  The main entry point. Orchestrates the YoE and Skill Depth logic to return the final label and text explanation.
