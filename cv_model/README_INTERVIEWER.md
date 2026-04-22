# 🗣️ Interview Question Generator (Module 6)

The final module of the CV Skill Matching Pipeline. After a candidate has been fully parsed, mapped against the job description, scored out of 100 via the ML Ranker, and audited for Fairness, this module bridges the gap to the human recruiter.

It dynamically constructs an **Anonymized Candidate Profile** and leverages OpenRouter Serverless LLMs to instantly generate personalized interview questions based exactly on what the candidate *knows* and what they *lack*.

## 🚀 Features

* **PII Privacy Shield (`scrub_cv`)**: Before sending any data to external OpenRouter APIs, the candidate's Contact Dictionary (Name, Phone, Email, Address, LinkedIn) is completely eradicated from the JSON payload. The LLM only receives their objective skills, education, and bullet points.
* **Provider Agnosticism (OpenRouter)**: Connects to the universal OpenRouter gateway via the `openai` Python SDK. This completely isolates the Interview Generation rate-limit buckets from the Gemini parsing rate limits logic used in Module 1.
* **Nvidia Nemotron Power**: Defaults to using the free, blazing-fast `nvidia/nemotron-3-nano-30b-a3b:free` model.
* **Intelligent Prompt Engineering**:
    * Identifies a **Matched Skill** and asks the LLM to generate a deep-dive Behavioral (STAR method) question to verify if the candidate actually possesses the skill.
    * Identifies a **Missing Skill (Gap)** and asks the LLM to generate a probing question to reveal if the candidate possesses transferable, adjacent experience.
    * Generates an **Ideal Answer Key** so the Human Recruiter knows exactly what signals to look out for.

## ⚙️ Prerequisites

You must set your API key in the `.env` file at the root directory:

```env
OPENROUTER_API_KEY=sk-or-v1-abcdefg...
OPENROUTER_MODEL=nvidia/nemotron-3-nano-30b-a3b:free # Optional, overrides default
```

*Ensure the matching library is installed:*
```bash
python -m pip install openai python-dotenv
```

## 💻 API Endpoint

When running the fast API via `uvicorn api:app --reload`, you may fetch custom questions for any candidate already in the database.

### `GET /generate-questions/{cv_id}/{job_id}`

The backend looks up the candidate's parsed JSON, the Job Description requirements, and the specific Skill Match intersection data. It scrubs it, asks OpenRouter for questions, and returns an array.

**Example Response Snippet:**
```json
{
  "questions": [
    {
      "category": "Matched Skill",
      "focus": "Communication",
      "question": "Can you describe a situation where you had to communicate complex operational updates to both guests and staff during a busy check-in period?",
      "ideal_answer": "Look for a clear Situation, specific Task, concrete Actions, and measurable Results such as reduced confusion."
    },
    {
      "category": "Missing Skill",
      "focus": "Conflict Resolution",
      "question": "Describe an instance where you encountered a disagreement between team members. How did you approach resolving it?",
      "ideal_answer": "Seek evidence of systematic conflict-resolution steps."
    }
  ]
}
```
