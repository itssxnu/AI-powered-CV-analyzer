<h1 align="center">
  <br>
  <img src="https://img.icons8.com/nolan/128/artificial-intelligence.png" alt="RecruitAI Logo" width="128">
  <br>
  RecruitAI
  <br>
</h1>

<h4 align="center">An Enterprise-Grade AI-Powered CV Matching, Parsing & Algorithmic Fairness Auditing Platform. Built on Spring Boot & FastAPI.</h4>

<p align="center">
  <img src="https://img.shields.io/badge/Platform-RecruitAI-8A2BE2?style=for-the-badge&logo=artificial-intelligence" alt="Platform">
  <img src="https://img.shields.io/badge/Backend-Spring%20Boot-6DB33F?style=for-the-badge&logo=springboot&logoColor=white" alt="Spring Boot">
  <img src="https://img.shields.io/badge/AI_Service-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Status-Active_%26_Ready-brightgreen?style=for-the-badge" alt="Status">
</p>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#system-architecture">System Architecture</a> •
  <a href="#tech-stack">Tech Stack</a> •
  <a href="#installation-and-setup">Installation</a> •
  <a href="#algorithmic-fairness">Fairness Check</a> •
  <a href="#contributors">Contributors</a>
</p>

---

## 🖥️ Platform Showcase

<p align="center">
  <img src="./assets/landing_page.png" alt="RecruitAI Landing Page" width="850" style="border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.45);">
</p>

---

## 🚀 Key Features

*   📂 **Multi-Modal CV Parsing**: Combines native `PyPDF2` text extraction with `Tesseract OCR` fallbacks, feeding into **Google Gemini 2.5 Flash** structured schema generation to map resumes with high confidence.
*   🧠 **Semantic Skill Matching**: Uses local **SentenceTransformers** (`all-MiniLM-L6-v2`) to perform vector cosine similarity matching between CV profiles and Job Descriptions, avoiding false positive matches.
*   📊 **Predictive Machine Learning Ranking**: Computes overall candidate fit using a pre-trained **Random Forest Regressor** and tiers career levels (Junior, Mid, Senior) using a custom **Decision Tree Classifier**.
*   ⚖️ **Algorithmic Fairness Check**: Runs real-time bias audits by synthetically cloning candidates into 16 demographic variants to monitor parity across 12 protected traits (gender, age, ethnicity, religion, disability, etc.).
*   🎙️ **Anonymized AI Interviewer**: Purges candidate PII and leverages **OpenRouter LLMs** to generate customized behavioral and technical interview questions based on the candidate's specific skill gaps.
*   🛡️ **Enterprise Security & Role Access**: Integrates robust Spring Security supporting role-based access for HR Managers, Candidates, and System Administrators.

---

## 🏛️ System Architecture

RecruitAI leverages a decoupled dual-service microservice layout connected via HTTP REST API boundaries:

```mermaid
graph LR
  subgraph Client ["Client Interface"]
    Thyme["Thymeleaf Web UI"]
  end

  subgraph Core ["Spring Boot Enterprise Backend"]
    Ctrl["Spring Controllers"]
    Serv["Spring Services"]
    JPA["Spring Data JPA"]
    DB[("MySQL Database")]
  end

  subgraph AI ["FastAPI AI Service"]
    FastAPI["FastAPI App Router"]
    Parser["pdf_extractor.py"]
    Matcher["matcher.py"]
    Classifier["classifier.py"]
    Auditor["bias_detector.py"]
  end

  Thyme <--> Ctrl
  Ctrl <--> Serv
  Serv <--> JPA
  JPA <--> DB
  Serv -- "HTTP REST" --> FastAPI
  FastAPI <--> DB
  FastAPI --> Parser
  FastAPI --> Matcher
  FastAPI --> Classifier
  FastAPI --> Auditor
```

---

## 🛠️ Tech Stack

<table align="center">
  <tr>
    <td align="center" width="96">
      <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/java/java-original.svg" alt="Java" width="48" height="48">
      <br>Java 21
    </td>
    <td align="center" width="96">
      <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/spring/spring-original.svg" alt="Spring Boot" width="48" height="48">
      <br>Spring Boot
    </td>
    <td align="center" width="96">
      <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/mysql/mysql-original.svg" alt="MySQL" width="48" height="48">
      <br>MySQL
    </td>
    <td align="center" width="96">
      <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="Python" width="48" height="48">
      <br>Python 3.10+
    </td>
    <td align="center" width="96">
      <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/fastapi/fastapi-original.svg" alt="FastAPI" width="48" height="48">
      <br>FastAPI
    </td>
    <td align="center" width="96">
      <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/bootstrap/bootstrap-original.svg" alt="Bootstrap" width="48" height="48">
      <br>Thymeleaf
    </td>
  </tr>
</table>

---

## 🛡️ Real-Time GitHub Analytics

<p align="center">
  <img src="https://github-readme-stats.vercel.app/api?username=itssxnu&theme=radical&show_icons=true" alt="GitHub Stats" width="400">
  <img src="https://github-readme-stats.vercel.app/api/top-langs/?username=itssxnu&langs_count=5&layout=compact&theme=radical" alt="Top Languages" width="350">
</p>

---

## ⚙️ Installation and Setup

### Prerequisites
*   **Java JDK 21+**
*   **MySQL Server 8.0+**
*   **Python 3.10+** (with virtual environment support)
*   **Tesseract OCR** (for image text fallback)

### Step 1: Database Setup
Launch MySQL Workbench or terminal and create the relational schema:
```sql
CREATE DATABASE aiml_project CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### Step 2: Spring Boot Backend
1. Create your local properties file from the provided example:
   ```bash
   cp backend/src/main/resources/application.properties.example backend/src/main/resources/application.properties
   ```
2. Update the database password on Line 7:
   ```properties
   spring.datasource.password=YOUR_MYSQL_PASSWORD
   ```
3. Boot up the server:
   ```bash
   cd backend
   ./mvnw spring-boot:run
   ```
   *The backend will run on **http://localhost:8080***

### Step 3: FastAPI Python ML Service
1. Create your local environment file:
   ```bash
   cp cv_model/.env.example cv_model/.env
   ```
2. Update the `.env` keys securely (ask the project lead for keys):
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   OPENROUTER_API_KEY=your_openrouter_key_here
   ```
3. Initialize the Python virtual environment and run the server:
   ```bash
   cd cv_model
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   pip install -r requirements.txt
   python api.py
   ```
   *The FastAPI AI service will run on **http://localhost:8000***

---

## ⚖️ Algorithmic Fairness & Bias Auditing

RecruitAI guarantees fairness by auditing candidate scores across **12 protected axes**. You can run a direct algorithmic health check via the API:

```http
POST http://localhost:8000/fairness-audit
```

**Audit Response Metrics**:
```json
{
  "status": "PASSED",
  "failures": [],
  "details": {
    "Baseline": {"score": 92.5, "status": "FAIR"},
    "Gender_FemaleName": {"score": 92.5, "status": "FAIR"},
    "Age_Older_GenX": {"score": 92.5, "status": "FAIR"},
    "Ethnicity_NameA": {"score": 92.5, "status": "FAIR"}
  }
}
```
---
<p align="center">
  Distributed under the MIT License. See <code>LICENSE</code> for more information.
</p>
