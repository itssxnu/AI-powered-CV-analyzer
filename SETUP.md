# 🛠️ Local Setup Guide

Follow these steps after cloning the repo. You need to create a few config files
that are **not committed** (they contain secrets/local paths).

---

## Prerequisites

| Tool | Version | Download |
|---|---|---|
| Java (JDK) | 21+ | https://adoptium.net |
| Maven | 3.9+ | bundled via `mvnw` |
| MySQL | 8.0+ | https://dev.mysql.com/downloads |
| Python | 3.10+ | https://www.python.org |
| pip | latest | included with Python |

---

## 1 — Clone the repo

```bash
git clone https://github.com/YOUR_ORG/YOUR_REPO.git
cd AI_proj
```

---

## 2 — Backend: `application.properties`

The real config file is **gitignored**. Create it from the template:

```bash
# Windows
copy backend\src\main\resources\application.properties.example ^
     backend\src\main\resources\application.properties

# macOS / Linux
cp backend/src/main/resources/application.properties.example \
   backend/src/main/resources/application.properties
```

Then open it and fill in **your** values:

```properties
# Line 7 — your local MySQL password
spring.datasource.password=YOUR_DB_PASSWORD_HERE

# Line 56-57 — only needed if you test email features
spring.mail.username=YOUR_EMAIL@gmail.com
spring.mail.password=YOUR_GMAIL_APP_PASSWORD
```

> **MySQL password** — whatever you set when installing MySQL on your machine.
> Leave everything else as-is for local development.

---

## 3 — Backend: create the database

Open MySQL Workbench (or the MySQL CLI) and run:

```sql
CREATE DATABASE aiml_project CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

Spring Boot will create/update the tables automatically on first run (`ddl-auto=update`).

---

## 4 — AI Service: `.env`

The real `.env` is **gitignored**. Create it from the template:

```bash
# Windows
copy cv_model\.env.example cv_model\.env

# macOS / Linux
cp cv_model/.env.example cv_model/.env
```

Then open `cv_model/.env` and fill in the API keys.
Ask your team lead to share them securely (e.g. via WhatsApp, shared password manager — **never over email/chat in plain text**):

```env
GEMINI_API_KEY=ask-team-lead
OPENROUTER_API_KEY=ask-team-lead
```

---

## 5 — AI Service: Python virtual environment

```bash
cd cv_model

# Create venv
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 6 — Run both services

### Backend (Spring Boot)
```bash
cd backend

# Windows
mvnw.cmd spring-boot:run

# macOS / Linux
./mvnw spring-boot:run
```
Opens at → **http://localhost:8080**

### AI Service (FastAPI)
```bash
cd cv_model
# Make sure .venv is activated first (step 5)
python api.py
```
Opens at → **http://localhost:8000**  
Swagger docs → **http://localhost:8000/docs**

---

## Quick checklist

```
[ ] JDK 21 installed
[ ] MySQL running with database "aiml_project" created
[ ] application.properties created and DB password filled in
[ ] Python .venv created and requirements installed
[ ] cv_model/.env created with API keys
[ ] Both services running (ports 8080 and 8000)
```

---

## Default login credentials (local dev)

| Role | Username | Password |
|---|---|---|
| Admin | `admin` | set via DataLoader / schema.sql |
| HR | `hr1` | set via DataLoader / schema.sql |
| Candidate | `john_doe` | set via DataLoader / schema.sql |

> Passwords are stored as bcrypt hashes — check `DataLoader.java` or `schema.sql`
> for the seeded credentials used in development.
