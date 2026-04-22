-- Create database
CREATE DATABASE IF NOT EXISTS aiml_project;
USE aiml_project;

-- Users table (simplified - resume-related fields removed)
CREATE TABLE IF NOT EXISTS users (
                                     id BIGINT AUTO_INCREMENT PRIMARY KEY,
                                     username VARCHAR(50) UNIQUE NOT NULL,
                                     email VARCHAR(100) UNIQUE NOT NULL,
                                     password VARCHAR(255) NOT NULL,
                                     first_name VARCHAR(50) NOT NULL,
                                     last_name VARCHAR(50) NOT NULL,
                                     phone VARCHAR(15),
                                     role ENUM('CANDIDATE', 'HR', 'ADMIN') DEFAULT 'CANDIDATE',

    -- Professional fields (keep these as their user profile data)
                                     skills VARCHAR(500),
                                     experience_years INT,
                                     education VARCHAR(200),

    -- Status
                                     is_active BOOLEAN DEFAULT TRUE,

    -- Audit fields
                                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                     updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

                                     INDEX idx_role (role),
                                     INDEX idx_active (is_active),
                                     FULLTEXT idx_search (first_name, last_name, email, skills)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- New Resume table (separate from User)
CREATE TABLE IF NOT EXISTS resumes (
                                       id BIGINT AUTO_INCREMENT PRIMARY KEY,
                                       user_id BIGINT NOT NULL,

    -- File metadata
                                       file_name VARCHAR(255) NOT NULL,
                                       original_name VARCHAR(255) NOT NULL,
                                       content_type VARCHAR(100) NOT NULL,
                                       file_size BIGINT NOT NULL,
                                       file_path VARCHAR(500) NOT NULL,
                                       upload_date DATETIME NOT NULL,

    -- Extracted text from PDF
                                       extracted_text LONGTEXT,

    -- AI/ML processed data
                                       ml_processed BOOLEAN DEFAULT FALSE,
                                       ml_score DOUBLE,
                                       ml_confidence DOUBLE,
                                       ml_processed_date DATETIME,
                                       ml_raw_response TEXT, -- Store raw API response if needed

    -- Version tracking (for resume updates)
                                       version INT DEFAULT 1,
                                       is_current BOOLEAN DEFAULT TRUE,

    -- Audit fields
                                       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                       updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

                                       FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                                       INDEX idx_user_resume (user_id),
                                       INDEX idx_ml_processed (ml_processed),
                                       INDEX idx_is_current (is_current)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Jobs table
CREATE TABLE IF NOT EXISTS jobs (
                                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                                    title VARCHAR(255) NOT NULL,
                                    description TEXT NOT NULL,
                                    department VARCHAR(100) NOT NULL,
                                    location VARCHAR(100) NOT NULL,
                                    job_type ENUM('FULL_TIME', 'PART_TIME', 'CONTRACT', 'INTERNSHIP', 'REMOTE', 'HYBRID') DEFAULT 'FULL_TIME',
                                    experience_required VARCHAR(100) NOT NULL,
                                    required_skills VARCHAR(1000),
                                    preferred_skills VARCHAR(1000),
                                    education_requirement VARCHAR(255) NOT NULL,
                                    min_salary INT,
                                    max_salary INT,
                                    posted_date DATETIME NOT NULL,
                                    expiry_date DATETIME,
                                    is_active BOOLEAN DEFAULT TRUE,
                                    vacancies INT NOT NULL DEFAULT 1,
                                    posted_by BIGINT NOT NULL,
                                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

                                    FOREIGN KEY (posted_by) REFERENCES users(id),
                                    INDEX idx_job_active (is_active),
                                    INDEX idx_job_dept (department),
                                    INDEX idx_job_type (job_type),
                                    FULLTEXT idx_job_search (title, description, department, required_skills)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Job applications table (updated to reference resumes)
CREATE TABLE IF NOT EXISTS job_applications (
                                                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                                                job_id BIGINT NOT NULL,
                                                candidate_id BIGINT NOT NULL,
                                                resume_id BIGINT, -- Reference to the resume used for this application
                                                applied_date DATETIME NOT NULL,
                                                status ENUM('PENDING', 'SHORTLISTED', 'INTERVIEW_SCHEDULED', 'REJECTED', 'HIRED', 'WITHDRAWN') DEFAULT 'PENDING',
                                                status_updated_date DATETIME,
                                                hr_notes TEXT,

    -- AI matching scores (calculated at time of application)
                                                match_score DOUBLE,
                                                skills_match_score DOUBLE,
                                                experience_match_score DOUBLE,
                                                education_match_score DOUBLE,
                                                ml_processed BOOLEAN DEFAULT FALSE,

                                                is_active BOOLEAN DEFAULT TRUE,
                                                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                                                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

                                                FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
                                                FOREIGN KEY (candidate_id) REFERENCES users(id) ON DELETE CASCADE,
                                                FOREIGN KEY (resume_id) REFERENCES resumes(id) ON DELETE SET NULL,

                                                UNIQUE KEY unique_application (job_id, candidate_id),
                                                INDEX idx_app_status (status),
                                                INDEX idx_app_score (match_score),
                                                INDEX idx_app_date (applied_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Insert sample data
-- Passwords are: admin123, hr123, candidate123 (you'll need to generate proper bcrypt hashes)
INSERT INTO users (username, email, password, first_name, last_name, phone, role, skills, experience_years, education, is_active) VALUES
                                                                                                                                      ('admin', 'admin@recruitment.com', '$2a$10$YourHashedPasswordHere', 'System', 'Administrator', '9999999999', 'ADMIN', NULL, NULL, NULL, TRUE),
                                                                                                                                      ('hr1', 'hr@recruitment.com', '$2a$10$YourHashedPasswordHere', 'HR', 'Manager', '8888888888', 'HR', 'Recruitment, Interviewing, HR Management', 5, 'MBA HR', TRUE),
                                                                                                                                      ('john_doe', 'john@example.com', '$2a$10$YourHashedPasswordHere', 'John', 'Doe', '7777777777', 'CANDIDATE', 'Java, Spring Boot, SQL, Machine Learning', 3, 'B.Tech Computer Science', TRUE),
                                                                                                                                      ('jane_smith', 'jane@example.com', '$2a$10$YourHashedPasswordHere', 'Jane', 'Smith', '6666666666', 'CANDIDATE', 'Python, Django, React, Data Science', 2, 'M.Sc Data Science', TRUE),
                                                                                                                                      ('bob_johnson', 'bob@example.com', '$2a$10$YourHashedPasswordHere', 'Bob', 'Johnson', '5555555555', 'CANDIDATE', 'JavaScript, Node.js, AWS, Docker', 4, 'B.E Computer Engineering', TRUE);

-- Skill Match Results table
CREATE TABLE IF NOT EXISTS skill_match_results (
                                                   id BIGINT AUTO_INCREMENT PRIMARY KEY,
                                                   job_id BIGINT NOT NULL,
                                                   candidate_id BIGINT NOT NULL,
                                                   resume_id BIGINT,

    -- Match metadata
                                                   match_date DATETIME NOT NULL,

    -- Overall match scores
                                                   overall_score DOUBLE,
                                                   skills_score DOUBLE,
                                                   experience_score DOUBLE,
                                                   education_score DOUBLE,
                                                   personality_score DOUBLE,
                                                   cultural_fit_score DOUBLE,

    -- Extracted structured data
                                                   extracted_skills VARCHAR(2000),
                                                   extracted_experience VARCHAR(1000),
                                                   extracted_education VARCHAR(500),
                                                   extracted_certifications VARCHAR(1000),
                                                   extracted_languages VARCHAR(500),
                                                   extracted_projects VARCHAR(2000),
                                                   raw_extracted_data LONGTEXT,

    -- Skill matching details
                                                   matched_skills VARCHAR(2000),
                                                   missing_skills VARCHAR(2000),
                                                   partial_skills VARCHAR(2000),

    -- AI processing metadata
                                                   ai_processed BOOLEAN DEFAULT FALSE,
                                                   ai_confidence DOUBLE,
                                                   ai_model_version VARCHAR(50),
                                                   ai_processing_time_ms BIGINT,
                                                   ai_raw_response LONGTEXT,

    -- Ranking information
                                                   rank_position INT,
                                                   total_candidates_ranked INT,
                                                   percentile DOUBLE,

    -- Status flags
                                                   is_active BOOLEAN DEFAULT TRUE,
                                                   is_latest BOOLEAN DEFAULT TRUE,
                                                   recalculation_count INT DEFAULT 0,
                                                   last_recalculated_date DATETIME,

    -- Audit fields
                                                   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                                   updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

                                                   FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
                                                   FOREIGN KEY (candidate_id) REFERENCES users(id) ON DELETE CASCADE,
                                                   FOREIGN KEY (resume_id) REFERENCES resumes(id) ON DELETE SET NULL,

                                                   INDEX idx_match_job (job_id),
                                                   INDEX idx_match_candidate (candidate_id),
                                                   INDEX idx_match_score (overall_score),
                                                   INDEX idx_match_date (match_date),
                                                   INDEX idx_match_active (is_active),
                                                   INDEX idx_match_latest (is_latest)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Candidate Rankings table
CREATE TABLE IF NOT EXISTS candidate_rankings (
                                                  id BIGINT AUTO_INCREMENT PRIMARY KEY,
                                                  job_id BIGINT NOT NULL,
                                                  candidate_id BIGINT NOT NULL,
                                                  skill_match_result_id BIGINT,
                                                  job_application_id BIGINT,

    -- Ranking information
                                                  rank_position INT,
                                                  previous_rank_position INT,
                                                  rank_change INT,
                                                  total_candidates_ranked INT,
                                                  percentile DOUBLE,

    -- AI-generated ranking score
                                                  ranking_score DOUBLE,
                                                  weighted_skills_score DOUBLE,
                                                  weighted_experience_score DOUBLE,
                                                  weighted_education_score DOUBLE,
                                                  weighted_personality_score DOUBLE,
                                                  weighted_cultural_fit_score DOUBLE,

    -- Weights used
                                                  skills_weight DOUBLE,
                                                  experience_weight DOUBLE,
                                                  education_weight DOUBLE,
                                                  personality_weight DOUBLE,
                                                  cultural_fit_weight DOUBLE,

    -- Ranking metadata
                                                  ranking_date DATETIME,
                                                  ranking_criteria_version VARCHAR(50),
                                                  is_current_ranking BOOLEAN DEFAULT TRUE,
                                                  notes VARCHAR(500),

    -- Shortlist status
                                                  is_shortlisted BOOLEAN DEFAULT FALSE,
                                                  shortlist_date DATETIME,
                                                  shortlist_notes VARCHAR(500),

    -- Interview status
                                                  interview_scheduled BOOLEAN DEFAULT FALSE,
                                                  interview_date DATETIME,
                                                  interview_feedback VARCHAR(1000),

    -- Hiring status
                                                  hiring_status ENUM('NOT_REVIEWED', 'UNDER_REVIEW', 'SHORTLISTED', 'INTERVIEWED',
                                                      'OFFER_EXTENDED', 'OFFER_ACCEPTED', 'OFFER_DECLINED', 'REJECTED', 'HIRED')
                                                      DEFAULT 'NOT_REVIEWED',
                                                  hiring_decision_date DATETIME,

    -- Audit fields
                                                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                                  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

                                                  FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
                                                  FOREIGN KEY (candidate_id) REFERENCES users(id) ON DELETE CASCADE,
                                                  FOREIGN KEY (skill_match_result_id) REFERENCES skill_match_results(id) ON DELETE SET NULL,
                                                  FOREIGN KEY (job_application_id) REFERENCES job_applications(id) ON DELETE SET NULL,

                                                  INDEX idx_ranking_job (job_id),
                                                  INDEX idx_ranking_candidate (candidate_id),
                                                  INDEX idx_ranking_score (ranking_score),
                                                  INDEX idx_ranking_position (rank_position),
                                                  INDEX idx_ranking_current (is_current_ranking),
                                                  INDEX idx_ranking_shortlisted (is_shortlisted),
                                                  INDEX idx_ranking_hiring_status (hiring_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
-- Add new columns to job_applications table
ALTER TABLE job_applications
-- Interview related
    ADD COLUMN interview_scheduled BOOLEAN DEFAULT FALSE,
    ADD COLUMN interview_date DATETIME,
    ADD COLUMN interview_type VARCHAR(20),
    ADD COLUMN interview_location VARCHAR(255),
    ADD COLUMN interview_link VARCHAR(500),
    ADD COLUMN interviewer_name VARCHAR(100),
    ADD COLUMN interviewer_email VARCHAR(100),
    ADD COLUMN interview_feedback TEXT,
    ADD COLUMN interview_rating INT,

-- Shortlist related
    ADD COLUMN shortlisted_date DATETIME,
    ADD COLUMN shortlisted_by VARCHAR(100),

-- Review related
    ADD COLUMN reviewed_date DATETIME,
    ADD COLUMN reviewed_by VARCHAR(100),
    ADD COLUMN review_rating INT,

-- Application details
    ADD COLUMN cover_letter TEXT,
    ADD COLUMN expected_salary DOUBLE,
    ADD COLUMN notice_period VARCHAR(50),
    ADD COLUMN current_company VARCHAR(100),
    ADD COLUMN current_position VARCHAR(100),
    ADD COLUMN location_preference VARCHAR(100),
    ADD COLUMN willing_to_relocate BOOLEAN DEFAULT FALSE,
    ADD COLUMN has_work_authorization BOOLEAN DEFAULT FALSE,
    ADD COLUMN work_authorization_country VARCHAR(100),

-- Notification preferences
    ADD COLUMN email_notifications BOOLEAN DEFAULT TRUE,
    ADD COLUMN sms_notifications BOOLEAN DEFAULT FALSE,

-- Hiring decision
    ADD COLUMN hiring_decision_date DATETIME,
    ADD COLUMN hiring_decision_by VARCHAR(100),
    ADD COLUMN offer_amount DOUBLE,
    ADD COLUMN offer_accepted_date DATETIME,
    ADD COLUMN joining_date DATETIME,
    ADD COLUMN rejection_reason VARCHAR(1000),
    ADD COLUMN withdrawal_reason VARCHAR(1000),

-- Archive related
    ADD COLUMN is_archived BOOLEAN DEFAULT FALSE,
    ADD COLUMN archive_date DATETIME;
USE aiml_project;

-- Add just the missing column first
ALTER TABLE job_applications ADD COLUMN interview_feedback TEXT;
