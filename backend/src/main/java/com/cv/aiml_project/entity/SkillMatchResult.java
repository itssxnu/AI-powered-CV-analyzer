package com.cv.aiml_project.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "skill_match_results")
public class SkillMatchResult {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "job_id", nullable = false)
    private Job job;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "candidate_id", nullable = false)
    private User candidate;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "resume_id")
    private Resume resume;

    @Column(name = "match_date", nullable = false)
    private LocalDateTime matchDate;

    // Overall match scores
    @Column(name = "overall_score")
    private Double overallScore;

    @Column(name = "skills_score")
    private Double skillsScore;

    @Column(name = "experience_score")
    private Double experienceScore;

    @Column(name = "education_score")
    private Double educationScore;

    @Column(name = "personality_score")
    private Double personalityScore;

    @Column(name = "cultural_fit_score")
    private Double culturalFitScore;

    // Extracted structured data
    @Column(name = "extracted_skills", length = 2000)
    private String extractedSkills;

    @Column(name = "extracted_experience", length = 1000)
    private String extractedExperience;

    @Column(name = "extracted_education", length = 500)
    private String extractedEducation;

    @Column(name = "extracted_certifications", length = 1000)
    private String extractedCertifications;

    @Column(name = "extracted_languages", length = 500)
    private String extractedLanguages;

    @Column(name = "extracted_projects", length = 2000)
    private String extractedProjects;

    @Column(name = "raw_extracted_data", columnDefinition = "LONGTEXT")
    private String rawExtractedData;

    // Skill matching details
    @Column(name = "matched_skills", length = 2000)
    private String matchedSkills;

    @Column(name = "missing_skills", length = 2000)
    private String missingSkills;

    @Column(name = "partial_skills", length = 2000)
    private String partialSkills;

    // AI processing metadata
    @Column(name = "ai_processed")
    private boolean aiProcessed = false;

    @Column(name = "ai_confidence")
    private Double aiConfidence;

    @Column(name = "ai_model_version")
    private String aiModelVersion;

    @Column(name = "ai_processing_time_ms")
    private Long aiProcessingTimeMs;

    @Column(name = "ai_raw_response", columnDefinition = "LONGTEXT")
    private String aiRawResponse;

    // Ranking information
    @Column(name = "rank_position")
    private Integer rankPosition;

    @Column(name = "total_candidates_ranked")
    private Integer totalCandidatesRanked;

    @Column(name = "percentile")
    private Double percentile;

    // Status flags
    @Column(name = "is_active")
    private boolean isActive = true;

    @Column(name = "is_latest")
    private boolean isLatest = true;

    @Column(name = "recalculation_count")
    private Integer recalculationCount = 0;

    @Column(name = "last_recalculated_date")
    private LocalDateTime lastRecalculatedDate;

    // Audit fields
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
        if (matchDate == null) {
            matchDate = LocalDateTime.now();
        }
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }

    // Constructors
    public SkillMatchResult() {}

    public SkillMatchResult(Job job, User candidate) {
        this.job = job;
        this.candidate = candidate;
        this.matchDate = LocalDateTime.now();
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public Job getJob() { return job; }
    public void setJob(Job job) { this.job = job; }

    public User getCandidate() { return candidate; }
    public void setCandidate(User candidate) { this.candidate = candidate; }

    public Resume getResume() { return resume; }
    public void setResume(Resume resume) { this.resume = resume; }

    public LocalDateTime getMatchDate() { return matchDate; }
    public void setMatchDate(LocalDateTime matchDate) { this.matchDate = matchDate; }

    public Double getOverallScore() { return overallScore; }
    public void setOverallScore(Double overallScore) { this.overallScore = overallScore; }

    public Double getSkillsScore() { return skillsScore; }
    public void setSkillsScore(Double skillsScore) { this.skillsScore = skillsScore; }

    public Double getExperienceScore() { return experienceScore; }
    public void setExperienceScore(Double experienceScore) { this.experienceScore = experienceScore; }

    public Double getEducationScore() { return educationScore; }
    public void setEducationScore(Double educationScore) { this.educationScore = educationScore; }

    public Double getPersonalityScore() { return personalityScore; }
    public void setPersonalityScore(Double personalityScore) { this.personalityScore = personalityScore; }

    public Double getCulturalFitScore() { return culturalFitScore; }
    public void setCulturalFitScore(Double culturalFitScore) { this.culturalFitScore = culturalFitScore; }

    public String getExtractedSkills() { return extractedSkills; }
    public void setExtractedSkills(String extractedSkills) { this.extractedSkills = extractedSkills; }

    public String getExtractedExperience() { return extractedExperience; }
    public void setExtractedExperience(String extractedExperience) { this.extractedExperience = extractedExperience; }

    public String getExtractedEducation() { return extractedEducation; }
    public void setExtractedEducation(String extractedEducation) { this.extractedEducation = extractedEducation; }

    public String getExtractedCertifications() { return extractedCertifications; }
    public void setExtractedCertifications(String extractedCertifications) { this.extractedCertifications = extractedCertifications; }

    public String getExtractedLanguages() { return extractedLanguages; }
    public void setExtractedLanguages(String extractedLanguages) { this.extractedLanguages = extractedLanguages; }

    public String getExtractedProjects() { return extractedProjects; }
    public void setExtractedProjects(String extractedProjects) { this.extractedProjects = extractedProjects; }

    public String getRawExtractedData() { return rawExtractedData; }
    public void setRawExtractedData(String rawExtractedData) { this.rawExtractedData = rawExtractedData; }

    public String getMatchedSkills() { return matchedSkills; }
    public void setMatchedSkills(String matchedSkills) { this.matchedSkills = matchedSkills; }

    public String getMissingSkills() { return missingSkills; }
    public void setMissingSkills(String missingSkills) { this.missingSkills = missingSkills; }

    public String getPartialSkills() { return partialSkills; }
    public void setPartialSkills(String partialSkills) { this.partialSkills = partialSkills; }

    public boolean isAiProcessed() { return aiProcessed; }
    public void setAiProcessed(boolean aiProcessed) { this.aiProcessed = aiProcessed; }

    public Double getAiConfidence() { return aiConfidence; }
    public void setAiConfidence(Double aiConfidence) { this.aiConfidence = aiConfidence; }

    public String getAiModelVersion() { return aiModelVersion; }
    public void setAiModelVersion(String aiModelVersion) { this.aiModelVersion = aiModelVersion; }

    public Long getAiProcessingTimeMs() { return aiProcessingTimeMs; }
    public void setAiProcessingTimeMs(Long aiProcessingTimeMs) { this.aiProcessingTimeMs = aiProcessingTimeMs; }

    public String getAiRawResponse() { return aiRawResponse; }
    public void setAiRawResponse(String aiRawResponse) { this.aiRawResponse = aiRawResponse; }

    public Integer getRankPosition() { return rankPosition; }
    public void setRankPosition(Integer rankPosition) { this.rankPosition = rankPosition; }

    public Integer getTotalCandidatesRanked() { return totalCandidatesRanked; }
    public void setTotalCandidatesRanked(Integer totalCandidatesRanked) { this.totalCandidatesRanked = totalCandidatesRanked; }

    public Double getPercentile() { return percentile; }
    public void setPercentile(Double percentile) { this.percentile = percentile; }

    public boolean isActive() { return isActive; }
    public void setActive(boolean active) { isActive = active; }

    public boolean isLatest() { return isLatest; }
    public void setLatest(boolean latest) { isLatest = latest; }

    public Integer getRecalculationCount() { return recalculationCount; }
    public void setRecalculationCount(Integer recalculationCount) { this.recalculationCount = recalculationCount; }

    public LocalDateTime getLastRecalculatedDate() { return lastRecalculatedDate; }
    public void setLastRecalculatedDate(LocalDateTime lastRecalculatedDate) { this.lastRecalculatedDate = lastRecalculatedDate; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }

    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }

    // Helper methods
    public String getScoreColorClass() {
        if (overallScore == null) return "bg-secondary";
        if (overallScore >= 80) return "bg-success";
        if (overallScore >= 60) return "bg-warning";
        if (overallScore >= 40) return "bg-danger";
        return "bg-dark";
    }

    public String getMatchLevel() {
        if (overallScore == null) return "Not Analyzed";
        if (overallScore >= 80) return "Excellent Match";
        if (overallScore >= 60) return "Good Match";
        if (overallScore >= 40) return "Average Match";
        return "Poor Match";
    }

    public int getMatchedSkillsCount() {
        if (matchedSkills == null) return 0;
        return matchedSkills.split(",").length;
    }

    public int getMissingSkillsCount() {
        if (missingSkills == null) return 0;
        return missingSkills.split(",").length;
    }

    public int getPartialSkillsCount() {
        if (partialSkills == null) return 0;
        return partialSkills.split(",").length;
    }
}
