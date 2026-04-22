package com.cv.aiml_project.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "candidate_rankings")
public class CandidateRanking {

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
    @JoinColumn(name = "skill_match_result_id")
    private SkillMatchResult skillMatchResult;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "job_application_id")
    private JobApplication jobApplication;

    // Ranking information
    @Column(name = "rank_position")
    private Integer rankPosition;

    @Column(name = "previous_rank_position")
    private Integer previousRankPosition;

    @Column(name = "rank_change")
    private Integer rankChange;

    @Column(name = "total_candidates_ranked")
    private Integer totalCandidatesRanked;

    @Column(name = "percentile")
    private Double percentile;

    // AI-generated ranking score (composite)
    @Column(name = "ranking_score")
    private Double rankingScore;

    // Individual component scores with weights applied
    @Column(name = "weighted_skills_score")
    private Double weightedSkillsScore;

    @Column(name = "weighted_experience_score")
    private Double weightedExperienceScore;

    @Column(name = "weighted_education_score")
    private Double weightedEducationScore;

    @Column(name = "weighted_personality_score")
    private Double weightedPersonalityScore;

    @Column(name = "weighted_cultural_fit_score")
    private Double weightedCulturalFitScore;

    // Weights used for this ranking (store the configuration)
    @Column(name = "skills_weight")
    private Double skillsWeight;

    @Column(name = "experience_weight")
    private Double experienceWeight;

    @Column(name = "education_weight")
    private Double educationWeight;

    @Column(name = "personality_weight")
    private Double personalityWeight;

    @Column(name = "cultural_fit_weight")
    private Double culturalFitWeight;

    // Ranking metadata
    @Column(name = "ranking_date")
    private LocalDateTime rankingDate;

    @Column(name = "ranking_criteria_version")
    private String rankingCriteriaVersion;

    @Column(name = "is_current_ranking")
    private boolean isCurrentRanking = true;

    @Column(name = "notes", length = 500)
    private String notes;

    // Shortlist status
    @Column(name = "is_shortlisted")
    private boolean isShortlisted = false;

    @Column(name = "shortlist_date")
    private LocalDateTime shortlistDate;

    @Column(name = "shortlist_notes", length = 500)
    private String shortlistNotes;

    // Interview status
    @Column(name = "interview_scheduled")
    private boolean interviewScheduled = false;

    @Column(name = "interview_date")
    private LocalDateTime interviewDate;

    @Column(name = "interview_feedback", length = 1000)
    private String interviewFeedback;

    // Hiring status
    @Column(name = "hiring_status")
    @Enumerated(EnumType.STRING)
    private HiringStatus hiringStatus = HiringStatus.NOT_REVIEWED;

    @Column(name = "hiring_decision_date")
    private LocalDateTime hiringDecisionDate;

    // Audit fields
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
        if (rankingDate == null) {
            rankingDate = LocalDateTime.now();
        }
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }

    // Constructors
    public CandidateRanking() {}

    public CandidateRanking(Job job, User candidate) {
        this.job = job;
        this.candidate = candidate;
        this.rankingDate = LocalDateTime.now();
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public Job getJob() { return job; }
    public void setJob(Job job) { this.job = job; }

    public User getCandidate() { return candidate; }
    public void setCandidate(User candidate) { this.candidate = candidate; }

    public SkillMatchResult getSkillMatchResult() { return skillMatchResult; }
    public void setSkillMatchResult(SkillMatchResult skillMatchResult) { this.skillMatchResult = skillMatchResult; }

    public JobApplication getJobApplication() { return jobApplication; }
    public void setJobApplication(JobApplication jobApplication) { this.jobApplication = jobApplication; }

    public Integer getRankPosition() { return rankPosition; }
    public void setRankPosition(Integer rankPosition) { this.rankPosition = rankPosition; }

    public Integer getPreviousRankPosition() { return previousRankPosition; }
    public void setPreviousRankPosition(Integer previousRankPosition) { this.previousRankPosition = previousRankPosition; }

    public Integer getRankChange() { return rankChange; }
    public void setRankChange(Integer rankChange) { this.rankChange = rankChange; }

    public Integer getTotalCandidatesRanked() { return totalCandidatesRanked; }
    public void setTotalCandidatesRanked(Integer totalCandidatesRanked) { this.totalCandidatesRanked = totalCandidatesRanked; }

    public Double getPercentile() { return percentile; }
    public void setPercentile(Double percentile) { this.percentile = percentile; }

    public Double getRankingScore() { return rankingScore; }
    public void setRankingScore(Double rankingScore) { this.rankingScore = rankingScore; }

    public Double getWeightedSkillsScore() { return weightedSkillsScore; }
    public void setWeightedSkillsScore(Double weightedSkillsScore) { this.weightedSkillsScore = weightedSkillsScore; }

    public Double getWeightedExperienceScore() { return weightedExperienceScore; }
    public void setWeightedExperienceScore(Double weightedExperienceScore) { this.weightedExperienceScore = weightedExperienceScore; }

    public Double getWeightedEducationScore() { return weightedEducationScore; }
    public void setWeightedEducationScore(Double weightedEducationScore) { this.weightedEducationScore = weightedEducationScore; }

    public Double getWeightedPersonalityScore() { return weightedPersonalityScore; }
    public void setWeightedPersonalityScore(Double weightedPersonalityScore) { this.weightedPersonalityScore = weightedPersonalityScore; }

    public Double getWeightedCulturalFitScore() { return weightedCulturalFitScore; }
    public void setWeightedCulturalFitScore(Double weightedCulturalFitScore) { this.weightedCulturalFitScore = weightedCulturalFitScore; }

    public Double getSkillsWeight() { return skillsWeight; }
    public void setSkillsWeight(Double skillsWeight) { this.skillsWeight = skillsWeight; }

    public Double getExperienceWeight() { return experienceWeight; }
    public void setExperienceWeight(Double experienceWeight) { this.experienceWeight = experienceWeight; }

    public Double getEducationWeight() { return educationWeight; }
    public void setEducationWeight(Double educationWeight) { this.educationWeight = educationWeight; }

    public Double getPersonalityWeight() { return personalityWeight; }
    public void setPersonalityWeight(Double personalityWeight) { this.personalityWeight = personalityWeight; }

    public Double getCulturalFitWeight() { return culturalFitWeight; }
    public void setCulturalFitWeight(Double culturalFitWeight) { this.culturalFitWeight = culturalFitWeight; }

    public LocalDateTime getRankingDate() { return rankingDate; }
    public void setRankingDate(LocalDateTime rankingDate) { this.rankingDate = rankingDate; }

    public String getRankingCriteriaVersion() { return rankingCriteriaVersion; }
    public void setRankingCriteriaVersion(String rankingCriteriaVersion) { this.rankingCriteriaVersion = rankingCriteriaVersion; }

    public boolean isCurrentRanking() { return isCurrentRanking; }
    public void setCurrentRanking(boolean currentRanking) { isCurrentRanking = currentRanking; }

    public String getNotes() { return notes; }
    public void setNotes(String notes) { this.notes = notes; }

    public boolean isShortlisted() { return isShortlisted; }
    public void setShortlisted(boolean shortlisted) { isShortlisted = shortlisted; }

    public LocalDateTime getShortlistDate() { return shortlistDate; }
    public void setShortlistDate(LocalDateTime shortlistDate) { this.shortlistDate = shortlistDate; }

    public String getShortlistNotes() { return shortlistNotes; }
    public void setShortlistNotes(String shortlistNotes) { this.shortlistNotes = shortlistNotes; }

    public boolean isInterviewScheduled() { return interviewScheduled; }
    public void setInterviewScheduled(boolean interviewScheduled) { this.interviewScheduled = interviewScheduled; }

    public LocalDateTime getInterviewDate() { return interviewDate; }
    public void setInterviewDate(LocalDateTime interviewDate) { this.interviewDate = interviewDate; }

    public String getInterviewFeedback() { return interviewFeedback; }
    public void setInterviewFeedback(String interviewFeedback) { this.interviewFeedback = interviewFeedback; }

    public HiringStatus getHiringStatus() { return hiringStatus; }
    public void setHiringStatus(HiringStatus hiringStatus) { this.hiringStatus = hiringStatus; }

    public LocalDateTime getHiringDecisionDate() { return hiringDecisionDate; }
    public void setHiringDecisionDate(LocalDateTime hiringDecisionDate) { this.hiringDecisionDate = hiringDecisionDate; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }

    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }

    // Helper methods
    public String getRankChangeIcon() {
        if (rankChange == null || rankChange == 0) return "bi-dash-circle";
        if (rankChange > 0) return "bi-arrow-up-circle text-success";
        return "bi-arrow-down-circle text-danger";
    }

    public String getRankChangeText() {
        if (rankChange == null || rankChange == 0) return "No change";
        if (rankChange > 0) return "+" + rankChange + " positions";
        return rankChange + " positions";
    }

    public String getHiringStatusBadgeClass() {
        if (hiringStatus == null) return "bg-secondary";
        switch (hiringStatus) {
            case NOT_REVIEWED: return "bg-secondary";
            case UNDER_REVIEW: return "bg-info";
            case SHORTLISTED: return "bg-success";
            case INTERVIEWED: return "bg-primary";
            case OFFER_EXTENDED: return "bg-warning";
            case OFFER_ACCEPTED: return "bg-success";
            case OFFER_DECLINED: return "bg-danger";
            case REJECTED: return "bg-danger";
            case HIRED: return "bg-success";
            default: return "bg-secondary";
        }
    }

    public boolean isTopRanked() {
        return rankPosition != null && rankPosition <= 5;
    }

    public double getNormalizedScore() {
        if (rankingScore == null) return 0.0;
        return rankingScore / 100.0;
    }
}
