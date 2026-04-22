package com.cv.aiml_project.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "job_applications")
public class JobApplication {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "job_id", nullable = false)
    private Job job;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "candidate_id", nullable = false)
    private User candidate;

    @Column(nullable = false)
    private LocalDateTime appliedDate;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private ApplicationStatus status = ApplicationStatus.PENDING;

    private LocalDateTime statusUpdatedDate;

    @Column(length = 5000) // Increased length for notes
    private String hrNotes;

    @Column(name = "resume_path")
    private String resumePath;

    @Column(name = "resume_original_name")
    private String resumeOriginalName;

    @Column(name = "resume_content_type")
    private String resumeContentType;

    @Column(name = "resume_file_size")
    private Long resumeFileSize;

    @Column(name = "resume_upload_date")
    private LocalDateTime resumeUploadDate;

    @Column(name = "resume_text_content", columnDefinition = "TEXT")
    private String resumeTextContent;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "resume_id")
    private Resume resume;

    // AI Matching scores
    private Double matchScore;
    private Double skillsMatchScore;
    private Double experienceMatchScore;
    private Double educationMatchScore;

    // ==================== NEW FIELDS ====================

    // Interview related fields
    @Column(name = "interview_scheduled")
    private Boolean interviewScheduled = false;

    @Column(name = "interview_date")
    private LocalDateTime interviewDate;

    @Column(name = "interview_type")
    private String interviewType; // ONLINE, PHONE, IN_PERSON

    @Column(name = "interview_location")
    private String interviewLocation;

    @Column(name = "interview_link")
    private String interviewLink;

    @Column(name = "interviewer_name")
    private String interviewerName;

    @Column(name = "interviewer_email")
    private String interviewerEmail;

    @Column(name = "interview_feedback", length = 5000)
    private String interviewFeedback;

    @Column(name = "interview_rating")
    private Integer interviewRating; // 1-5 rating

    // AI Interview Q&A
    @Column(name = "interview_questions_json", columnDefinition = "TEXT")
    private String interviewQuestionsJson;

    @Column(name = "interview_answers_json", columnDefinition = "TEXT")
    private String interviewAnswersJson;

    @Column(name = "questions_generated_at")
    private LocalDateTime questionsGeneratedAt;

    @Column(name = "answers_submitted_at")
    private LocalDateTime answersSubmittedAt;

    // Shortlist information
    @Column(name = "shortlisted_date")
    private LocalDateTime shortlistedDate;

    @Column(name = "shortlisted_by")
    private String shortlistedBy;

    // Review information
    @Column(name = "reviewed_date")
    private LocalDateTime reviewedDate;

    @Column(name = "reviewed_by")
    private String reviewedBy;

    @Column(name = "review_rating")
    private Integer reviewRating; // 1-5 rating

    // Additional application fields
    @Column(name = "cover_letter")
    private String coverLetter;

    @Column(name = "expected_salary")
    private Double expectedSalary;

    @Column(name = "notice_period")
    private String noticePeriod; // e.g., "30 days", "Immediate"

    @Column(name = "current_company")
    private String currentCompany;

    @Column(name = "current_position")
    private String currentPosition;

    @Column(name = "location_preference")
    private String locationPreference;

    @Column(name = "willing_to_relocate")
    private Boolean willingToRelocate = false;

    @Column(name = "has_work_authorization")
    private Boolean hasWorkAuthorization = false;

    @Column(name = "work_authorization_country")
    private String workAuthorizationCountry;

    // Notification preferences
    @Column(name = "email_notifications")
    private Boolean emailNotifications = true;

    @Column(name = "sms_notifications")
    private Boolean smsNotifications = false;

    // Hiring decision fields
    @Column(name = "hiring_decision_date")
    private LocalDateTime hiringDecisionDate;

    @Column(name = "hiring_decision_by")
    private String hiringDecisionBy;

    @Column(name = "offer_amount")
    private Double offerAmount;

    @Column(name = "offer_accepted_date")
    private LocalDateTime offerAcceptedDate;

    @Column(name = "joining_date")
    private LocalDateTime joiningDate;

    @Column(name = "rejection_reason", length = 1000)
    private String rejectionReason;

    @Column(name = "withdrawal_reason", length = 1000)
    private String withdrawalReason;

    // Status flags
    @Column(name = "is_active", columnDefinition = "boolean default true")
    private Boolean isActive = true;

    @Column(name = "is_archived")
    private Boolean isArchived = false;

    @Column(name = "archive_date")
    private LocalDateTime archiveDate;

    // Audit fields
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
        appliedDate = LocalDateTime.now();
        statusUpdatedDate = LocalDateTime.now();
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
        statusUpdatedDate = LocalDateTime.now();
    }

    // Constructors
    public JobApplication() {}

    public JobApplication(Job job, User candidate) {
        this.job = job;
        this.candidate = candidate;
        this.status = ApplicationStatus.PENDING;
        this.appliedDate = LocalDateTime.now();
        this.statusUpdatedDate = LocalDateTime.now();
        this.emailNotifications = true;
        this.smsNotifications = false;
        this.willingToRelocate = false;
        this.hasWorkAuthorization = false;
    }

    // ==================== GETTERS AND SETTERS ====================

    // Original getters and setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public Job getJob() { return job; }
    public void setJob(Job job) { this.job = job; }

    public User getCandidate() { return candidate; }
    public void setCandidate(User candidate) { this.candidate = candidate; }

    public LocalDateTime getAppliedDate() { return appliedDate; }
    public void setAppliedDate(LocalDateTime appliedDate) { this.appliedDate = appliedDate; }

    public ApplicationStatus getStatus() { return status; }
    public void setStatus(ApplicationStatus status) { this.status = status; }

    public LocalDateTime getStatusUpdatedDate() { return statusUpdatedDate; }
    public void setStatusUpdatedDate(LocalDateTime statusUpdatedDate) { this.statusUpdatedDate = statusUpdatedDate; }

    public String getHrNotes() { return hrNotes; }
    public void setHrNotes(String hrNotes) { this.hrNotes = hrNotes; }

    public String getResumePath() { return resumePath; }
    public void setResumePath(String resumePath) { this.resumePath = resumePath; }

    public String getResumeOriginalName() { return resumeOriginalName; }
    public void setResumeOriginalName(String resumeOriginalName) { this.resumeOriginalName = resumeOriginalName; }

    public String getResumeContentType() { return resumeContentType; }
    public void setResumeContentType(String resumeContentType) { this.resumeContentType = resumeContentType; }

    public Long getResumeFileSize() { return resumeFileSize; }
    public void setResumeFileSize(Long resumeFileSize) { this.resumeFileSize = resumeFileSize; }

    public LocalDateTime getResumeUploadDate() { return resumeUploadDate; }
    public void setResumeUploadDate(LocalDateTime resumeUploadDate) { this.resumeUploadDate = resumeUploadDate; }

    public String getResumeTextContent() { return resumeTextContent; }
    public void setResumeTextContent(String resumeTextContent) { this.resumeTextContent = resumeTextContent; }

    public Resume getResume() { return resume; }
    public void setResume(Resume resume) { this.resume = resume; }

    public Long getResumeId() { return resume != null ? resume.getId() : null; }

    public Double getMatchScore() { return matchScore; }
    public void setMatchScore(Double matchScore) { this.matchScore = matchScore; }

    public Double getSkillsMatchScore() { return skillsMatchScore; }
    public void setSkillsMatchScore(Double skillsMatchScore) { this.skillsMatchScore = skillsMatchScore; }

    public Double getExperienceMatchScore() { return experienceMatchScore; }
    public void setExperienceMatchScore(Double experienceMatchScore) { this.experienceMatchScore = experienceMatchScore; }

    public Double getEducationMatchScore() { return educationMatchScore; }
    public void setEducationMatchScore(Double educationMatchScore) { this.educationMatchScore = educationMatchScore; }

    public Boolean isActive() { return isActive; }
    public void setActive(Boolean active) { isActive = active; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }

    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }

    // ==================== NEW GETTERS AND SETTERS ====================

    // Interview related
    public boolean isInterviewScheduled() {
        return interviewScheduled != null ? interviewScheduled : false;
    }
    public void setInterviewScheduled(Boolean interviewScheduled) {
        this.interviewScheduled = interviewScheduled;
    }
    public LocalDateTime getInterviewDate() { return interviewDate; }
    public void setInterviewDate(LocalDateTime interviewDate) { this.interviewDate = interviewDate; }

    public String getInterviewType() { return interviewType; }
    public void setInterviewType(String interviewType) { this.interviewType = interviewType; }

    public String getInterviewLocation() { return interviewLocation; }
    public void setInterviewLocation(String interviewLocation) { this.interviewLocation = interviewLocation; }

    public String getInterviewLink() { return interviewLink; }
    public void setInterviewLink(String interviewLink) { this.interviewLink = interviewLink; }

    public String getInterviewerName() { return interviewerName; }
    public void setInterviewerName(String interviewerName) { this.interviewerName = interviewerName; }

    public String getInterviewerEmail() { return interviewerEmail; }
    public void setInterviewerEmail(String interviewerEmail) { this.interviewerEmail = interviewerEmail; }

    public String getInterviewFeedback() { return interviewFeedback; }
    public void setInterviewFeedback(String interviewFeedback) { this.interviewFeedback = interviewFeedback; }

    public Integer getInterviewRating() { return interviewRating; }
    public void setInterviewRating(Integer interviewRating) { this.interviewRating = interviewRating; }

    // AI Interview Q&A getters/setters
    public String getInterviewQuestionsJson() { return interviewQuestionsJson; }
    public void setInterviewQuestionsJson(String interviewQuestionsJson) { this.interviewQuestionsJson = interviewQuestionsJson; }

    public String getInterviewAnswersJson() { return interviewAnswersJson; }
    public void setInterviewAnswersJson(String interviewAnswersJson) { this.interviewAnswersJson = interviewAnswersJson; }

    public LocalDateTime getQuestionsGeneratedAt() { return questionsGeneratedAt; }
    public void setQuestionsGeneratedAt(LocalDateTime questionsGeneratedAt) { this.questionsGeneratedAt = questionsGeneratedAt; }

    public LocalDateTime getAnswersSubmittedAt() { return answersSubmittedAt; }
    public void setAnswersSubmittedAt(LocalDateTime answersSubmittedAt) { this.answersSubmittedAt = answersSubmittedAt; }

    public boolean hasInterviewQuestions() { return interviewQuestionsJson != null && !interviewQuestionsJson.isBlank(); }
    public boolean hasInterviewAnswers() { return interviewAnswersJson != null && !interviewAnswersJson.isBlank(); }

    // Shortlist related
    public LocalDateTime getShortlistedDate() { return shortlistedDate; }
    public void setShortlistedDate(LocalDateTime shortlistedDate) { this.shortlistedDate = shortlistedDate; }

    public String getShortlistedBy() { return shortlistedBy; }
    public void setShortlistedBy(String shortlistedBy) { this.shortlistedBy = shortlistedBy; }

    // Review related
    public LocalDateTime getReviewedDate() { return reviewedDate; }
    public void setReviewedDate(LocalDateTime reviewedDate) { this.reviewedDate = reviewedDate; }

    public String getReviewedBy() { return reviewedBy; }
    public void setReviewedBy(String reviewedBy) { this.reviewedBy = reviewedBy; }

    public Integer getReviewRating() { return reviewRating; }
    public void setReviewRating(Integer reviewRating) { this.reviewRating = reviewRating; }

    // Application details
    public String getCoverLetter() { return coverLetter; }
    public void setCoverLetter(String coverLetter) { this.coverLetter = coverLetter; }

    public Double getExpectedSalary() { return expectedSalary; }
    public void setExpectedSalary(Double expectedSalary) { this.expectedSalary = expectedSalary; }

    public String getNoticePeriod() { return noticePeriod; }
    public void setNoticePeriod(String noticePeriod) { this.noticePeriod = noticePeriod; }

    public String getCurrentCompany() { return currentCompany; }
    public void setCurrentCompany(String currentCompany) { this.currentCompany = currentCompany; }

    public String getCurrentPosition() { return currentPosition; }
    public void setCurrentPosition(String currentPosition) { this.currentPosition = currentPosition; }

    public String getLocationPreference() { return locationPreference; }
    public void setLocationPreference(String locationPreference) { this.locationPreference = locationPreference; }

    public Boolean getWillingToRelocate() { return willingToRelocate; }
    public void setWillingToRelocate(Boolean willingToRelocate) { this.willingToRelocate = willingToRelocate; }

    public Boolean getHasWorkAuthorization() { return hasWorkAuthorization; }
    public void setHasWorkAuthorization(Boolean hasWorkAuthorization) { this.hasWorkAuthorization = hasWorkAuthorization; }

    public String getWorkAuthorizationCountry() { return workAuthorizationCountry; }
    public void setWorkAuthorizationCountry(String workAuthorizationCountry) { this.workAuthorizationCountry = workAuthorizationCountry; }

    // Notification preferences
    public Boolean getEmailNotifications() { return emailNotifications; }
    public void setEmailNotifications(Boolean emailNotifications) { this.emailNotifications = emailNotifications; }

    public Boolean getSmsNotifications() { return smsNotifications; }
    public void setSmsNotifications(Boolean smsNotifications) { this.smsNotifications = smsNotifications; }

    // Hiring decision
    public LocalDateTime getHiringDecisionDate() { return hiringDecisionDate; }
    public void setHiringDecisionDate(LocalDateTime hiringDecisionDate) { this.hiringDecisionDate = hiringDecisionDate; }

    public String getHiringDecisionBy() { return hiringDecisionBy; }
    public void setHiringDecisionBy(String hiringDecisionBy) { this.hiringDecisionBy = hiringDecisionBy; }

    public Double getOfferAmount() { return offerAmount; }
    public void setOfferAmount(Double offerAmount) { this.offerAmount = offerAmount; }

    public LocalDateTime getOfferAcceptedDate() { return offerAcceptedDate; }
    public void setOfferAcceptedDate(LocalDateTime offerAcceptedDate) { this.offerAcceptedDate = offerAcceptedDate; }

    public LocalDateTime getJoiningDate() { return joiningDate; }
    public void setJoiningDate(LocalDateTime joiningDate) { this.joiningDate = joiningDate; }

    public String getRejectionReason() { return rejectionReason; }
    public void setRejectionReason(String rejectionReason) { this.rejectionReason = rejectionReason; }

    public String getWithdrawalReason() { return withdrawalReason; }
    public void setWithdrawalReason(String withdrawalReason) { this.withdrawalReason = withdrawalReason; }

    // Archive related
    public Boolean isArchived() { return isArchived; }
    public void setArchived(boolean archived) { isArchived = archived; }

    public LocalDateTime getArchiveDate() { return archiveDate; }
    public void setArchiveDate(LocalDateTime archiveDate) { this.archiveDate = archiveDate; }

    // ==================== HELPER METHODS ====================

    public String getStatusBadgeClass() {
        switch (status) {
            case PENDING: return "bg-warning";
            case SHORTLISTED: return "bg-info";
            case INTERVIEW_SCHEDULED: return "bg-primary";
            case REJECTED: return "bg-danger";
            case HIRED: return "bg-success";
            case WITHDRAWN: return "bg-secondary";
            default: return "bg-secondary";
        }
    }

    public String getMatchScoreColorClass() {
        if (matchScore == null) return "";
        if (matchScore >= 80) return "bg-success";
        if (matchScore >= 60) return "bg-warning";
        return "bg-danger";
    }

    public String getInterviewTypeDisplay() {
        if (interviewType == null) return "Not specified";
        switch (interviewType) {
            case "ONLINE": return "Online Interview";
            case "PHONE": return "Phone Interview";
            case "IN_PERSON": return "In-Person Interview";
            default: return interviewType;
        }
    }

    public boolean isShortlisted() {
        return status == ApplicationStatus.SHORTLISTED || shortlistedDate != null;
    }

    public boolean isRejected() {
        return status == ApplicationStatus.REJECTED;
    }

    public boolean isHired() {
        return status == ApplicationStatus.HIRED;
    }

    public boolean isWithdrawn() {
        return status == ApplicationStatus.WITHDRAWN;
    }

    public String getFormattedExpectedSalary() {
        if (expectedSalary == null) return "Not specified";
        return String.format("$%,.2f", expectedSalary);
    }

    public int getDaysSinceApplied() {
        if (appliedDate == null) return 0;
        return (int) java.time.Duration.between(appliedDate, LocalDateTime.now()).toDays();
    }

    public int getDaysInCurrentStatus() {
        if (statusUpdatedDate == null) return 0;
        return (int) java.time.Duration.between(statusUpdatedDate, LocalDateTime.now()).toDays();
    }

    @Override
    public String toString() {
        return "JobApplication{" +
                "id=" + id +
                ", job=" + (job != null ? job.getTitle() : "null") +
                ", candidate=" + (candidate != null ? candidate.getFullName() : "null") +
                ", status=" + status +
                ", appliedDate=" + appliedDate +
                '}';
    }
    public String getRowClass() {
        if (status == null) return "";
        switch (status) {
            case PENDING: return "status-pending";
            case SHORTLISTED: return "status-shortlisted";
            case INTERVIEW_SCHEDULED: return "status-interview";
            case REJECTED: return "status-rejected";
            case HIRED: return "status-hired";
            default: return "";
        }
    }
    public String getStatusAlertClass() {
        if (status == null) return "alert-secondary";
        switch (status) {
            case PENDING: return "alert-warning";
            case SHORTLISTED: return "alert-info";
            case INTERVIEW_SCHEDULED: return "alert-primary";
            case HIRED: return "alert-success";
            case REJECTED: return "alert-danger";
            case WITHDRAWN: return "alert-secondary";
            default: return "alert-secondary";
        }
    }
}