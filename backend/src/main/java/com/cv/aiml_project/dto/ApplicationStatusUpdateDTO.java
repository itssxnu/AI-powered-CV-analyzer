package com.cv.aiml_project.dto;

import com.cv.aiml_project.entity.ApplicationStatus;
import jakarta.validation.constraints.NotNull;

public class ApplicationStatusUpdateDTO {

    @NotNull
    private Long applicationId;

    @NotNull
    private ApplicationStatus status;

    private String notes;

    private String feedback;

    private Boolean notifyCandidate;

    // Getters and Setters
    public Long getApplicationId() { return applicationId; }
    public void setApplicationId(Long applicationId) { this.applicationId = applicationId; }

    public ApplicationStatus getStatus() { return status; }
    public void setStatus(ApplicationStatus status) { this.status = status; }

    public String getNotes() { return notes; }
    public void setNotes(String notes) { this.notes = notes; }

    public String getFeedback() { return feedback; }
    public void setFeedback(String feedback) { this.feedback = feedback; }

    public Boolean getNotifyCandidate() { return notifyCandidate; }
    public void setNotifyCandidate(Boolean notifyCandidate) { this.notifyCandidate = notifyCandidate; }
}
