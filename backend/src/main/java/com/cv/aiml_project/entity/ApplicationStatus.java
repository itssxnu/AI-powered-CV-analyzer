package com.cv.aiml_project.entity;

public enum ApplicationStatus {
    PENDING("Pending Review"),
    SHORTLISTED("Shortlisted"),
    INTERVIEW_SCHEDULED("Interview Scheduled"),
    REJECTED("Rejected"),
    HIRED("Hired"),
    WITHDRAWN("Withdrawn");

    private final String displayName;

    ApplicationStatus(String displayName) {
        this.displayName = displayName;
    }

    public String getDisplayName() {
        return displayName;
    }
}