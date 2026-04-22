package com.cv.aiml_project.entity;

public enum JobType {
    FULL_TIME("Full Time"),
    PART_TIME("Part Time"),
    CONTRACT("Contract"),
    INTERNSHIP("Internship"),
    REMOTE("Remote"),
    HYBRID("Hybrid");

    private final String displayName;

    JobType(String displayName) {
        this.displayName = displayName;
    }

    public String getDisplayName() {
        return displayName;
    }
}
