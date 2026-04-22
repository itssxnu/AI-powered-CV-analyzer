package com.cv.aiml_project.dto;

public class ApplicationNoteDTO {

    private Long applicationId;
    private String note;
    private String noteType; // GENERAL, HR, INTERVIEW, FEEDBACK
    private Boolean isPrivate;

    // Getters and Setters
    public Long getApplicationId() { return applicationId; }
    public void setApplicationId(Long applicationId) { this.applicationId = applicationId; }

    public String getNote() { return note; }
    public void setNote(String note) { this.note = note; }

    public String getNoteType() { return noteType; }
    public void setNoteType(String noteType) { this.noteType = noteType; }

    public Boolean getIsPrivate() { return isPrivate; }
    public void setIsPrivate(Boolean isPrivate) { this.isPrivate = isPrivate; }
}
