package com.cv.aiml_project.dto;

import com.cv.aiml_project.entity.ApplicationStatus;
import java.time.LocalDateTime;

public class ApplicationFilterDTO {

    private Long jobId;
    private ApplicationStatus status;
    private LocalDateTime fromDate;
    private LocalDateTime toDate;
    private Double minMatchScore;
    private String searchKeyword;
    private Boolean shortlistedOnly;
    private Boolean interviewedOnly;
    private String sortBy;
    private String sortDirection;

    // Getters and Setters
    public Long getJobId() { return jobId; }
    public void setJobId(Long jobId) { this.jobId = jobId; }

    public ApplicationStatus getStatus() { return status; }
    public void setStatus(ApplicationStatus status) { this.status = status; }

    public LocalDateTime getFromDate() { return fromDate; }
    public void setFromDate(LocalDateTime fromDate) { this.fromDate = fromDate; }

    public LocalDateTime getToDate() { return toDate; }
    public void setToDate(LocalDateTime toDate) { this.toDate = toDate; }

    public Double getMinMatchScore() { return minMatchScore; }
    public void setMinMatchScore(Double minMatchScore) { this.minMatchScore = minMatchScore; }

    public String getSearchKeyword() { return searchKeyword; }
    public void setSearchKeyword(String searchKeyword) { this.searchKeyword = searchKeyword; }

    public Boolean getShortlistedOnly() { return shortlistedOnly; }
    public void setShortlistedOnly(Boolean shortlistedOnly) { this.shortlistedOnly = shortlistedOnly; }

    public Boolean getInterviewedOnly() { return interviewedOnly; }
    public void setInterviewedOnly(Boolean interviewedOnly) { this.interviewedOnly = interviewedOnly; }

    public String getSortBy() { return sortBy; }
    public void setSortBy(String sortBy) { this.sortBy = sortBy; }

    public String getSortDirection() { return sortDirection; }
    public void setSortDirection(String sortDirection) { this.sortDirection = sortDirection; }
}
