package com.cv.aiml_project.dto;

import org.springframework.web.multipart.MultipartFile;

public class JobApplicationRequest {
    private Long jobId;
    private MultipartFile resume;

    // Getters and setters
    public Long getJobId() { return jobId; }
    public void setJobId(Long jobId) { this.jobId = jobId; }

    public MultipartFile getResume() { return resume; }
    public void setResume(MultipartFile resume) { this.resume = resume; }
}
