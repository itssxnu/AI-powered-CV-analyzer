package com.cv.aiml_project.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "resumes")
public class Resume {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    // File metadata
    @Column(name = "file_name", nullable = false)
    private String fileName;

    @Column(name = "original_name", nullable = false)
    private String originalName;

    @Column(name = "content_type", nullable = false)
    private String contentType;

    @Column(name = "file_size", nullable = false)
    private Long fileSize;

    @Column(name = "file_path", nullable = false)
    private String filePath;

    @Column(name = "upload_date", nullable = false)
    private LocalDateTime uploadDate;

    // Extracted text from PDF
    @Column(name = "extracted_text", columnDefinition = "LONGTEXT")
    private String extractedText;

    // AI/ML processed data
    @Column(name = "ml_processed")
    private boolean mlProcessed = false;

    @Column(name = "ml_score")
    private Double mlScore;

    @Column(name = "ml_confidence")
    private Double mlConfidence;

    @Column(name = "ml_processed_date")
    private LocalDateTime mlProcessedDate;

    @Column(name = "ml_raw_response", columnDefinition = "TEXT")
    private String mlRawResponse;

    // Version tracking
    @Column(name = "version")
    private Integer version = 1;

    @Column(name = "is_current")
    private boolean isCurrent = true;

    // Audit fields
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
        if (uploadDate == null) {
            uploadDate = LocalDateTime.now();
        }
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }

    // Constructors
    public Resume() {}

    public Resume(User user, String fileName, String originalName,
                  String contentType, Long fileSize, String filePath) {
        this.user = user;
        this.fileName = fileName;
        this.originalName = originalName;
        this.contentType = contentType;
        this.fileSize = fileSize;
        this.filePath = filePath;
        this.uploadDate = LocalDateTime.now();
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public User getUser() { return user; }
    public void setUser(User user) { this.user = user; }

    public String getFileName() { return fileName; }
    public void setFileName(String fileName) { this.fileName = fileName; }

    public String getOriginalName() { return originalName; }
    public void setOriginalName(String originalName) { this.originalName = originalName; }

    public String getContentType() { return contentType; }
    public void setContentType(String contentType) { this.contentType = contentType; }

    public Long getFileSize() { return fileSize; }
    public void setFileSize(Long fileSize) { this.fileSize = fileSize; }

    public String getFilePath() { return filePath; }
    public void setFilePath(String filePath) { this.filePath = filePath; }

    public LocalDateTime getUploadDate() { return uploadDate; }
    public void setUploadDate(LocalDateTime uploadDate) { this.uploadDate = uploadDate; }

    public String getExtractedText() { return extractedText; }
    public void setExtractedText(String extractedText) { this.extractedText = extractedText; }

    public boolean isMlProcessed() { return mlProcessed; }
    public void setMlProcessed(boolean mlProcessed) { this.mlProcessed = mlProcessed; }

    public Double getMlScore() { return mlScore; }
    public void setMlScore(Double mlScore) { this.mlScore = mlScore; }

    public Double getMlConfidence() { return mlConfidence; }
    public void setMlConfidence(Double mlConfidence) { this.mlConfidence = mlConfidence; }

    public LocalDateTime getMlProcessedDate() { return mlProcessedDate; }
    public void setMlProcessedDate(LocalDateTime mlProcessedDate) { this.mlProcessedDate = mlProcessedDate; }

    public String getMlRawResponse() { return mlRawResponse; }
    public void setMlRawResponse(String mlRawResponse) { this.mlRawResponse = mlRawResponse; }

    public Integer getVersion() { return version; }
    public void setVersion(Integer version) { this.version = version; }

    public boolean isCurrent() { return isCurrent; }
    public void setCurrent(boolean current) { isCurrent = current; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }

    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }

    // Helper methods
    public String getFormattedFileSize() {
        if (fileSize == null) return "0 B";
        if (fileSize < 1024) return fileSize + " B";
        if (fileSize < 1024 * 1024) return String.format("%.1f KB", fileSize / 1024.0);
        return String.format("%.1f MB", fileSize / (1024.0 * 1024.0));
    }

    public String getScoreColorClass() {
        if (mlScore == null) return "";
        if (mlScore >= 80) return "bg-success";
        if (mlScore >= 60) return "bg-warning";
        return "bg-danger";
    }
}