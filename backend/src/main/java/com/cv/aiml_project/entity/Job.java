package com.cv.aiml_project.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name = "jobs")
public class Job {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String title;

    @Column(nullable = false, length = 2000)
    private String description;

    @Column(nullable = false)
    private String department;

    @Column(nullable = false)
    private String location;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private JobType jobType = JobType.FULL_TIME;

    @Column(nullable = false)
    private String experienceRequired;

    @Column(length = 1000)
    private String requiredSkills;

    @Column(length = 1000)
    private String preferredSkills;

    @Column(nullable = false)
    private String educationRequirement;

    @Column(nullable = false)
    private Integer minSalary;

    private Integer maxSalary;

    @Column(nullable = false)
    private LocalDateTime postedDate;

    private LocalDateTime expiryDate;

    @Column(nullable = false)
    private boolean isActive = true;

    @Column(nullable = false)
    private Integer vacancies;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "posted_by", nullable = false)
    private User postedBy;

    @OneToMany(mappedBy = "job", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<JobApplication> applications = new ArrayList<>();

    // Audit fields
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
        if (postedDate == null) {
            postedDate = LocalDateTime.now();
        }
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }

    // Constructors
    public Job() {}

    public Job(String title, String description, String department, String location,
               JobType jobType, String experienceRequired, String requiredSkills,
               String educationRequirement, Integer minSalary, Integer maxSalary,
               LocalDateTime expiryDate, Integer vacancies, User postedBy) {
        this.title = title;
        this.description = description;
        this.department = department;
        this.location = location;
        this.jobType = jobType;
        this.experienceRequired = experienceRequired;
        this.requiredSkills = requiredSkills;
        this.educationRequirement = educationRequirement;
        this.minSalary = minSalary;
        this.maxSalary = maxSalary;
        this.expiryDate = expiryDate;
        this.vacancies = vacancies;
        this.postedBy = postedBy;
        this.postedDate = LocalDateTime.now();
        this.isActive = true;
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getTitle() { return title; }
    public void setTitle(String title) { this.title = title; }

    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }

    public String getDepartment() { return department; }
    public void setDepartment(String department) { this.department = department; }

    public String getLocation() { return location; }
    public void setLocation(String location) { this.location = location; }

    public JobType getJobType() { return jobType; }
    public void setJobType(JobType jobType) { this.jobType = jobType; }

    public String getExperienceRequired() { return experienceRequired; }
    public void setExperienceRequired(String experienceRequired) { this.experienceRequired = experienceRequired; }

    public String getRequiredSkills() { return requiredSkills; }
    public void setRequiredSkills(String requiredSkills) { this.requiredSkills = requiredSkills; }

    public String getPreferredSkills() { return preferredSkills; }
    public void setPreferredSkills(String preferredSkills) { this.preferredSkills = preferredSkills; }

    public String getEducationRequirement() { return educationRequirement; }
    public void setEducationRequirement(String educationRequirement) { this.educationRequirement = educationRequirement; }

    public Integer getMinSalary() { return minSalary; }
    public void setMinSalary(Integer minSalary) { this.minSalary = minSalary; }

    public Integer getMaxSalary() { return maxSalary; }
    public void setMaxSalary(Integer maxSalary) { this.maxSalary = maxSalary; }

    public LocalDateTime getPostedDate() { return postedDate; }
    public void setPostedDate(LocalDateTime postedDate) { this.postedDate = postedDate; }

    public LocalDateTime getExpiryDate() { return expiryDate; }
    public void setExpiryDate(LocalDateTime expiryDate) { this.expiryDate = expiryDate; }

    public boolean isActive() { return isActive; }
    public void setActive(boolean active) { isActive = active; }

    public Integer getVacancies() { return vacancies; }
    public void setVacancies(Integer vacancies) { this.vacancies = vacancies; }

    public User getPostedBy() { return postedBy; }
    public void setPostedBy(User postedBy) { this.postedBy = postedBy; }

    public List<JobApplication> getApplications() { return applications; }
    public void setApplications(List<JobApplication> applications) { this.applications = applications; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }

    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }

    // Helper methods
    public String getSalaryRange() {
        if (minSalary == null && maxSalary == null) return "Not disclosed";
        if (minSalary == null) return "Up to $" + maxSalary;
        if (maxSalary == null) return "From $" + minSalary;
        return "$" + minSalary + " - $" + maxSalary;
    }

    public boolean isExpired() {
        return expiryDate != null && expiryDate.isBefore(LocalDateTime.now());
    }

    public int getRemainingDays() {
        if (expiryDate == null) return 30;
        return (int) java.time.Duration.between(LocalDateTime.now(), expiryDate).toDays();
    }

    public int getApplicationCount() {
        return applications != null ? applications.size() : 0;
    }

    public int getPendingApplicationCount() {
        if (applications == null) return 0;
        return (int) applications.stream()
                .filter(app -> app.getStatus() == ApplicationStatus.PENDING)
                .count();
    }

}
