package com.cv.aiml_project.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name = "users")
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true)
    private String username;

    @Column(nullable = false, unique = true)
    private String email;

    @Column(nullable = false)
    private String password;

    @Column(nullable = false)
    private String firstName;

    @Column(nullable = false)
    private String lastName;

    private String phone;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private Role role = Role.CANDIDATE;

    // Professional info (keep these)
    @Column(length = 2000)
    private String skills;

    private Integer experienceYears;

    @Column(length = 2000)
    private String education;

    /** Career level classified by AI: Junior, Mid, or Senior */
    @Column(name = "career_level", length = 20)
    private String careerLevel;

    // Remove all resume-related fields from here!
    // Remove: resumePath, resumeFileName, resumeOriginalName, resumeContentType,
    // resumeFileSize, resumeUploadDate, resumeTextContent, mlProcessed, mlScore, mlConfidence

    // One-to-many relationship with resumes
    @OneToMany(mappedBy = "user", cascade = CascadeType.ALL, orphanRemoval = true)
    @OrderBy("uploadDate DESC")
    private List<Resume> resumes = new ArrayList<>();

    @Column(name = "is_active")
    private boolean isActive = true;

    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }

    // Constructors
    public User() {}

    public User(String username, String email, String password, String firstName,
                String lastName, String phone, Role role) {
        this();
        this.username = username;
        this.email = email;
        this.password = password;
        this.firstName = firstName;
        this.lastName = lastName;
        this.phone = phone;
        this.role = role;
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getUsername() { return username; }
    public void setUsername(String username) { this.username = username; }

    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }

    public String getPassword() { return password; }
    public void setPassword(String password) { this.password = password; }

    public String getFirstName() { return firstName; }
    public void setFirstName(String firstName) { this.firstName = firstName; }

    public String getLastName() { return lastName; }
    public void setLastName(String lastName) { this.lastName = lastName; }

    public String getPhone() { return phone; }
    public void setPhone(String phone) { this.phone = phone; }

    public Role getRole() { return role; }
    public void setRole(Role role) { this.role = role; }

    public String getSkills() { return skills; }
    public void setSkills(String skills) { this.skills = skills; }

    public Integer getExperienceYears() { return experienceYears; }
    public void setExperienceYears(Integer experienceYears) { this.experienceYears = experienceYears; }

    public String getEducation() { return education; }
    public void setEducation(String education) { this.education = education; }

    public String getCareerLevel() { return careerLevel; }
    public void setCareerLevel(String careerLevel) { this.careerLevel = careerLevel; }

    public List<Resume> getResumes() { return resumes; }
    public void setResumes(List<Resume> resumes) { this.resumes = resumes; }

    public boolean isActive() { return isActive; }
    public void setActive(boolean active) { isActive = active; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }

    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }

    // Helper methods
    public String getFullName() {
        return firstName + " " + lastName;
    }

    public boolean isCandidate() {
        return role == Role.CANDIDATE;
    }

    public boolean isHR() {
        return role == Role.HR;
    }

    public boolean isAdmin() {
        return role == Role.ADMIN;
    }

    // Resume management helpers
    public Resume getCurrentResume() {
        return resumes.stream()
                .filter(Resume::isCurrent)
                .findFirst()
                .orElse(null);
    }

    public boolean hasResume() {
        return getCurrentResume() != null;
    }

    public Double getCurrentMlScore() {
        Resume current = getCurrentResume();
        return current != null ? current.getMlScore() : null;
    }

    public boolean isMlProcessed() {
        Resume current = getCurrentResume();
        return current != null && current.isMlProcessed();
    }

    public void addResume(Resume resume) {
        resumes.add(resume);
        resume.setUser(this);
    }

    public void removeResume(Resume resume) {
        resumes.remove(resume);
        resume.setUser(null);
    }

    public void setTempAttribute(String matchScore, Object score) {
    }
}