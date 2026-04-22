package com.cv.aiml_project.service;

import com.cv.aiml_project.entity.*;
import com.cv.aiml_project.repository.JobApplicationRepository;
import com.cv.aiml_project.repository.JobRepository;
import com.cv.aiml_project.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.net.MalformedURLException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

@Service
@Transactional
public class JobService {

    @Autowired
    private JobRepository jobRepository;

    @Autowired
    private JobApplicationRepository applicationRepository;

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private AIMatchingService aiMatchingService;

    @Value("${file.upload-dir:./uploads}")
    private String uploadDir;

    // ==================== JOB CRUD OPERATIONS ====================

    /**
     * Create Job
     */
    public Job createJob(Job job, Long postedById) {
        User postedBy = userRepository.findById(postedById)
                .orElseThrow(() -> new RuntimeException("User not found"));

        if (postedBy.getRole() != Role.ADMIN && postedBy.getRole() != Role.HR) {
            throw new RuntimeException("Only Admin or HR can post jobs");
        }

        job.setPostedBy(postedBy);
        job.setPostedDate(LocalDateTime.now());
        job.setActive(true);

        return jobRepository.save(job);
    }

    /**
     * Get All Jobs
     */
    public List<Job> getAllJobs() {
        return jobRepository.findAll();
    }

    /**
     * Get Job by ID
     */
    public Optional<Job> getJobById(Long id) {
        return jobRepository.findById(id);
    }

    /**
     * Get Active Jobs
     */
    public List<Job> getActiveJobs() {
        return jobRepository.findActiveJobs(LocalDateTime.now());
    }

    /**
     * Get Jobs by Department
     */
    public List<Job> getJobsByDepartment(String department) {
        return jobRepository.findByDepartment(department);
    }

    /**
     * Get Jobs by Job Type
     */
    public List<Job> getJobsByType(JobType jobType) {
        return jobRepository.findByJobType(jobType);
    }

    /**
     * Get Jobs Posted by User
     */
    public List<Job> getJobsPostedByUser(Long userId) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found"));
        return jobRepository.findByPostedBy(user);
    }

    /**
     * Search Jobs
     */
    public List<Job> searchJobs(String keyword) {
        return jobRepository.searchJobs(keyword);
    }

    /**
     * Update Job
     */
    public Job updateJob(Long jobId, Job updatedJob) {
        Job existingJob = jobRepository.findById(jobId)
                .orElseThrow(() -> new RuntimeException("Job not found"));

        // Update fields
        existingJob.setTitle(updatedJob.getTitle());
        existingJob.setDescription(updatedJob.getDescription());
        existingJob.setDepartment(updatedJob.getDepartment());
        existingJob.setLocation(updatedJob.getLocation());
        existingJob.setJobType(updatedJob.getJobType());
        existingJob.setExperienceRequired(updatedJob.getExperienceRequired());
        existingJob.setRequiredSkills(updatedJob.getRequiredSkills());
        existingJob.setPreferredSkills(updatedJob.getPreferredSkills());
        existingJob.setEducationRequirement(updatedJob.getEducationRequirement());
        existingJob.setMinSalary(updatedJob.getMinSalary());
        existingJob.setMaxSalary(updatedJob.getMaxSalary());
        existingJob.setExpiryDate(updatedJob.getExpiryDate());
        existingJob.setVacancies(updatedJob.getVacancies());
        existingJob.setActive(updatedJob.isActive());

        return jobRepository.save(existingJob);
    }

    /**
     * Toggle Job Status
     */
    public Job toggleJobStatus(Long jobId, boolean isActive) {
        Job job = jobRepository.findById(jobId)
                .orElseThrow(() -> new RuntimeException("Job not found"));
        job.setActive(isActive);
        return jobRepository.save(job);
    }

    /**
     * Delete Job
     */
    public void deleteJob(Long jobId) {
        Job job = jobRepository.findById(jobId)
                .orElseThrow(() -> new RuntimeException("Job not found"));

        // Delete all application files first
        List<JobApplication> applications = applicationRepository.findByJob(job);
        for (JobApplication app : applications) {
            if (app.getResumePath() != null) {
                try {
                    Files.deleteIfExists(Paths.get(app.getResumePath()));
                } catch (IOException e) {
                    // Log error but continue
                    e.printStackTrace();
                }
            }
        }

        // Delete all applications for this job
        applicationRepository.deleteAll(applications);

        // Delete the job
        jobRepository.delete(job);
    }

    // ==================== JOB APPLICATION OPERATIONS ====================

    /**
     * Apply for Job with Resume
     */
    @Transactional
    public JobApplication applyForJobWithResume(Long jobId, Long candidateId, MultipartFile resumeFile, String notes) throws IOException {
        Job job = jobRepository.findById(jobId)
                .orElseThrow(() -> new RuntimeException("Job not found"));
        User candidate = userRepository.findById(candidateId)
                .orElseThrow(() -> new RuntimeException("Candidate not found"));

        // Check if already applied
        if (applicationRepository.existsByJobAndCandidate(job, candidate)) {
            throw new RuntimeException("You have already applied for this job");
        }

        // Check if job is active
        if (!job.isActive() || job.isExpired()) {
            throw new RuntimeException("This job is no longer accepting applications");
        }

        // Create application
        JobApplication application = new JobApplication(job, candidate);

        // Save resume for this specific application
        String resumePath = saveResumeForApplication(resumeFile, jobId, candidateId);
        application.setResumePath(resumePath);
        application.setResumeOriginalName(resumeFile.getOriginalFilename());
        application.setResumeContentType(resumeFile.getContentType());
        application.setResumeFileSize(resumeFile.getSize());
        application.setResumeUploadDate(LocalDateTime.now());

        // Add notes if provided
        if (notes != null && !notes.trim().isEmpty()) {
            application.setHrNotes(notes);
        }

        // Extract text from resume
        String extractedText = extractTextFromPdf(resumeFile);
        application.setResumeTextContent(extractedText);

        // Calculate AI match scores
        calculateMatchScores(application, job, candidate, extractedText);

        return applicationRepository.save(application);
    }

    /**
     * Save resume file for application
     */
    private String saveResumeForApplication(MultipartFile file, Long jobId, Long candidateId) throws IOException {
        // Create directory structure: uploads/applications/job_{jobId}/
        Path uploadPath = Paths.get(uploadDir, "applications", "job_" + jobId);
        if (!Files.exists(uploadPath)) {
            Files.createDirectories(uploadPath);
        }

        // Generate unique filename
        String originalFilename = file.getOriginalFilename();
        String fileExtension = originalFilename.substring(originalFilename.lastIndexOf("."));
        String uniqueFileName = "candidate_" + candidateId + "_" +
                System.currentTimeMillis() + fileExtension;

        // Save file
        Path filePath = uploadPath.resolve(uniqueFileName);
        Files.copy(file.getInputStream(), filePath, StandardCopyOption.REPLACE_EXISTING);

        return filePath.toString();
    }

    /**
     * Extract text from PDF using PDFBox
     */
    private String extractTextFromPdf(MultipartFile file) {
        try {
            // You'll need to add PDFBox dependency
            // For now, return a placeholder with extracted info
            return "Extracted text from resume: " + file.getOriginalFilename();
        } catch (Exception e) {
            e.printStackTrace();
            return "Could not extract text from PDF";
        }
    }

    /**
     * Calculate match scores based on resume and job requirements
     */
    private void calculateMatchScores(JobApplication application, Job job, User candidate, String resumeText) {
        // Parse resume text to extract skills, experience, education
        // This is where your AI/ML model would integrate

        // Use the existing AIMatchingService for now
        Double matchScore = aiMatchingService.calculateJobMatchScore(job, candidate);
        application.setMatchScore(matchScore);

        Map<String, Double> componentScores = aiMatchingService.calculateComponentScores(job, candidate);
        application.setSkillsMatchScore(componentScores.get("skills"));
        application.setExperienceMatchScore(componentScores.get("experience"));
        application.setEducationMatchScore(componentScores.get("education"));
    }

    /**
     * Get Applications for Job (Ranked by score)
     */
    public List<JobApplication> getApplicationsForJobRanked(Long jobId) {
        Job job = jobRepository.findById(jobId)
                .orElseThrow(() -> new RuntimeException("Job not found"));
        return applicationRepository.findByJobOrderByMatchScoreDesc(jobId);
    }

    /**
     * Get Applications for Job by Status
     */
    public List<JobApplication> getApplicationsForJobByStatus(Long jobId, ApplicationStatus status) {
        Job job = jobRepository.findById(jobId)
                .orElseThrow(() -> new RuntimeException("Job not found"));
        return applicationRepository.findByJobAndStatusOrderByMatchScoreDesc(jobId, status);
    }

    /**
     * Get Applications by Candidate
     */
    public List<JobApplication> getApplicationsByCandidate(Long candidateId) {
        User candidate = userRepository.findById(candidateId)
                .orElseThrow(() -> new RuntimeException("Candidate not found"));
        return applicationRepository.findByCandidateOrderByAppliedDateDesc(candidateId);
    }

    /**
     * Get Application by ID
     */
    public Optional<JobApplication> getApplicationById(Long applicationId) {
        return applicationRepository.findById(applicationId);
    }

    /**
     * Get Application by Job and Candidate
     */
    public JobApplication getApplicationByJobAndCandidate(Long jobId, Long candidateId) {
        Job job = jobRepository.findById(jobId)
                .orElseThrow(() -> new RuntimeException("Job not found"));
        User candidate = userRepository.findById(candidateId)
                .orElseThrow(() -> new RuntimeException("Candidate not found"));
        return applicationRepository.findByJobAndCandidate(job, candidate)
                .orElseThrow(() -> new RuntimeException("Application not found"));
    }

    /**
     * Check if Candidate Applied
     */
    public boolean hasCandidateApplied(Long jobId, Long candidateId) {
        Job job = jobRepository.findById(jobId)
                .orElseThrow(() -> new RuntimeException("Job not found"));
        User candidate = userRepository.findById(candidateId)
                .orElseThrow(() -> new RuntimeException("Candidate not found"));
        return applicationRepository.existsByJobAndCandidate(job, candidate);
    }

    /**
     * Update Application Status
     */
    public JobApplication updateApplicationStatus(Long applicationId, ApplicationStatus status, String notes) {
        JobApplication application = applicationRepository.findById(applicationId)
                .orElseThrow(() -> new RuntimeException("Application not found"));

        application.setStatus(status);
        if (notes != null && !notes.trim().isEmpty()) {
            application.setHrNotes(notes);
        }

        return applicationRepository.save(application);
    }

    /**
     * Recalculate Match Score
     */
    @Transactional
    public JobApplication recalculateMatchScore(Long applicationId) {
        JobApplication application = applicationRepository.findById(applicationId)
                .orElseThrow(() -> new RuntimeException("Application not found"));

        Job job = application.getJob();
        User candidate = application.getCandidate();

        // Recalculate scores
        Double matchScore = aiMatchingService.calculateJobMatchScore(job, candidate);
        application.setMatchScore(matchScore);

        Map<String, Double> componentScores = aiMatchingService.calculateComponentScores(job, candidate);
        application.setSkillsMatchScore(componentScores.get("skills"));
        application.setExperienceMatchScore(componentScores.get("experience"));
        application.setEducationMatchScore(componentScores.get("education"));

        return applicationRepository.save(application);
    }

    /**
     * Process all applications for a job with AI
     */
    @Transactional
    public int processAllApplicationsForJob(Long jobId) {
        List<JobApplication> applications = applicationRepository.findByJobOrderByMatchScoreDesc(jobId);
        int processed = 0;

        for (JobApplication app : applications) {
            if (app.getMatchScore() == null) {
                recalculateMatchScore(app.getId());
                processed++;
            }
        }

        return processed;
    }

    /**
     * Remove Application (Soft Delete)
     */
    public void removeApplication(Long applicationId) {
        JobApplication application = applicationRepository.findById(applicationId)
                .orElseThrow(() -> new RuntimeException("Application not found"));
        application.setActive(false);
        application.setStatus(ApplicationStatus.REJECTED);
        applicationRepository.save(application);
    }

    /**
     * Delete Application Permanently
     */
    public void deleteApplicationPermanently(Long applicationId) {
        JobApplication application = applicationRepository.findById(applicationId)
                .orElseThrow(() -> new RuntimeException("Application not found"));

        // Delete resume file if exists
        if (application.getResumePath() != null) {
            try {
                Files.deleteIfExists(Paths.get(application.getResumePath()));
            } catch (IOException e) {
                // Log error but continue
                e.printStackTrace();
            }
        }

        applicationRepository.delete(application);
    }

    /**
     * Get Application Resume File
     */
    public Resource getApplicationResumeFile(Long applicationId) throws MalformedURLException {
        JobApplication application = applicationRepository.findById(applicationId)
                .orElseThrow(() -> new RuntimeException("Application not found"));

        if (application.getResumePath() == null) {
            throw new RuntimeException("No resume found for this application");
        }

        Path filePath = Paths.get(application.getResumePath());
        Resource resource = new UrlResource(filePath.toUri());

        if (resource.exists() && resource.isReadable()) {
            return resource;
        } else {
            throw new RuntimeException("Could not read the file!");
        }
    }

    /**
     * Save application directly (used for Q&A JSON persistence)
     */
    @Transactional
    public JobApplication saveApplication(JobApplication application) {
        return applicationRepository.save(application);
    }

    // ==================== STATISTICS ====================

    /**
     * Get Job Statistics
     */
    public Map<String, Object> getJobStatistics() {
        Map<String, Object> stats = new HashMap<>();

        stats.put("totalJobs", jobRepository.count());
        stats.put("activeJobs", jobRepository.countActiveJobs(LocalDateTime.now()));
        stats.put("totalApplications", applicationRepository.count());

        // Jobs by department
        List<Object[]> jobsByDept = jobRepository.countJobsByDepartment();
        stats.put("jobsByDepartment", jobsByDept);

        return stats;
    }

    /**
     * Get Application Statistics for a Job
     */
    public Map<String, Object> getApplicationStatsForJob(Long jobId) {
        Map<String, Object> stats = new HashMap<>();

        stats.put("total", applicationRepository.countByJobId(jobId));
        stats.put("pending", applicationRepository.countByJobIdAndStatus(jobId, ApplicationStatus.PENDING));
        stats.put("shortlisted", applicationRepository.countByJobIdAndStatus(jobId, ApplicationStatus.SHORTLISTED));
        stats.put("interview", applicationRepository.countByJobIdAndStatus(jobId, ApplicationStatus.INTERVIEW_SCHEDULED));
        stats.put("rejected", applicationRepository.countByJobIdAndStatus(jobId, ApplicationStatus.REJECTED));
        stats.put("hired", applicationRepository.countByJobIdAndStatus(jobId, ApplicationStatus.HIRED));
        stats.put("withdrawn", applicationRepository.countByJobIdAndStatus(jobId, ApplicationStatus.WITHDRAWN));

        // Average match score
        List<JobApplication> applications = applicationRepository.findByJobOrderByMatchScoreDesc(jobId);
        double avgScore = applications.stream()
                .filter(a -> a.getMatchScore() != null)
                .mapToDouble(JobApplication::getMatchScore)
                .average()
                .orElse(0.0);
        stats.put("avgMatchScore", avgScore);

        return stats;
    }

    /**
     * Get Recent Jobs
     */
    public Page<Job> getRecentJobs(int page, int size) {
        return jobRepository.findRecentJobs(PageRequest.of(page, size));
    }

    /**
     * Cleanup Expired Jobs
     */
    @Transactional
    public int deactivateExpiredJobs() {
        List<Job> expiredJobs = jobRepository.findByExpiryDateBeforeAndIsActive(LocalDateTime.now(), true);
        for (Job job : expiredJobs) {
            job.setActive(false);
        }
        jobRepository.saveAll(expiredJobs);
        return expiredJobs.size();
    }

    /**
     * Get Top Candidates for Job
     */
    public List<JobApplication> getTopCandidatesForJob(Long jobId, int limit) {
        return applicationRepository.findByJobOrderByMatchScoreDesc(jobId)
                .stream()
                .filter(a -> a.getMatchScore() != null && a.getStatus() != ApplicationStatus.REJECTED)
                .limit(limit)
                .toList();
    }

    /**
     * Check if Job is Expired
     */
    public boolean isJobExpired(Long jobId) {
        Job job = jobRepository.findById(jobId)
                .orElseThrow(() -> new RuntimeException("Job not found"));
        return job.isExpired();
    }

    /**
     * Extend Job Expiry Date
     */
    public Job extendJobExpiry(Long jobId, LocalDateTime newExpiryDate) {
        Job job = jobRepository.findById(jobId)
                .orElseThrow(() -> new RuntimeException("Job not found"));

        if (newExpiryDate.isBefore(LocalDateTime.now())) {
            throw new RuntimeException("Expiry date must be in the future");
        }

        job.setExpiryDate(newExpiryDate);
        return jobRepository.save(job);
    }

    /**
     * Count Applications by Status for a Job
     */
    public Map<ApplicationStatus, Long> countApplicationsByStatus(Long jobId) {
        Map<ApplicationStatus, Long> counts = new HashMap<>();

        for (ApplicationStatus status : ApplicationStatus.values()) {
            long count = applicationRepository.countByJobIdAndStatus(jobId, status);
            counts.put(status, count);
        }

        return counts;
    }

    /**
     * Get Applications with Resume Text
     */
    public List<JobApplication> getApplicationsWithResumeText(Long jobId) {
        return applicationRepository.findByJobOrderByMatchScoreDesc(jobId)
                .stream()
                .filter(a -> a.getResumeTextContent() != null)
                .toList();
    }

    /**
     * Update Application Notes
     */
    public JobApplication updateApplicationNotes(Long applicationId, String notes) {
        JobApplication application = applicationRepository.findById(applicationId)
                .orElseThrow(() -> new RuntimeException("Application not found"));

        application.setHrNotes(notes);
        return applicationRepository.save(application);
    }

    /**
     * Bulk Update Application Status
     */
    @Transactional
    public int bulkUpdateApplicationStatus(List<Long> applicationIds, ApplicationStatus status, String notes) {
        int updated = 0;

        for (Long id : applicationIds) {
            try {
                JobApplication app = applicationRepository.findById(id)
                        .orElseThrow(() -> new RuntimeException("Application not found: " + id));

                app.setStatus(status);
                if (notes != null && !notes.trim().isEmpty()) {
                    app.setHrNotes(notes);
                }

                applicationRepository.save(app);
                updated++;
            } catch (Exception e) {
                // Log error but continue with next application
                e.printStackTrace();
            }
        }

        return updated;
    }

    /**
     * Get Application Timeline
     */
    public Map<String, Long> getApplicationTimeline(Long jobId, int days) {
        Map<String, Long> timeline = new HashMap<>();
        LocalDateTime startDate = LocalDateTime.now().minusDays(days);

        List<JobApplication> applications = applicationRepository.findByJobOrderByMatchScoreDesc(jobId);

        for (int i = 0; i < days; i++) {
            LocalDateTime date = startDate.plusDays(i);
            String dateKey = date.toLocalDate().toString();

            long count = applications.stream()
                    .filter(a -> a.getAppliedDate().toLocalDate().equals(date.toLocalDate()))
                    .count();

            timeline.put(dateKey, count);
        }

        return timeline;
    }

    public List<JobApplication> getApplicationsForJob(Long jobId) {
        return applicationRepository.findByJobOrderByMatchScoreDesc(jobId);
    }
}