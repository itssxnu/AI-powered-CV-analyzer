package com.cv.aiml_project.service;

import com.cv.aiml_project.dto.ApplicationFilterDTO;
import com.cv.aiml_project.dto.ApplicationStatusUpdateDTO;
import com.cv.aiml_project.dto.InterviewScheduleDTO;
import com.cv.aiml_project.entity.*;
import com.cv.aiml_project.repository.JobApplicationRepository; // Change this import
import com.cv.aiml_project.repository.JobRepository;
import com.cv.aiml_project.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
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
import java.time.format.DateTimeFormatter;
import java.util.*;

@Service
@Transactional
public class ApplicationService {

    @Autowired
    private JobApplicationRepository applicationRepository; // Changed to JobApplicationRepository

    @Autowired
    private JobRepository jobRepository;

    @Autowired
    private UserRepository userRepository;

    @Autowired(required = false) // Make optional to avoid errors if not configured
    private JavaMailSender mailSender;

    @Value("${file.upload.application-dir:./uploads/applications}")
    private String uploadDir;

    @Value("${app.base-url:http://localhost:8080}")
    private String baseUrl;

    // ==================== APPLICATION OPERATIONS ====================

    /**
     * Apply for a job
     */
    @Transactional
    public JobApplication applyForJob(Long jobId, Long candidateId, MultipartFile resumeFile,
                                      String coverLetter, Double expectedSalary,
                                      String noticePeriod, String currentCompany,
                                      String currentPosition) throws IOException {

        Job job = jobRepository.findById(jobId)
                .orElseThrow(() -> new RuntimeException("Job not found with id: " + jobId));

        User candidate = userRepository.findById(candidateId)
                .orElseThrow(() -> new RuntimeException("Candidate not found with id: " + candidateId));

        // Check if already applied - using existsByJobAndCandidate
        if (applicationRepository.existsByJobAndCandidate(job, candidate)) {
            throw new RuntimeException("You have already applied for this job");
        }

        // Check if job is active
        if (!job.isActive() || job.isExpired()) {
            throw new RuntimeException("This job is no longer accepting applications");
        }

        // Create application
        JobApplication application = new JobApplication(job, candidate);
        application.setCoverLetter(coverLetter);
        application.setExpectedSalary(expectedSalary);
        application.setNoticePeriod(noticePeriod);
        application.setCurrentCompany(currentCompany);
        application.setCurrentPosition(currentPosition);

        // Save resume
        if (resumeFile != null && !resumeFile.isEmpty()) {
            String resumePath = saveResumeFile(resumeFile, jobId, candidateId);
            application.setResumePath(resumePath);
            application.setResumeOriginalName(resumeFile.getOriginalFilename());
            application.setResumeContentType(resumeFile.getContentType());
            application.setResumeFileSize(resumeFile.getSize());
            application.setResumeUploadDate(LocalDateTime.now());
        }

        return applicationRepository.save(application);
    }

    /**
     * Save resume file
     */
    private String saveResumeFile(MultipartFile file, Long jobId, Long candidateId) throws IOException {
        // Create directory structure: uploads/applications/job_{jobId}/candidate_{candidateId}/
        Path uploadPath = Paths.get(uploadDir, "job_" + jobId, "candidate_" + candidateId);
        if (!Files.exists(uploadPath)) {
            Files.createDirectories(uploadPath);
        }

        // Generate unique filename - Fixed null pointer issue
        String originalFilename = file.getOriginalFilename();
        String fileExtension = "";
        if (originalFilename != null && originalFilename.contains(".")) {
            fileExtension = originalFilename.substring(originalFilename.lastIndexOf("."));
        } else {
            fileExtension = ".pdf"; // Default extension
        }
        String fileName = "resume_" + System.currentTimeMillis() + fileExtension;

        // Save file
        Path filePath = uploadPath.resolve(fileName);
        Files.copy(file.getInputStream(), filePath, StandardCopyOption.REPLACE_EXISTING);

        return filePath.toString();
    }

    /**
     * Get application by ID
     */
    public Optional<JobApplication> getApplicationById(Long applicationId) {
        return applicationRepository.findById(applicationId);
    }

    /**
     * Get applications with filters - Custom implementation since the repository method doesn't exist
     */
    public List<JobApplication> getApplicationsWithFilters(ApplicationFilterDTO filters) {
        // Start with all applications for the job
        List<JobApplication> applications = applicationRepository.findByJobOrderByMatchScoreDesc(filters.getJobId());

        // Apply filters manually
        return applications.stream()
                .filter(app -> filters.getStatus() == null || app.getStatus() == filters.getStatus())
                .filter(app -> filters.getFromDate() == null || !app.getAppliedDate().isBefore(filters.getFromDate()))
                .filter(app -> filters.getToDate() == null || !app.getAppliedDate().isAfter(filters.getToDate()))
                .filter(app -> filters.getMinMatchScore() == null ||
                        (app.getMatchScore() != null && app.getMatchScore() >= filters.getMinMatchScore()))
                .filter(app -> filters.getShortlistedOnly() == null || !filters.getShortlistedOnly() ||
                        (app.getShortlistedDate() != null))
                .filter(app -> filters.getInterviewedOnly() == null || !filters.getInterviewedOnly() ||
                        app.isInterviewScheduled())
                .filter(app -> filters.getSearchKeyword() == null ||
                        filters.getSearchKeyword().isEmpty() ||
                        (app.getCandidate().getFullName() != null &&
                                app.getCandidate().getFullName().toLowerCase().contains(filters.getSearchKeyword().toLowerCase())) ||
                        (app.getCandidate().getEmail() != null &&
                                app.getCandidate().getEmail().toLowerCase().contains(filters.getSearchKeyword().toLowerCase())))
                .toList();
    }

    /**
     * Get paginated applications - Simplified version
     */
    public Page<JobApplication> getApplicationsPaginated(Long jobId, ApplicationStatus status,
                                                         int page, int size, String sortBy, String direction) {
        Sort sort = direction.equalsIgnoreCase("desc") ?
                Sort.by(sortBy).descending() : Sort.by(sortBy).ascending();
        Pageable pageable = PageRequest.of(page, size, sort);

        // Since we don't have the custom method, get all and convert to page (simplified)
        List<JobApplication> allApps = applicationRepository.findByJobOrderByMatchScoreDesc(jobId);
        int start = (int) pageable.getOffset();
        int end = Math.min((start + pageable.getPageSize()), allApps.size());

        return new org.springframework.data.domain.PageImpl<>(
                allApps.subList(start, end), pageable, allApps.size()
        );
    }

    /**
     * Get applications by candidate
     */
    public List<JobApplication> getApplicationsByCandidate(Long candidateId) {
        User candidate = userRepository.findById(candidateId)
                .orElseThrow(() -> new RuntimeException("Candidate not found with id: " + candidateId));
        return applicationRepository.findByCandidateOrderByAppliedDateDesc(candidateId);
    }

    /**
     * Get applications for job
     */
    public List<JobApplication> getApplicationsForJob(Long jobId) {
        return applicationRepository.findByJobOrderByMatchScoreDesc(jobId);
    }

    // ==================== STATUS MANAGEMENT ====================

    /**
     * Save application directly (used for Q&A JSON persistence)
     */
    @Transactional
    public JobApplication saveApplication(JobApplication application) {
        return applicationRepository.save(application);
    }

    /**
     * Update application status
     */
    @Transactional
    public JobApplication updateApplicationStatus(ApplicationStatusUpdateDTO updateDTO) {
        JobApplication application = applicationRepository.findById(updateDTO.getApplicationId())
                .orElseThrow(() -> new RuntimeException("Application not found with id: " + updateDTO.getApplicationId()));

        ApplicationStatus oldStatus = application.getStatus();
        ApplicationStatus newStatus = updateDTO.getStatus();

        application.setStatus(newStatus);
        application.setStatusUpdatedDate(LocalDateTime.now());

        if (updateDTO.getNotes() != null) {
            // Append to existing notes rather than replace
            String existingNotes = application.getHrNotes();
            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("dd MMM yyyy HH:mm"));
            String newNote = String.format("[%s] Status changed to %s: %s", timestamp, newStatus.getDisplayName(), updateDTO.getNotes());
            application.setHrNotes(existingNotes != null ? existingNotes + "\n" + newNote : newNote);
        }

        // Handle special status transitions
        if (newStatus == ApplicationStatus.SHORTLISTED) {
            application.setShortlistedDate(LocalDateTime.now());
            application.setShortlistedBy(getCurrentUser());
        } else if (newStatus == ApplicationStatus.REJECTED) {
            application.setRejectionReason(updateDTO.getFeedback());
        } else if (newStatus == ApplicationStatus.HIRED) {
            application.setHiringDecisionDate(LocalDateTime.now());
            application.setHiringDecisionBy(getCurrentUser());
        } else if (newStatus == ApplicationStatus.WITHDRAWN) {
            application.setWithdrawalReason(updateDTO.getFeedback());
        }

        // Send notification if requested
        if (updateDTO.getNotifyCandidate() != null && updateDTO.getNotifyCandidate()) {
            sendStatusChangeNotification(application, oldStatus, newStatus);
        }

        return applicationRepository.save(application);
    }

    /**
     * Bulk update application status
     */
    @Transactional
    public int bulkUpdateStatus(List<Long> applicationIds, ApplicationStatus status, String notes) {
        int updated = 0;

        for (Long id : applicationIds) {
            try {
                JobApplication app = applicationRepository.findById(id).orElse(null);
                if (app != null) {
                    app.setStatus(status);
                    app.setStatusUpdatedDate(LocalDateTime.now());

                    if (notes != null && !notes.isEmpty()) {
                        String existingNotes = app.getHrNotes();
                        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("dd MMM yyyy HH:mm"));
                        String newNote = String.format("[%s] Bulk update to %s: %s", timestamp, status.getDisplayName(), notes);
                        app.setHrNotes(existingNotes != null ? existingNotes + "\n" + newNote : newNote);
                    }

                    applicationRepository.save(app);
                    updated++;

                    // Send notification
                    sendStatusChangeNotification(app, app.getStatus(), status);
                }
            } catch (Exception e) {
                System.err.println("Failed to update application " + id + ": " + e.getMessage());
            }
        }

        return updated;
    }

    // ==================== INTERVIEW MANAGEMENT ====================

    /**
     * Schedule interview
     */
    @Transactional
    public JobApplication scheduleInterview(InterviewScheduleDTO scheduleDTO) {
        JobApplication application = applicationRepository.findById(scheduleDTO.getApplicationId())
                .orElseThrow(() -> new RuntimeException("Application not found with id: " + scheduleDTO.getApplicationId()));

        application.setInterviewScheduled(true);
        application.setInterviewDate(scheduleDTO.getInterviewDateTime());
        application.setInterviewType(scheduleDTO.getInterviewType());
        application.setInterviewLocation(scheduleDTO.getInterviewLocation());
        application.setInterviewLink(scheduleDTO.getInterviewLink());
        application.setInterviewerName(scheduleDTO.getInterviewerName());
        application.setInterviewerEmail(scheduleDTO.getInterviewerEmail());
        application.setStatus(ApplicationStatus.INTERVIEW_SCHEDULED);
        application.setStatusUpdatedDate(LocalDateTime.now());

        // Save interview instructions in notes
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("dd MMM yyyy HH:mm"));
        String interviewNotes = String.format(
                "[%s] Interview scheduled for %s\nType: %s\n%s\nInstructions: %s",
                timestamp,
                scheduleDTO.getInterviewDateTime().format(DateTimeFormatter.ofPattern("dd MMM yyyy HH:mm")),
                scheduleDTO.getInterviewType(),
                scheduleDTO.getInterviewLocation() != null ? "Location: " + scheduleDTO.getInterviewLocation() : "Link: " + scheduleDTO.getInterviewLink(),
                scheduleDTO.getInstructions() != null ? scheduleDTO.getInstructions() : "None"
        );

        String existingNotes = application.getHrNotes();
        application.setHrNotes(existingNotes != null ? existingNotes + "\n" + interviewNotes : interviewNotes);

        // Send interview invitation
        sendInterviewInvitation(application, scheduleDTO);

        return applicationRepository.save(application);
    }

    /**
     * Reschedule interview
     */
    @Transactional
    public JobApplication rescheduleInterview(Long applicationId, LocalDateTime newDateTime, String reason) {
        JobApplication application = applicationRepository.findById(applicationId)
                .orElseThrow(() -> new RuntimeException("Application not found with id: " + applicationId));

        LocalDateTime oldDate = application.getInterviewDate();
        application.setInterviewDate(newDateTime);

        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("dd MMM yyyy HH:mm"));
        String rescheduleNote = String.format(
                "[%s] Interview rescheduled from %s to %s. Reason: %s",
                timestamp,
                oldDate != null ? oldDate.format(DateTimeFormatter.ofPattern("dd MMM yyyy HH:mm")) : "unknown",
                newDateTime.format(DateTimeFormatter.ofPattern("dd MMM yyyy HH:mm")),
                reason
        );

        String existingNotes = application.getHrNotes();
        application.setHrNotes(existingNotes != null ? existingNotes + "\n" + rescheduleNote : rescheduleNote);

        // Send reschedule notification
        sendInterviewRescheduleNotification(application, oldDate, newDateTime, reason);

        return applicationRepository.save(application);
    }

    /**
     * Cancel interview
     */
    @Transactional
    public JobApplication cancelInterview(Long applicationId, String reason) {
        JobApplication application = applicationRepository.findById(applicationId)
                .orElseThrow(() -> new RuntimeException("Application not found with id: " + applicationId));

        application.setInterviewScheduled(false);
        application.setInterviewDate(null);
        application.setStatus(ApplicationStatus.SHORTLISTED);

        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("dd MMM yyyy HH:mm"));
        String cancelNote = String.format("[%s] Interview cancelled. Reason: %s", timestamp, reason);

        String existingNotes = application.getHrNotes();
        application.setHrNotes(existingNotes != null ? existingNotes + "\n" + cancelNote : cancelNote);

        // Send cancellation notification
        sendInterviewCancellationNotification(application, reason);

        return applicationRepository.save(application);
    }

    /**
     * Add interview feedback
     */
    @Transactional
    public JobApplication addInterviewFeedback(Long applicationId, String feedback, Integer rating) {
        JobApplication application = applicationRepository.findById(applicationId)
                .orElseThrow(() -> new RuntimeException("Application not found with id: " + applicationId));

        application.setInterviewFeedback(feedback);
        application.setInterviewRating(rating);

        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("dd MMM yyyy HH:mm"));
        String feedbackNote = String.format("[%s] Interview feedback added. Rating: %d/5", timestamp, rating != null ? rating : 0);

        String existingNotes = application.getHrNotes();
        application.setHrNotes(existingNotes != null ? existingNotes + "\n" + feedbackNote : feedbackNote);

        return applicationRepository.save(application);
    }

    /**
     * Get upcoming interviews - Custom implementation
     */
    public List<JobApplication> getUpcomingInterviews() {
        // Get all applications and filter manually
        List<JobApplication> allApps = applicationRepository.findAll();
        LocalDateTime now = LocalDateTime.now();

        return allApps.stream()
                .filter(app -> app.isActive() == null || app.isActive())
                .filter(JobApplication::isInterviewScheduled)
                .filter(app -> app.getInterviewDate() != null && app.getInterviewDate().isAfter(now))
                .sorted(Comparator.comparing(JobApplication::getInterviewDate))
                .toList();
    }

    /**
     * Get interviews by date range - Custom implementation
     */
    public List<JobApplication> getInterviewsInDateRange(LocalDateTime startDate, LocalDateTime endDate) {
        List<JobApplication> allApps = applicationRepository.findAll();

        return allApps.stream()
                .filter(app -> app.isActive() == null || app.isActive())
                .filter(JobApplication::isInterviewScheduled)
                .filter(app -> app.getInterviewDate() != null &&
                        !app.getInterviewDate().isBefore(startDate) &&
                        !app.getInterviewDate().isAfter(endDate))
                .sorted(Comparator.comparing(JobApplication::getInterviewDate))
                .toList();
    }

    // ==================== SHORTLIST MANAGEMENT ====================

    /**
     * Shortlist candidates
     */
    @Transactional
    public List<JobApplication> shortlistCandidates(Long jobId, List<Long> candidateIds, String notes) {
        List<JobApplication> shortlisted = new ArrayList<>();

        for (Long candidateId : candidateIds) {
            Job job = jobRepository.getReferenceById(jobId);
            User candidate = userRepository.getReferenceById(candidateId);

            Optional<JobApplication> appOpt = applicationRepository.findByJobAndCandidate(job, candidate);

            if (appOpt.isPresent()) {
                JobApplication app = appOpt.get();
                app.setStatus(ApplicationStatus.SHORTLISTED);
                app.setShortlistedDate(LocalDateTime.now());
                app.setShortlistedBy(getCurrentUser());

                String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("dd MMM yyyy HH:mm"));
                String shortlistNote = String.format("[%s] Shortlisted. Notes: %s", timestamp, notes != null ? notes : "None");

                String existingNotes = app.getHrNotes();
                app.setHrNotes(existingNotes != null ? existingNotes + "\n" + shortlistNote : shortlistNote);

                shortlisted.add(applicationRepository.save(app));

                // Send notification
                sendShortlistNotification(app);
            }
        }

        return shortlisted;
    }

    /**
     * Get shortlisted candidates for job - Custom implementation
     */
    public List<JobApplication> getShortlistedForJob(Long jobId) {
        List<JobApplication> allApps = applicationRepository.findByJobOrderByMatchScoreDesc(jobId);

        return allApps.stream()
                .filter(app -> app.getShortlistedDate() != null || app.getStatus() == ApplicationStatus.SHORTLISTED)
                .sorted((a, b) -> {
                    if (a.getShortlistedDate() == null) return 1;
                    if (b.getShortlistedDate() == null) return -1;
                    return b.getShortlistedDate().compareTo(a.getShortlistedDate());
                })
                .toList();
    }

    // ==================== REJECTION MANAGEMENT ====================

    /**
     * Remove rejected applications (soft delete)
     */
    @Transactional
    public int removeRejectedApplications(Long jobId) {
        List<JobApplication> allApps = applicationRepository.findByJobOrderByMatchScoreDesc(jobId);
        int removed = 0;

        for (JobApplication app : allApps) {
            if (app.getStatus() == ApplicationStatus.REJECTED) {
                app.setActive(false);
                app.setArchived(true);
                app.setArchiveDate(LocalDateTime.now());
                applicationRepository.save(app);
                removed++;
            }
        }

        return removed;
    }

    /**
     * Archive old applications
     */
    @Transactional
    public int archiveOldApplications(int daysOld) {
        LocalDateTime thresholdDate = LocalDateTime.now().minusDays(daysOld);
        List<JobApplication> allApps = applicationRepository.findAll();
        int archived = 0;

        for (JobApplication app : allApps) {
            if (app.getStatus() == ApplicationStatus.REJECTED ||
                    app.getStatus() == ApplicationStatus.WITHDRAWN ||
                    (app.getStatus() == ApplicationStatus.HIRED && app.getHiringDecisionDate() != null &&
                            app.getHiringDecisionDate().isBefore(thresholdDate))) {

                if (!app.isArchived()) {
                    app.setArchived(true);
                    app.setArchiveDate(LocalDateTime.now());
                    applicationRepository.save(app);
                    archived++;
                }
            }
        }

        return archived;
    }

    // ==================== NOTES MANAGEMENT ====================

    /**
     * Add note to application
     */
    @Transactional
    public JobApplication addNote(Long applicationId, String note, String noteType, boolean isPrivate) {
        JobApplication application = applicationRepository.findById(applicationId)
                .orElseThrow(() -> new RuntimeException("Application not found with id: " + applicationId));

        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("dd MMM yyyy HH:mm"));
        String privacy = isPrivate ? "[PRIVATE] " : "";
        String formattedNote = String.format("%s[%s] %s: %s", privacy, timestamp, noteType, note);

        String existingNotes = application.getHrNotes();
        application.setHrNotes(existingNotes != null ? existingNotes + "\n" + formattedNote : formattedNote);

        return applicationRepository.save(application);
    }

    // ==================== STATISTICS AND ANALYTICS ====================

    /**
     * Get application statistics for a job
     */
    public Map<String, Object> getApplicationStatistics(Long jobId) {
        Map<String, Object> stats = new HashMap<>();

        List<JobApplication> applications = applicationRepository.findByJobOrderByMatchScoreDesc(jobId);

        // Status counts
        Map<String, Long> statusMap = new HashMap<>();
        for (ApplicationStatus status : ApplicationStatus.values()) {
            long count = applications.stream()
                    .filter(app -> app.getStatus() == status)
                    .count();
            statusMap.put(status.name(), count);
        }
        stats.put("statusCounts", statusMap);

        // Total applications
        stats.put("totalApplications", (long) applications.size());

        // Average match score
        Double avgScore = applications.stream()
                .filter(app -> app.getMatchScore() != null)
                .mapToDouble(JobApplication::getMatchScore)
                .average()
                .orElse(0.0);
        stats.put("averageMatchScore", avgScore);

        // Average score by status
        Map<String, Double> scoreMap = new HashMap<>();
        for (ApplicationStatus status : ApplicationStatus.values()) {
            Double avgStatusScore = applications.stream()
                    .filter(app -> app.getStatus() == status && app.getMatchScore() != null)
                    .mapToDouble(JobApplication::getMatchScore)
                    .average()
                    .orElse(0.0);
            scoreMap.put(status.name(), avgStatusScore);
        }
        stats.put("averageScoreByStatus", scoreMap);

        // Shortlist stats
        long shortlisted = applications.stream()
                .filter(app -> app.getShortlistedDate() != null || app.getStatus() == ApplicationStatus.SHORTLISTED)
                .count();
        stats.put("shortlisted", shortlisted);

        // Interview stats
        long interviewed = applications.stream()
                .filter(JobApplication::isInterviewScheduled)
                .count();
        stats.put("interviewed", interviewed);

        // Hired stats
        long hired = applications.stream()
                .filter(app -> app.getStatus() == ApplicationStatus.HIRED)
                .count();
        stats.put("hired", hired);

        // Rejected stats
        long rejected = applications.stream()
                .filter(app -> app.getStatus() == ApplicationStatus.REJECTED)
                .count();
        stats.put("rejected", rejected);

        return stats;
    }

    /**
     * Get application trends
     */
    public Map<String, Long> getApplicationTrends(Long jobId, int days) {
        LocalDateTime endDate = LocalDateTime.now();
        LocalDateTime startDate = endDate.minusDays(days);

        List<JobApplication> applications = applicationRepository.findByJobOrderByMatchScoreDesc(jobId);

        Map<String, Long> trendMap = new LinkedHashMap<>();

        for (int i = 0; i <= days; i++) {
            LocalDateTime date = startDate.plusDays(i);
            String dateKey = date.toLocalDate().toString();

            long count = applications.stream()
                    .filter(app -> !app.getAppliedDate().isBefore(date) &&
                            !app.getAppliedDate().isAfter(date.plusDays(1)))
                    .count();

            trendMap.put(dateKey, count);
        }

        return trendMap;
    }

    /**
     * Get job-wise application counts
     */
    public Map<String, Long> getJobWiseApplicationCounts() {
        List<Job> allJobs = jobRepository.findAll();
        Map<String, Long> jobCountMap = new LinkedHashMap<>();

        for (Job job : allJobs) {
            long count = applicationRepository.countByJobId(job.getId());
            jobCountMap.put(job.getTitle() + " (ID: " + job.getId() + ")", count);
        }

        return jobCountMap;
    }

    // ==================== FILE OPERATIONS ====================

    /**
     * Get application resume file
     */
    public Resource getApplicationResumeFile(Long applicationId) throws MalformedURLException {
        JobApplication application = applicationRepository.findById(applicationId)
                .orElseThrow(() -> new RuntimeException("Application not found with id: " + applicationId));

        if (application.getResumePath() == null) {
            throw new RuntimeException("No resume found for this application");
        }

        Path filePath = Paths.get(application.getResumePath());
        Resource resource = new UrlResource(filePath.toUri());

        if (resource.exists() && resource.isReadable()) {
            return resource;
        } else {
            throw new RuntimeException("Could not read the resume file at: " + application.getResumePath());
        }
    }

    // ==================== HELPER METHODS ====================

    /**
     * Get current username
     */
    private String getCurrentUser() {
        org.springframework.security.core.Authentication auth =
                org.springframework.security.core.context.SecurityContextHolder.getContext().getAuthentication();
        return auth != null ? auth.getName() : "System";
    }

    /**
     * Send status change notification email
     */
    private void sendStatusChangeNotification(JobApplication application, ApplicationStatus oldStatus, ApplicationStatus newStatus) {
        if (mailSender == null) {
            System.err.println("Mail sender not configured. Skipping email notification.");
            return;
        }

        try {
            SimpleMailMessage message = new SimpleMailMessage();
            message.setTo(application.getCandidate().getEmail());
            message.setSubject("Application Status Update - " + application.getJob().getTitle());

            String body = String.format(
                    "Dear %s,\n\n" +
                            "Your application for the position '%s' has been updated.\n\n" +
                            "Previous Status: %s\n" +
                            "New Status: %s\n\n" +
                            "You can view your application status at: %s/candidate/my-applications\n\n" +
                            "Thank you for your interest in our company.\n\n" +
                            "Best regards,\nRecruitment Team",
                    application.getCandidate().getFullName(),
                    application.getJob().getTitle(),
                    oldStatus != null ? oldStatus.getDisplayName() : "Unknown",
                    newStatus.getDisplayName(),
                    baseUrl
            );

            message.setText(body);
            mailSender.send(message);
        } catch (Exception e) {
            System.err.println("Failed to send email: " + e.getMessage());
        }
    }

    /**
     * Send interview invitation email
     */
    private void sendInterviewInvitation(JobApplication application, InterviewScheduleDTO schedule) {
        if (mailSender == null) return;

        try {
            SimpleMailMessage message = new SimpleMailMessage();
            message.setTo(application.getCandidate().getEmail());
            if (schedule.getInterviewerEmail() != null && !schedule.getInterviewerEmail().isEmpty()) {
                message.setCc(schedule.getInterviewerEmail());
            }

            message.setSubject("Interview Invitation - " + application.getJob().getTitle());

            String dateTimeStr = schedule.getInterviewDateTime()
                    .format(DateTimeFormatter.ofPattern("dd MMMM yyyy 'at' hh:mm a"));

            String locationStr = "ONLINE".equals(schedule.getInterviewType()) ?
                    "Online Meeting: " + (schedule.getInterviewLink() != null ? schedule.getInterviewLink() : "TBD") :
                    "Location: " + (schedule.getInterviewLocation() != null ? schedule.getInterviewLocation() : "TBD");

            String body = String.format(
                    "Dear %s,\n\n" +
                            "We are pleased to invite you for an interview for the position '%s'.\n\n" +
                            "Interview Details:\n" +
                            "Date & Time: %s\n" +
                            "Type: %s\n" +
                            "%s\n" +
                            "Interviewer: %s\n" +
                            "Duration: %s\n\n" +
                            "Instructions:\n%s\n\n" +
                            "Please confirm your availability by replying to this email.\n\n" +
                            "Best regards,\nRecruitment Team",
                    application.getCandidate().getFullName(),
                    application.getJob().getTitle(),
                    dateTimeStr,
                    schedule.getInterviewType(),
                    locationStr,
                    schedule.getInterviewerName() != null ? schedule.getInterviewerName() : "TBD",
                    schedule.getInterviewDuration() != null ? schedule.getInterviewDuration() : "60 minutes",
                    schedule.getInstructions() != null ? schedule.getInstructions() : "None"
            );

            message.setText(body);
            mailSender.send(message);
        } catch (Exception e) {
            System.err.println("Failed to send interview invitation: " + e.getMessage());
        }
    }

    /**
     * Send interview reschedule notification
     */
    private void sendInterviewRescheduleNotification(JobApplication application, LocalDateTime oldDate,
                                                     LocalDateTime newDate, String reason) {
        if (mailSender == null) return;

        try {
            SimpleMailMessage message = new SimpleMailMessage();
            message.setTo(application.getCandidate().getEmail());
            message.setSubject("Interview Rescheduled - " + application.getJob().getTitle());

            String oldDateStr = oldDate != null ?
                    oldDate.format(DateTimeFormatter.ofPattern("dd MMM yyyy HH:mm")) : "unknown";
            String newDateStr = newDate.format(DateTimeFormatter.ofPattern("dd MMM yyyy HH:mm"));

            String body = String.format(
                    "Dear %s,\n\n" +
                            "Your interview for '%s' has been rescheduled.\n\n" +
                            "Previous Date & Time: %s\n" +
                            "New Date & Time: %s\n" +
                            "Reason: %s\n\n" +
                            "Please let us know if this works for you.\n\n" +
                            "Best regards,\nRecruitment Team",
                    application.getCandidate().getFullName(),
                    application.getJob().getTitle(),
                    oldDateStr,
                    newDateStr,
                    reason != null ? reason : "Not specified"
            );

            message.setText(body);
            mailSender.send(message);
        } catch (Exception e) {
            System.err.println("Failed to send reschedule notification: " + e.getMessage());
        }
    }

    /**
     * Send interview cancellation notification
     */
    private void sendInterviewCancellationNotification(JobApplication application, String reason) {
        if (mailSender == null) return;

        try {
            SimpleMailMessage message = new SimpleMailMessage();
            message.setTo(application.getCandidate().getEmail());
            message.setSubject("Interview Cancelled - " + application.getJob().getTitle());

            String body = String.format(
                    "Dear %s,\n\n" +
                            "We regret to inform you that your interview for '%s' has been cancelled.\n\n" +
                            "Reason: %s\n\n" +
                            "We will contact you shortly with further updates.\n\n" +
                            "Best regards,\nRecruitment Team",
                    application.getCandidate().getFullName(),
                    application.getJob().getTitle(),
                    reason != null ? reason : "Not specified"
            );

            message.setText(body);
            mailSender.send(message);
        } catch (Exception e) {
            System.err.println("Failed to send cancellation notification: " + e.getMessage());
        }
    }

    /**
     * Send shortlist notification
     */
    private void sendShortlistNotification(JobApplication application) {
        if (mailSender == null) return;

        try {
            SimpleMailMessage message = new SimpleMailMessage();
            message.setTo(application.getCandidate().getEmail());
            message.setSubject("Congratulations! You've been shortlisted - " + application.getJob().getTitle());

            String body = String.format(
                    "Dear %s,\n\n" +
                            "Congratulations! Based on your application and skills, you have been shortlisted for the position '%s'.\n\n" +
                            "Our recruitment team will contact you shortly to schedule the next steps.\n\n" +
                            "Best regards,\nRecruitment Team",
                    application.getCandidate().getFullName(),
                    application.getJob().getTitle()
            );

            message.setText(body);
            mailSender.send(message);
        } catch (Exception e) {
            System.err.println("Failed to send shortlist notification: " + e.getMessage());
        }
    }

    /**
     * Send rejection email
     */
    private void sendRejectionEmail(JobApplication application, String feedback) {
        if (mailSender == null) return;

        try {
            SimpleMailMessage message = new SimpleMailMessage();
            message.setTo(application.getCandidate().getEmail());
            message.setSubject("Update on your application - " + application.getJob().getTitle());

            String body = String.format(
                    "Dear %s,\n\n" +
                            "Thank you for your interest in the '%s' position and for taking the time to apply.\n\n" +
                            "After careful review of your application, we regret to inform you that we have decided to move forward with other candidates whose qualifications more closely match our current requirements.\n\n" +
                            "%s\n\n" +
                            "We appreciate your interest in our company and wish you success in your job search.\n\n" +
                            "Best regards,\nRecruitment Team",
                    application.getCandidate().getFullName(),
                    application.getJob().getTitle(),
                    feedback != null ? "Feedback: " + feedback : "We encourage you to apply for future positions that match your profile."
            );

            message.setText(body);
            mailSender.send(message);
        } catch (Exception e) {
            System.err.println("Failed to send rejection email: " + e.getMessage());
        }
    }
}