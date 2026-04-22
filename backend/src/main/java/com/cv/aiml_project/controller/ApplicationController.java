package com.cv.aiml_project.controller;

import com.cv.aiml_project.dto.ApplicationFilterDTO;
import com.cv.aiml_project.dto.ApplicationStatusUpdateDTO;
import com.cv.aiml_project.dto.InterviewScheduleDTO;
import com.cv.aiml_project.entity.ApplicationStatus;
import com.cv.aiml_project.entity.JobApplication;
import com.cv.aiml_project.repository.JobApplicationRepository;
import com.cv.aiml_project.service.ApplicationService;
import com.cv.aiml_project.service.JobService;
import com.cv.aiml_project.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.data.domain.Page;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@Controller
@RequestMapping("/applications")
public class ApplicationController {

    @Autowired
    private ApplicationService applicationService;

    @Autowired
    private JobService jobService;

    @Autowired
    private UserService userService;

    @Autowired
    private JobApplicationRepository jobApplicationRepository;

    // ==================== MAIN VIEWS ====================

    /**
     * Applications dashboard
     */
    @GetMapping("/dashboard")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String applicationsDashboard(Model model) {
        model.addAttribute("jobWiseCounts", applicationService.getJobWiseApplicationCounts());
        model.addAttribute("upcomingInterviews", applicationService.getUpcomingInterviews());
        return "applications/dashboard";
    }

    /**
     * View applications for a job
     */
    @GetMapping("/job/{jobId}")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String viewJobApplications(@PathVariable Long jobId,
                                      @RequestParam(required = false) ApplicationStatus status,
                                      @RequestParam(required = false) String search,
                                      @RequestParam(defaultValue = "1") int page,
                                      @RequestParam(defaultValue = "20") int size,
                                      Model model) {

        var job = jobService.getJobById(jobId)
                .orElseThrow(() -> new IllegalArgumentException("Job not found"));

        ApplicationFilterDTO filters = new ApplicationFilterDTO();
        filters.setJobId(jobId);
        filters.setStatus(status);
        filters.setSearchKeyword(search);

        List<JobApplication> applications = applicationService.getApplicationsWithFilters(filters);
        Map<String, Object> stats = applicationService.getApplicationStatistics(jobId);

        model.addAttribute("job", job);
        model.addAttribute("applications", applications);
        model.addAttribute("stats", stats);
        model.addAttribute("statuses", ApplicationStatus.values());
        model.addAttribute("currentStatus", status);
        model.addAttribute("search", search);

        return "applications/job-applications";
    }

    /**
     * View application details
     */
    @GetMapping("/{applicationId}")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN', 'CANDIDATE')")
    public String viewApplicationDetails(@PathVariable Long applicationId, Model model) {
        JobApplication application = applicationService.getApplicationById(applicationId)
                .orElseThrow(() -> new IllegalArgumentException("Application not found"));

        // Check permissions for candidate view
        if (hasRole("CANDIDATE")) {
            org.springframework.security.core.Authentication auth =
                    org.springframework.security.core.context.SecurityContextHolder.getContext().getAuthentication();
            var currentUser = userService.getUserByUsername(auth.getName()).orElse(null);
            if (currentUser == null || !application.getCandidate().getId().equals(currentUser.getId())) {
                throw new RuntimeException("Access denied");
            }
        }

        model.addAttribute("application", application);
        model.addAttribute("candidate", application.getCandidate());
        model.addAttribute("job", application.getJob());
        model.addAttribute("statuses", ApplicationStatus.values());

        return "applications/application-details";
    }

    /**
     * View candidate's own applications
     */
    @GetMapping("/my-applications")
    @PreAuthorize("hasRole('CANDIDATE')")
    public String viewMyApplications(Model model) {
        org.springframework.security.core.Authentication auth =
                org.springframework.security.core.context.SecurityContextHolder.getContext().getAuthentication();
        var currentUser = userService.getUserByUsername(auth.getName())
                .orElseThrow(() -> new RuntimeException("User not found"));

        List<JobApplication> applications = applicationService.getApplicationsByCandidate(currentUser.getId());

        model.addAttribute("applications", applications);
        return "applications/my-applications";
    }

    /**
     * List all applications with generated interview questions (for candidate)
     */
    @GetMapping("/my-interviews")
    @PreAuthorize("hasRole('CANDIDATE')")
    public String viewMyInterviews(Model model) {
        org.springframework.security.core.Authentication auth =
                org.springframework.security.core.context.SecurityContextHolder.getContext().getAuthentication();
        var currentUser = userService.getUserByUsername(auth.getName())
                .orElseThrow(() -> new RuntimeException("User not found"));

        List<JobApplication> appsWithQuestions = jobApplicationRepository.findApplicationsWithQuestionsByCandidate(currentUser.getId());
        model.addAttribute("applications", appsWithQuestions);
        return "candidate/interviews";
    }

    /**
     * Show interview Q&A form for a specific application
     */
    @GetMapping("/{applicationId}/interview")
    @PreAuthorize("hasRole('CANDIDATE')")
    public String viewInterviewForm(@PathVariable Long applicationId, Model model) {
        org.springframework.security.core.Authentication auth =
                org.springframework.security.core.context.SecurityContextHolder.getContext().getAuthentication();
        var currentUser = userService.getUserByUsername(auth.getName())
                .orElseThrow(() -> new RuntimeException("User not found"));

        JobApplication application = applicationService.getApplicationById(applicationId)
                .orElseThrow(() -> new IllegalArgumentException("Application not found"));

        if (!application.getCandidate().getId().equals(currentUser.getId())) {
            throw new RuntimeException("Access denied");
        }
        if (!application.hasInterviewQuestions()) {
            return "redirect:/applications/my-interviews";
        }

        // Parse questions JSON into a list for the template
        try {
            com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
            java.util.List<?> questions = mapper.readValue(application.getInterviewQuestionsJson(), java.util.List.class);
            model.addAttribute("questions", questions);

            if (application.hasInterviewAnswers()) {
                java.util.List<?> answers = mapper.readValue(application.getInterviewAnswersJson(), java.util.List.class);
                model.addAttribute("answers", answers);
            }
        } catch (Exception e) {
            model.addAttribute("parseError", "Could not load questions. Please contact HR.");
        }

        model.addAttribute("application", application);
        model.addAttribute("job", application.getJob());
        return "candidate/interview-answer";
    }

    /**
     * Submit candidate's interview answers
     */
    @PostMapping("/{applicationId}/interview/submit")
    @PreAuthorize("hasRole('CANDIDATE')")
    public String submitInterviewAnswers(@PathVariable Long applicationId,
                                         @RequestParam java.util.Map<String, String> params,
                                         RedirectAttributes redirectAttributes) {
        org.springframework.security.core.Authentication auth =
                org.springframework.security.core.context.SecurityContextHolder.getContext().getAuthentication();
        var currentUser = userService.getUserByUsername(auth.getName())
                .orElseThrow(() -> new RuntimeException("User not found"));

        JobApplication application = applicationService.getApplicationById(applicationId)
                .orElseThrow(() -> new IllegalArgumentException("Application not found"));

        if (!application.getCandidate().getId().equals(currentUser.getId())) {
            throw new RuntimeException("Access denied");
        }

        try {
            com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
            java.util.List<?> questions = mapper.readValue(application.getInterviewQuestionsJson(), java.util.List.class);

            // Build answers list matching question indices
            java.util.List<java.util.Map<String, String>> answers = new java.util.ArrayList<>();
            for (int i = 0; i < questions.size(); i++) {
                java.util.Map<String, String> answerEntry = new java.util.LinkedHashMap<>();
                answerEntry.put("answer", params.getOrDefault("answer_" + i, ""));
                answers.add(answerEntry);
            }

            application.setInterviewAnswersJson(mapper.writeValueAsString(answers));
            application.setAnswersSubmittedAt(java.time.LocalDateTime.now());
            applicationService.saveApplication(application);

            redirectAttributes.addFlashAttribute("message", "Your answers have been submitted successfully!");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to save answers: " + e.getMessage());
        }
        return "redirect:/applications/my-interviews";
    }

    // ==================== STATUS MANAGEMENT ====================

    /**
     * Update application status
     */
    @PostMapping("/{applicationId}/status")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String updateApplicationStatus(@PathVariable Long applicationId,
                                          @RequestParam ApplicationStatus status,
                                          @RequestParam(required = false) String notes,
                                          @RequestParam(required = false) Boolean notify,
                                          RedirectAttributes redirectAttributes) {
        try {
            ApplicationStatusUpdateDTO updateDTO = new ApplicationStatusUpdateDTO();
            updateDTO.setApplicationId(applicationId);
            updateDTO.setStatus(status);
            updateDTO.setNotes(notes);
            updateDTO.setNotifyCandidate(notify != null ? notify : true);

            JobApplication application = applicationService.updateApplicationStatus(updateDTO);

            redirectAttributes.addFlashAttribute("message",
                    "Application status updated to " + status.getDisplayName());
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to update status: " + e.getMessage());
        }
        return "redirect:/applications/" + applicationId;
    }

    /**
     * Bulk update status
     */
    @PostMapping("/bulk-status-update")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String bulkStatusUpdate(@RequestParam List<Long> applicationIds,
                                   @RequestParam ApplicationStatus status,
                                   @RequestParam(required = false) String notes,
                                   @RequestParam Long jobId,
                                   RedirectAttributes redirectAttributes) {
        try {
            int updated = applicationService.bulkUpdateStatus(applicationIds, status, notes);
            redirectAttributes.addFlashAttribute("message",
                    "Updated " + updated + " applications to " + status.getDisplayName());
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Bulk update failed: " + e.getMessage());
        }
        return "redirect:/applications/job/" + jobId;
    }

    // ==================== INTERVIEW MANAGEMENT ====================

    /**
     * Show schedule interview page
     */
    @GetMapping("/{applicationId}/schedule-interview")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String showScheduleInterview(@PathVariable Long applicationId, Model model) {
        JobApplication application = applicationService.getApplicationById(applicationId)
                .orElseThrow(() -> new IllegalArgumentException("Application not found"));

        model.addAttribute("application", application);
        model.addAttribute("applicationId", applicationId);
        model.addAttribute("jobId", application.getJob().getId()); // for back/cancel links
        model.addAttribute("candidate", application.getCandidate());
        model.addAttribute("job", application.getJob());
        model.addAttribute("interviewTypes", new String[]{"ONLINE", "PHONE", "IN_PERSON"});

        return "applications/schedule-interview";
    }

    /**
     * Schedule interview
     */
    @PostMapping("/{applicationId}/schedule-interview")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String scheduleInterview(@PathVariable Long applicationId,
                                    @ModelAttribute InterviewScheduleDTO scheduleDTO,
                                    @RequestParam(name = "interviewDateTime", required = false) String interviewDateTimeStr,
                                    @RequestParam(name = "jobId", required = false) Long jobId,
                                    RedirectAttributes redirectAttributes) {
        String redirectTarget = jobId != null
                ? "redirect:/jobs/" + jobId + "/applications"
                : "redirect:/applications/" + applicationId;
        try {
            scheduleDTO.setApplicationId(applicationId);

            // Parse the datetime string manually — bypasses Spring's @DateTimeFormat
            // Flatpickr sends "2026-04-01T10:30:00" (Y-m-dTH:i:S format)
            if (interviewDateTimeStr != null && !interviewDateTimeStr.isBlank()) {
                scheduleDTO.setInterviewDateTime(
                    java.time.LocalDateTime.parse(interviewDateTimeStr,
                        java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss")));
            } else {
                redirectAttributes.addFlashAttribute("error", "Interview date and time are required.");
                return "redirect:/applications/" + applicationId + "/schedule-interview";
            }

            JobApplication application = applicationService.scheduleInterview(scheduleDTO);
            redirectAttributes.addFlashAttribute("message",
                    "Interview scheduled successfully for " + application.getCandidate().getFullName());
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to schedule interview: " + e.getMessage());
        }
        return redirectTarget;
    }

    /**
     * Reschedule interview
     */
    @PostMapping("/{applicationId}/reschedule-interview")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String rescheduleInterview(@PathVariable Long applicationId,
                                      @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime newDateTime,
                                      @RequestParam String reason,
                                      RedirectAttributes redirectAttributes) {
        try {
            JobApplication application = applicationService.rescheduleInterview(applicationId, newDateTime, reason);
            redirectAttributes.addFlashAttribute("message", "Interview rescheduled successfully");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to reschedule: " + e.getMessage());
        }
        return "redirect:/applications/" + applicationId;
    }

    /**
     * Cancel interview
     */
    @PostMapping("/{applicationId}/cancel-interview")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String cancelInterview(@PathVariable Long applicationId,
                                  @RequestParam String reason,
                                  RedirectAttributes redirectAttributes) {
        try {
            JobApplication application = applicationService.cancelInterview(applicationId, reason);
            redirectAttributes.addFlashAttribute("message", "Interview cancelled");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to cancel interview: " + e.getMessage());
        }
        return "redirect:/applications/" + applicationId;
    }

    /**
     * Add interview feedback
     */
    @PostMapping("/{applicationId}/interview-feedback")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String addInterviewFeedback(@PathVariable Long applicationId,
                                       @RequestParam String feedback,
                                       @RequestParam(required = false) Integer rating,
                                       RedirectAttributes redirectAttributes) {
        try {
            JobApplication application = applicationService.addInterviewFeedback(applicationId, feedback, rating);
            redirectAttributes.addFlashAttribute("message", "Interview feedback added");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to add feedback: " + e.getMessage());
        }
        return "redirect:/applications/" + applicationId;
    }

    // ==================== SHORTLIST MANAGEMENT ====================

    /**
     * View shortlisted candidates
     */
    @GetMapping("/job/{jobId}/shortlisted")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String viewShortlisted(@PathVariable Long jobId, Model model) {
        var job = jobService.getJobById(jobId)
                .orElseThrow(() -> new IllegalArgumentException("Job not found"));

        List<JobApplication> shortlisted = applicationService.getShortlistedForJob(jobId);

        model.addAttribute("job", job);
        model.addAttribute("shortlisted", shortlisted);

        return "applications/shortlisted";
    }

    /**
     * Shortlist candidates (bulk)
     */
    @PostMapping("/job/{jobId}/shortlist")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String shortlistCandidates(@PathVariable Long jobId,
                                      @RequestParam List<Long> candidateIds,
                                      @RequestParam(required = false) String notes,
                                      RedirectAttributes redirectAttributes) {
        try {
            List<JobApplication> shortlisted = applicationService.shortlistCandidates(jobId, candidateIds, notes);
            redirectAttributes.addFlashAttribute("message",
                    shortlisted.size() + " candidates shortlisted successfully");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to shortlist candidates: " + e.getMessage());
        }
        return "redirect:/applications/job/" + jobId;
    }

    // ==================== NOTES MANAGEMENT ====================

    /**
     * Add note to application
     */
    @PostMapping("/{applicationId}/notes")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String addNote(@PathVariable Long applicationId,
                          @RequestParam String note,
                          @RequestParam(defaultValue = "GENERAL") String noteType,
                          @RequestParam(defaultValue = "false") boolean isPrivate,
                          RedirectAttributes redirectAttributes) {
        try {
            applicationService.addNote(applicationId, note, noteType, isPrivate);
            redirectAttributes.addFlashAttribute("message", "Note added successfully");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to add note: " + e.getMessage());
        }
        return "redirect:/applications/" + applicationId;
    }

    // ==================== REJECTION MANAGEMENT ====================

    /**
     * Remove rejected applications
     */
    @PostMapping("/job/{jobId}/remove-rejected")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String removeRejectedApplications(@PathVariable Long jobId,
                                             RedirectAttributes redirectAttributes) {
        try {
            int removed = applicationService.removeRejectedApplications(jobId);
            redirectAttributes.addFlashAttribute("message",
                    removed + " rejected applications removed");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to remove rejected applications: " + e.getMessage());
        }
        return "redirect:/applications/job/" + jobId;
    }

    // ==================== FILE OPERATIONS ====================

    /**
     * Download application resume
     */
    @GetMapping("/{applicationId}/resume/download")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN', 'CANDIDATE')")
    public ResponseEntity<Resource> downloadResume(@PathVariable Long applicationId) {
        try {
            Resource resource = applicationService.getApplicationResumeFile(applicationId);
            JobApplication application = applicationService.getApplicationById(applicationId).orElseThrow();

            return ResponseEntity.ok()
                    .contentType(MediaType.APPLICATION_OCTET_STREAM)
                    .header(HttpHeaders.CONTENT_DISPOSITION,
                            "attachment; filename=\"" + application.getResumeOriginalName() + "\"")
                    .body(resource);
        } catch (Exception e) {
            return ResponseEntity.notFound().build();
        }
    }

    /**
     * View application resume
     */
    @GetMapping("/{applicationId}/resume/view")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN', 'CANDIDATE')")
    public ResponseEntity<Resource> viewResume(@PathVariable Long applicationId) {
        try {
            Resource resource = applicationService.getApplicationResumeFile(applicationId);
            JobApplication application = applicationService.getApplicationById(applicationId).orElseThrow();

            return ResponseEntity.ok()
                    .contentType(MediaType.APPLICATION_PDF)
                    .header(HttpHeaders.CONTENT_DISPOSITION,
                            "inline; filename=\"" + application.getResumeOriginalName() + "\"")
                    .body(resource);
        } catch (Exception e) {
            return ResponseEntity.notFound().build();
        }
    }

    // ==================== STATISTICS API ====================

    /**
     * Get application statistics (JSON)
     */
    @GetMapping("/api/statistics/job/{jobId}")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    @ResponseBody
    public ResponseEntity<Map<String, Object>> getApplicationStatistics(@PathVariable Long jobId) {
        try {
            Map<String, Object> stats = applicationService.getApplicationStatistics(jobId);
            return ResponseEntity.ok(stats);
        } catch (Exception e) {
            return ResponseEntity.badRequest().build();
        }
    }

    /**
     * Get application trends (JSON)
     */
    @GetMapping("/api/trends/job/{jobId}")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    @ResponseBody
    public ResponseEntity<Map<String, Long>> getApplicationTrends(@PathVariable Long jobId,
                                                                  @RequestParam(defaultValue = "30") int days) {
        try {
            Map<String, Long> trends = applicationService.getApplicationTrends(jobId, days);
            return ResponseEntity.ok(trends);
        } catch (Exception e) {
            return ResponseEntity.badRequest().build();
        }
    }

    // ==================== HELPER METHODS ====================

    private boolean hasRole(String role) {
        org.springframework.security.core.Authentication auth =
                org.springframework.security.core.context.SecurityContextHolder.getContext().getAuthentication();
        return auth.getAuthorities().stream()
                .anyMatch(a -> a.getAuthority().equals("ROLE_" + role));
    }
    /**
     * Get applications for a specific job and candidate (for linking from other modules)
     */
    @GetMapping("/job/{jobId}/candidate/{candidateId}")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String getApplicationByJobAndCandidate(@PathVariable Long jobId,
                                                  @PathVariable Long candidateId) {
        // Find the application for this job and candidate
        List<JobApplication> applications = applicationService.getApplicationsForJob(jobId);

        Optional<JobApplication> application = applications.stream()
                .filter(app -> app.getCandidate().getId().equals(candidateId))
                .findFirst();

        if (application.isPresent()) {
            return "redirect:/applications/" + application.get().getId();
        } else {
            return "redirect:/applications/job/" + jobId;
        }
    }

    /**
     * Get upcoming interviews view
     */
    @GetMapping("/upcoming-interviews")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String upcomingInterviews(Model model) {
        List<JobApplication> interviews = applicationService.getUpcomingInterviews();
        model.addAttribute("interviews", interviews);
        return "applications/upcoming-interviews";
    }

    /**
     * Get applications by status (for quick filters)
     */
    @GetMapping("")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String getApplicationsByStatus(@RequestParam(required = false) String status,
                                          Model model) {
        ApplicationFilterDTO filters = new ApplicationFilterDTO();
        if (status != null) {
            try {
                filters.setStatus(ApplicationStatus.valueOf(status));
            } catch (IllegalArgumentException e) {
                // Ignore invalid status
            }
        }

        List<JobApplication> applications = applicationService.getApplicationsWithFilters(filters);
        model.addAttribute("applications", applications);
        model.addAttribute("currentStatus", status);
        return "applications/list";
    }

    /**
     * Export applications for a job
     */
    @GetMapping("/job/{jobId}/export")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public ResponseEntity<String> exportApplications(@PathVariable Long jobId,
                                                     @RequestParam(required = false) String status,
                                                     @RequestParam(required = false) String search) {
        // Implementation for CSV export
        List<JobApplication> applications = applicationService.getApplicationsForJob(jobId);

        StringBuilder csv = new StringBuilder();
        csv.append("Candidate Name,Email,Phone,Applied Date,Status,Match Score,Skills Score,Experience Score,Education Score,Interview Scheduled\n");

        for (JobApplication app : applications) {
            csv.append("\"").append(app.getCandidate().getFullName()).append("\",")
                    .append("\"").append(app.getCandidate().getEmail()).append("\",")
                    .append("\"").append(app.getCandidate().getPhone() != null ? app.getCandidate().getPhone() : "").append("\",")
                    .append(app.getAppliedDate().toString()).append(",")
                    .append(app.getStatus().toString()).append(",")
                    .append(app.getMatchScore() != null ? app.getMatchScore() : "").append(",")
                    .append(app.getSkillsMatchScore() != null ? app.getSkillsMatchScore() : "").append(",")
                    .append(app.getExperienceMatchScore() != null ? app.getExperienceMatchScore() : "").append(",")
                    .append(app.getEducationMatchScore() != null ? app.getEducationMatchScore() : "").append(",")
                    .append(app.isInterviewScheduled() ? "Yes" : "No")
                    .append("\n");
        }

        return ResponseEntity.ok()
                .header("Content-Disposition", "attachment; filename=applications_job_" + jobId + ".csv")
                .header("Content-Type", "text/csv")
                .body(csv.toString());
    }
}