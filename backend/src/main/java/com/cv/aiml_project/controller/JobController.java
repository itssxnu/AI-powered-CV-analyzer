package com.cv.aiml_project.controller;

import com.cv.aiml_project.entity.*;
import com.cv.aiml_project.service.JobService;
import com.cv.aiml_project.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Map;

@Controller
@RequestMapping("/jobs")
public class JobController {

    @Autowired
    private JobService jobService;

    @Autowired
    private UserService userService;

    @Autowired
    private com.cv.aiml_project.service.SkillMatchResultService skillMatchResultService;

    // ==================== PUBLIC JOB VIEWING (All Users) ====================

    /**
     * Browse Jobs (Public)
     */
    @GetMapping("/browse")
    public String browseJobs(Model model,
                             @RequestParam(required = false) String search,
                             @RequestParam(required = false) String department,
                             @RequestParam(required = false) JobType jobType) {

        List<Job> jobs;

        if (search != null && !search.trim().isEmpty()) {
            jobs = jobService.searchJobs(search);
            model.addAttribute("search", search);
        } else if (department != null && !department.trim().isEmpty()) {
            jobs = jobService.getJobsByDepartment(department);
            model.addAttribute("department", department);
        } else if (jobType != null) {
            jobs = jobService.getJobsByType(jobType);
            model.addAttribute("jobType", jobType);
        } else {
            jobs = jobService.getActiveJobs();
        }

        // Get statistics for display
        Map<String, Object> stats = jobService.getJobStatistics();

        model.addAttribute("jobs", jobs);
        model.addAttribute("stats", stats);
        model.addAttribute("jobTypes", JobType.values());

        return "jobs/browse";
    }

    /**
     * View Job Details (Public)
     */
    @GetMapping("/view/{id}")
    public String viewJobDetails(@PathVariable Long id, Model model) {
        Job job = jobService.getJobById(id)
                .orElseThrow(() -> new IllegalArgumentException("Job not found"));

        model.addAttribute("job", job);

        // Check if current user is authenticated and is a candidate
        try {
            Authentication auth = SecurityContextHolder.getContext().getAuthentication();
            if (auth != null && auth.isAuthenticated() && !auth.getName().equals("anonymousUser")) {
                User currentUser = userService.getUserByUsername(auth.getName()).orElse(null);
                if (currentUser != null) {
                    model.addAttribute("currentUser", currentUser);

                    if (currentUser.isCandidate()) {
                        boolean hasApplied = jobService.hasCandidateApplied(id, currentUser.getId());
                        model.addAttribute("hasApplied", hasApplied);
                    }
                }
            }
        } catch (Exception e) {
            // Ignore - user not logged in
        }

        return "jobs/view";
    }

    // ==================== CANDIDATE JOB OPERATIONS ====================

    /**
     * Show job application form with resume upload
     */
    @GetMapping("/apply/{id}")
    @PreAuthorize("hasRole('CANDIDATE')")
    public String showApplicationForm(@PathVariable Long id, Model model, RedirectAttributes redirectAttributes) {
        try {
            Job job = jobService.getJobById(id)
                    .orElseThrow(() -> new IllegalArgumentException("Job not found"));

            // Check if job is active and not expired
            if (!job.isActive() || job.isExpired()) {
                redirectAttributes.addFlashAttribute("error", "This job is no longer accepting applications");
                return "redirect:/jobs/view/" + id;
            }

            Authentication auth = SecurityContextHolder.getContext().getAuthentication();
            User candidate = userService.getUserByUsername(auth.getName())
                    .orElseThrow(() -> new RuntimeException("User not found"));

            // Check if already applied
            if (jobService.hasCandidateApplied(id, candidate.getId())) {
                redirectAttributes.addFlashAttribute("error", "You have already applied for this job");
                return "redirect:/jobs/my-applications";
            }

            model.addAttribute("job", job);
            model.addAttribute("candidate", candidate);

            return "jobs/apply";

        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Error: " + e.getMessage());
            return "redirect:/jobs/browse";
        }
    }

    /**
     * Process job application with resume upload
     */
    @PostMapping("/apply/{id}")
    @PreAuthorize("hasRole('CANDIDATE')")
    public String processApplication(@PathVariable Long id,
                                     @RequestParam("resume") MultipartFile resumeFile,
                                     @RequestParam(required = false) String notes,
                                     RedirectAttributes redirectAttributes) {
        try {
            Authentication auth = SecurityContextHolder.getContext().getAuthentication();
            User candidate = userService.getUserByUsername(auth.getName())
                    .orElseThrow(() -> new RuntimeException("User not found"));

            // Validate resume
            if (resumeFile == null || resumeFile.isEmpty()) {
                redirectAttributes.addFlashAttribute("error", "Please select a resume file");
                return "redirect:/jobs/apply/" + id;
            }

            // Check file type
            String contentType = resumeFile.getContentType();
            if (!"application/pdf".equals(contentType)) {
                redirectAttributes.addFlashAttribute("error", "Only PDF files are allowed");
                return "redirect:/jobs/apply/" + id;
            }

            // Check file size (max 5MB)
            if (resumeFile.getSize() > 5 * 1024 * 1024) {
                redirectAttributes.addFlashAttribute("error", "File size must be less than 5MB");
                return "redirect:/jobs/apply/" + id;
            }

            // Submit application with resume
            JobApplication application = jobService.applyForJobWithResume(id, candidate.getId(), resumeFile, notes);

            String message = "Application submitted successfully! ";
            if (application.getMatchScore() != null) {
                message += "Match Score: " + String.format("%.1f", application.getMatchScore()) + "%";
            } else {
                message += "Your resume is being analyzed by AI.";
            }

            redirectAttributes.addFlashAttribute("message", message);

            return "redirect:/jobs/my-applications";

        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to apply: " + e.getMessage());
            return "redirect:/jobs/apply/" + id;
        }
    }

    /**
     * View My Applications (Candidate)
     */
    @GetMapping("/my-applications")
    @PreAuthorize("hasRole('CANDIDATE')")
    public String viewMyApplications(Model model) {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        User candidate = userService.getUserByUsername(auth.getName())
                .orElseThrow(() -> new RuntimeException("User not found"));

        List<JobApplication> applications = jobService.getApplicationsByCandidate(candidate.getId());

        model.addAttribute("applications", applications);
        return "jobs/my-applications";
    }

    /**
     * Withdraw Application
     */
    @PostMapping("/withdraw/{applicationId}")
    @PreAuthorize("hasRole('CANDIDATE')")
    public String withdrawApplication(@PathVariable Long applicationId,
                                      RedirectAttributes redirectAttributes) {
        try {
            Authentication auth = SecurityContextHolder.getContext().getAuthentication();
            User currentUser = userService.getUserByUsername(auth.getName())
                    .orElseThrow(() -> new RuntimeException("User not found"));

            JobApplication application = jobService.getApplicationById(applicationId)
                    .orElseThrow(() -> new RuntimeException("Application not found"));

            // Verify ownership
            if (!application.getCandidate().getId().equals(currentUser.getId())) {
                throw new RuntimeException("You don't have permission to withdraw this application");
            }

            jobService.updateApplicationStatus(applicationId, ApplicationStatus.WITHDRAWN,
                    "Withdrawn by candidate");

            redirectAttributes.addFlashAttribute("message", "Application withdrawn successfully");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to withdraw: " + e.getMessage());
        }
        return "redirect:/jobs/my-applications";
    }

    // ==================== HR/ADMIN JOB MANAGEMENT ====================

    /**
     * Job Management Dashboard (HR/Admin)
     */
    @GetMapping("/manage")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String manageJobs(Model model) {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        User currentUser = userService.getUserByUsername(auth.getName())
                .orElseThrow(() -> new RuntimeException("User not found"));

        List<Job> jobs;
        if (currentUser.isAdmin()) {
            jobs = jobService.getAllJobs();
        } else {
            jobs = jobService.getJobsPostedByUser(currentUser.getId());
        }

        model.addAttribute("jobs", jobs);
        model.addAttribute("jobTypes", JobType.values());

        return "jobs/manage";
    }

    /**
     * Show Create Job Form
     */
    @GetMapping("/create")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String showCreateJobForm(Model model) {
        model.addAttribute("job", new Job());
        model.addAttribute("jobTypes", JobType.values());
        return "jobs/create";
    }

    /**
     * Create Job
     */
    @PostMapping("/create")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String createJob(@ModelAttribute Job job,
                            @RequestParam(required = false) String expiryDateStr,
                            RedirectAttributes redirectAttributes) {
        try {
            Authentication auth = SecurityContextHolder.getContext().getAuthentication();
            User currentUser = userService.getUserByUsername(auth.getName())
                    .orElseThrow(() -> new RuntimeException("User not found"));

            // Parse expiry date if provided
            if (expiryDateStr != null && !expiryDateStr.isEmpty()) {
                DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm");
                LocalDateTime expiryDate = LocalDateTime.parse(expiryDateStr, formatter);
                job.setExpiryDate(expiryDate);
            }

            Job created = jobService.createJob(job, currentUser.getId());

            redirectAttributes.addFlashAttribute("message",
                    "Job posted successfully! Job ID: " + created.getId());

            return "redirect:/jobs/manage";
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to create job: " + e.getMessage());
            return "redirect:/jobs/create";
        }
    }

    /**
     * Show Edit Job Form
     */
    @GetMapping("/edit/{id}")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String showEditJobForm(@PathVariable Long id, Model model) {
        Job job = jobService.getJobById(id)
                .orElseThrow(() -> new IllegalArgumentException("Job not found"));

        // Check permissions
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        User currentUser = userService.getUserByUsername(auth.getName())
                .orElseThrow(() -> new RuntimeException("User not found"));

        if (!currentUser.isAdmin() && !job.getPostedBy().getId().equals(currentUser.getId())) {
            throw new RuntimeException("You don't have permission to edit this job");
        }

        model.addAttribute("job", job);
        model.addAttribute("jobTypes", JobType.values());

        return "jobs/edit";
    }

    /**
     * Update Job
     */
    @PostMapping("/edit/{id}")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String updateJob(@PathVariable Long id,
                            @ModelAttribute Job job,
                            @RequestParam(required = false) String expiryDateStr,
                            RedirectAttributes redirectAttributes) {
        try {
            // Parse expiry date if provided
            if (expiryDateStr != null && !expiryDateStr.isEmpty()) {
                DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm");
                LocalDateTime expiryDate = LocalDateTime.parse(expiryDateStr, formatter);
                job.setExpiryDate(expiryDate);
            }

            Job updated = jobService.updateJob(id, job);

            redirectAttributes.addFlashAttribute("message", "Job updated successfully");

            return "redirect:/jobs/manage";
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to update job: " + e.getMessage());
            return "redirect:/jobs/edit/" + id;
        }
    }

    /**
     * Toggle Job Status (Activate/Deactivate)
     */
    @PostMapping("/toggle-status/{id}")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String toggleJobStatus(@PathVariable Long id,
                                  @RequestParam boolean active,
                                  RedirectAttributes redirectAttributes) {
        try {
            jobService.toggleJobStatus(id, active);
            redirectAttributes.addFlashAttribute("message",
                    "Job " + (active ? "activated" : "deactivated") + " successfully");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to update status: " + e.getMessage());
        }
        return "redirect:/jobs/manage";
    }

    /**
     * Delete Job
     */
    @PostMapping("/delete/{id}")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String deleteJob(@PathVariable Long id, RedirectAttributes redirectAttributes) {
        try {
            jobService.deleteJob(id);
            redirectAttributes.addFlashAttribute("message", "Job deleted successfully");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to delete job: " + e.getMessage());
        }
        return "redirect:/jobs/manage";
    }

    // ==================== HR/ADMIN APPLICATION MANAGEMENT ====================

    /**
     * View Applications for a Job (HR/Admin)
     */
    @GetMapping("/{jobId}/applications")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String viewJobApplications(@PathVariable Long jobId, Model model) {
        Job job = jobService.getJobById(jobId)
                .orElseThrow(() -> new IllegalArgumentException("Job not found"));

        List<JobApplication> applications = jobService.getApplicationsForJob(jobId);
        Map<String, Object> stats = jobService.getApplicationStatsForJob(jobId);

        model.addAttribute("job", job);
        model.addAttribute("applications", applications);
        model.addAttribute("stats", stats);
        model.addAttribute("statuses", ApplicationStatus.values());

        return "applications/job-applications";
    }

    /**
     * View Application Details (HR/Admin)
     */
    @GetMapping("/applications/{applicationId}")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String viewApplicationDetails(@PathVariable Long applicationId, Model model) {
        JobApplication application = jobService.getApplicationById(applicationId)
                .orElseThrow(() -> new IllegalArgumentException("Application not found"));

        model.addAttribute("application", application);
        model.addAttribute("candidate", application.getCandidate());
        model.addAttribute("job", application.getJob());
        model.addAttribute("statuses", ApplicationStatus.values());

        // Get AI match details if available
        java.util.Optional<com.cv.aiml_project.entity.SkillMatchResult> matchResult = skillMatchResultService.getLatestMatchForJobAndCandidate(
                application.getJob().getId(), 
                application.getCandidate().getId()
        );

        if (matchResult.isPresent()) {
            model.addAttribute("matchResult", matchResult.get());
        }

        return "applications/application-details";
    }

    /**
     * Update Application Status (HR/Admin)
     */
    @PostMapping("/applications/{applicationId}/status")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String updateApplicationStatus(@PathVariable Long applicationId,
                                          @RequestParam ApplicationStatus status,
                                          @RequestParam(required = false) String notes,
                                          RedirectAttributes redirectAttributes) {
        try {
            JobApplication application = jobService.updateApplicationStatus(applicationId, status, notes);
            redirectAttributes.addFlashAttribute("message",
                    "Application status updated to " + status.getDisplayName());
            return "redirect:/jobs/" + application.getJob().getId() + "/applications";
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to update status: " + e.getMessage());
            return "redirect:/jobs/applications/" + applicationId;
        }
    }

    /**
     * Download Application Resume (HR/Admin)
     */
    @GetMapping("/applications/{applicationId}/resume/download")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public ResponseEntity<Resource> downloadApplicationResume(@PathVariable Long applicationId) throws IOException {
        JobApplication application = jobService.getApplicationById(applicationId)
                .orElseThrow(() -> new IllegalArgumentException("Application not found"));

        if (application.getResumePath() == null) {
            throw new IllegalArgumentException("No resume found for this application");
        }

        Resource resource = jobService.getApplicationResumeFile(applicationId);

        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_PDF)
                .header(HttpHeaders.CONTENT_DISPOSITION,
                        "attachment; filename=\"" + application.getResumeOriginalName() + "\"")
                .body(resource);
    }

    /**
     * Recalculate AI Match Score (HR/Admin)
     */
    @PostMapping("/applications/{applicationId}/recalculate")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String recalculateMatchScore(@PathVariable Long applicationId,
                                        RedirectAttributes redirectAttributes) {
        try {
            JobApplication application = jobService.recalculateMatchScore(applicationId);
            redirectAttributes.addFlashAttribute("message",
                    "Match score recalculated: " + String.format("%.1f", application.getMatchScore()) + "%");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to recalculate: " + e.getMessage());
        }
        return "redirect:/jobs/applications/" + applicationId;
    }

    /**
     * Remove Application (Soft Delete)
     */
    @PostMapping("/applications/{applicationId}/remove")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String removeApplication(@PathVariable Long applicationId,
                                    RedirectAttributes redirectAttributes) {
        try {
            JobApplication application = jobService.getApplicationById(applicationId)
                    .orElseThrow(() -> new RuntimeException("Application not found"));

            jobService.removeApplication(applicationId);

            redirectAttributes.addFlashAttribute("message", "Application removed successfully");
            return "redirect:/jobs/" + application.getJob().getId() + "/applications";
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to remove: " + e.getMessage());
            return "redirect:/jobs/applications/" + applicationId;
        }
    }
}