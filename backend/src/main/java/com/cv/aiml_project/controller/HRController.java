package com.cv.aiml_project.controller;

import com.cv.aiml_project.entity.*;
import com.cv.aiml_project.service.JobService;
import com.cv.aiml_project.service.ResumeService;
import com.cv.aiml_project.service.UserService;
import com.cv.aiml_project.service.AIMLIntegrationService;
import com.cv.aiml_project.service.SkillMatchResultService;
import jakarta.servlet.http.HttpSession;
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
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import java.io.IOException;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Controller
@RequestMapping("/hr")
@PreAuthorize("hasRole('HR')")
public class HRController {

    @Autowired
    private UserService userService;

    @Autowired
    private JobService jobService;

    @Autowired
    private ResumeService resumeService;

    @Autowired
    private AIMLIntegrationService aiMlIntegrationService;

    @Autowired
    private SkillMatchResultService skillMatchResultService;

    /**
     * Get the currently logged-in HR user
     */
    private User getCurrentUser() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        return userService.getUserByUsername(auth.getName())
                .orElseThrow(() -> new RuntimeException("User not found"));
    }

    // ==================== DASHBOARD ====================

    /**
     * HR Dashboard - Overview of candidates and AI statistics
     */
    @GetMapping("/dashboard")
    public String hrDashboard(Model model, HttpSession session) {
        User currentUser = getCurrentUser();
        session.setAttribute("user", currentUser);

        // Get statistics
        long totalCandidates = userService.getTotalCandidates();
        long activeCandidates = userService.getActiveCandidates();
        long mlProcessed = resumeService.getProcessedResumeCount();
        long resumesCount = resumeService.getTotalResumeCount();

        // Get recent candidates (last 10)
        List<User> recentCandidates = userService.getCandidates().stream()
                .sorted(Comparator.comparing(User::getCreatedAt).reversed())
                .limit(10)
                .collect(Collectors.toList());

        // Get top 5 candidates based on resume AI scores
        List<User> topCandidates = userService.getCandidates().stream()
                .filter(user -> {
                    Resume resume = user.getCurrentResume();
                    return resume != null && resume.isMlProcessed();
                })
                .sorted((u1, u2) -> {
                    Double score1 = u1.getCurrentResume().getMlScore();
                    Double score2 = u2.getCurrentResume().getMlScore();
                    if (score1 == null && score2 == null) return 0;
                    if (score1 == null) return 1;
                    if (score2 == null) return -1;
                    return score2.compareTo(score1);
                })
                .limit(5)
                .collect(Collectors.toList());

        // Get job statistics
        Map<String, Object> jobStats = jobService.getJobStatistics();

        model.addAttribute("totalCandidates", totalCandidates);
        model.addAttribute("activeCandidates", activeCandidates);
        model.addAttribute("mlProcessed", mlProcessed);
        model.addAttribute("resumesCount", resumesCount);
        model.addAttribute("recentCandidates", recentCandidates);
        model.addAttribute("topCandidates", topCandidates);
        model.addAttribute("jobStats", jobStats);

        return "hr/dashboard";
    }

    // ==================== CANDIDATE MANAGEMENT ====================

    /**
     * View all candidates with optional filtering
     */
    @GetMapping("/candidates")
    public String viewCandidates(Model model,
                                 @RequestParam(required = false) String search,
                                 @RequestParam(required = false) Boolean mlProcessed,
                                 @RequestParam(required = false) String sortBy) {

        List<User> candidates = userService.getCandidates();

        // Apply search filter
        if (search != null && !search.trim().isEmpty()) {
            String searchLower = search.toLowerCase();
            candidates = candidates.stream()
                    .filter(u ->
                            u.getFullName().toLowerCase().contains(searchLower) ||
                                    (u.getEmail() != null && u.getEmail().toLowerCase().contains(searchLower)) ||
                                    (u.getSkills() != null && u.getSkills().toLowerCase().contains(searchLower)) ||
                                    (u.getCurrentResume() != null &&
                                            u.getCurrentResume().getExtractedText() != null &&
                                            u.getCurrentResume().getExtractedText().toLowerCase().contains(searchLower))
                    )
                    .collect(Collectors.toList());
            model.addAttribute("search", search);
        }

        // Apply AI processed filter
        if (mlProcessed != null) {
            candidates = candidates.stream()
                    .filter(u -> {
                        Resume resume = u.getCurrentResume();
                        return resume != null && resume.isMlProcessed() == mlProcessed;
                    })
                    .collect(Collectors.toList());
            model.addAttribute("mlProcessed", mlProcessed);
        }

        // Apply sorting
        if (sortBy != null) {
            switch (sortBy) {
                case "score":
                    candidates.sort((u1, u2) -> {
                        Double score1 = u1.getCurrentResume() != null ? u1.getCurrentResume().getMlScore() : null;
                        Double score2 = u2.getCurrentResume() != null ? u2.getCurrentResume().getMlScore() : null;
                        if (score1 == null && score2 == null) return 0;
                        if (score1 == null) return 1;
                        if (score2 == null) return -1;
                        return score2.compareTo(score1);
                    });
                    break;
                case "name":
                    candidates.sort(Comparator.comparing(User::getFullName));
                    break;
                case "recent":
                    candidates.sort(Comparator.comparing(User::getCreatedAt).reversed());
                    break;
            }
            model.addAttribute("sortBy", sortBy);
        }

        // Get statistics for display
        long processedCount = candidates.stream()
                .filter(u -> u.getCurrentResume() != null && u.getCurrentResume().isMlProcessed())
                .count();

        model.addAttribute("candidates", candidates);
        model.addAttribute("processedCount", processedCount);
        model.addAttribute("totalCount", candidates.size());

        return "hr/candidates";
    }

    /**
     * View candidate details with resume and application history
     */
    @GetMapping("/candidates/{id}")
    public String viewCandidateDetails(@PathVariable Long id, Model model) {
        User candidate = userService.getUserById(id)
                .orElseThrow(() -> new IllegalArgumentException("Candidate not found"));

        if (!candidate.isCandidate()) {
            throw new IllegalArgumentException("User is not a candidate");
        }

        // Get candidate's resumes (all versions)
        List<Resume> allResumes = resumeService.getAllUserResumes(id);
        Resume currentResume = candidate.getCurrentResume();

        // Get candidate's job applications
        List<JobApplication> candidateApplications = jobService.getApplicationsByCandidate(id);

        // Get application statistics
        Map<ApplicationStatus, Long> applicationStats = candidateApplications.stream()
                .collect(Collectors.groupingBy(JobApplication::getStatus, Collectors.counting()));

        // Calculate average match score across applications
        double avgMatchScore = candidateApplications.stream()
                .filter(app -> app.getMatchScore() != null)
                .mapToDouble(JobApplication::getMatchScore)
                .average()
                .orElse(0.0);

        model.addAttribute("candidate", candidate);
        model.addAttribute("currentResume", currentResume);
        model.addAttribute("allResumes", allResumes);
        model.addAttribute("candidateApplications", candidateApplications);
        model.addAttribute("applicationStats", applicationStats);
        model.addAttribute("avgMatchScore", avgMatchScore);

        return "hr/candidate-details";
    }

    /**
     * Export candidate data as JSON (for API integration)
     */
    @GetMapping("/candidates/{id}/export")
    @ResponseBody
    public CandidateExportDTO exportCandidateData(@PathVariable Long id) {
        User candidate = userService.getUserById(id)
                .orElseThrow(() -> new IllegalArgumentException("Candidate not found"));

        Resume currentResume = candidate.getCurrentResume();

        CandidateExportDTO dto = new CandidateExportDTO();
        dto.setId(candidate.getId());
        dto.setName(candidate.getFullName());
        dto.setEmail(candidate.getEmail());
        dto.setPhone(candidate.getPhone());
        dto.setSkills(candidate.getSkills());
        dto.setExperienceYears(candidate.getExperienceYears());
        dto.setEducation(candidate.getEducation());

        if (currentResume != null) {
            dto.setResumeId(currentResume.getId());
            dto.setResumeFileName(currentResume.getOriginalName());
            dto.setResumeUploadDate(currentResume.getUploadDate());
            dto.setMlProcessed(currentResume.isMlProcessed());
            dto.setMlScore(currentResume.getMlScore());
            dto.setMlConfidence(currentResume.getMlConfidence());
            dto.setExtractedText(currentResume.getExtractedText());
        }

        return dto;
    }

    // DTO class for candidate export
    public static class CandidateExportDTO {
        private Long id;
        private String name;
        private String email;
        private String phone;
        private String skills;
        private Integer experienceYears;
        private String education;
        private Long resumeId;
        private String resumeFileName;
        private java.time.LocalDateTime resumeUploadDate;
        private boolean mlProcessed;
        private Double mlScore;
        private Double mlConfidence;
        private String extractedText;

        // Getters and setters
        public Long getId() { return id; }
        public void setId(Long id) { this.id = id; }
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
        public String getPhone() { return phone; }
        public void setPhone(String phone) { this.phone = phone; }
        public String getSkills() { return skills; }
        public void setSkills(String skills) { this.skills = skills; }
        public Integer getExperienceYears() { return experienceYears; }
        public void setExperienceYears(Integer experienceYears) { this.experienceYears = experienceYears; }
        public String getEducation() { return education; }
        public void setEducation(String education) { this.education = education; }
        public Long getResumeId() { return resumeId; }
        public void setResumeId(Long resumeId) { this.resumeId = resumeId; }
        public String getResumeFileName() { return resumeFileName; }
        public void setResumeFileName(String resumeFileName) { this.resumeFileName = resumeFileName; }
        public java.time.LocalDateTime getResumeUploadDate() { return resumeUploadDate; }
        public void setResumeUploadDate(java.time.LocalDateTime resumeUploadDate) { this.resumeUploadDate = resumeUploadDate; }
        public boolean isMlProcessed() { return mlProcessed; }
        public void setMlProcessed(boolean mlProcessed) { this.mlProcessed = mlProcessed; }
        public Double getMlScore() { return mlScore; }
        public void setMlScore(Double mlScore) { this.mlScore = mlScore; }
        public Double getMlConfidence() { return mlConfidence; }
        public void setMlConfidence(Double mlConfidence) { this.mlConfidence = mlConfidence; }
        public String getExtractedText() { return extractedText; }
        public void setExtractedText(String extractedText) { this.extractedText = extractedText; }
    }

    // ==================== RESUME OPERATIONS ====================

    /**
     * View candidate's current resume in browser
     */
    @GetMapping("/candidates/{id}/resume/view")
    public ResponseEntity<Resource> viewCandidateResume(@PathVariable Long id) throws IOException {
        User candidate = userService.getUserById(id)
                .orElseThrow(() -> new IllegalArgumentException("Candidate not found"));

        Resume currentResume = candidate.getCurrentResume();
        if (currentResume == null) {
            return ResponseEntity.notFound().build();
        }

        Resource resource = resumeService.getResumeFile(currentResume.getId());

        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_PDF)
                .header(HttpHeaders.CONTENT_DISPOSITION,
                        "inline; filename=\"" + currentResume.getOriginalName() + "\"")
                .body(resource);
    }

    /**
     * View specific version of candidate's resume
     */
    @GetMapping("/candidates/{candidateId}/resume/view/{resumeId}")
    public ResponseEntity<Resource> viewCandidateResumeVersion(
            @PathVariable Long candidateId,
            @PathVariable Long resumeId) throws IOException {

        User candidate = userService.getUserById(candidateId)
                .orElseThrow(() -> new IllegalArgumentException("Candidate not found"));

        Resume resume = resumeService.getResumeById(resumeId)
                .orElseThrow(() -> new IllegalArgumentException("Resume not found"));

        // Verify that the resume belongs to the candidate
        if (!resume.getUser().getId().equals(candidateId)) {
            return ResponseEntity.status(403).build();
        }

        Resource resource = resumeService.getResumeFile(resumeId);

        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_PDF)
                .header(HttpHeaders.CONTENT_DISPOSITION,
                        "inline; filename=\"" + resume.getOriginalName() + "\"")
                .body(resource);
    }

    /**
     * Download candidate's current resume
     */
    @GetMapping("/candidates/{id}/resume/download")
    public ResponseEntity<Resource> downloadCandidateResume(@PathVariable Long id) throws IOException {
        User candidate = userService.getUserById(id)
                .orElseThrow(() -> new IllegalArgumentException("Candidate not found"));

        Resume currentResume = candidate.getCurrentResume();
        if (currentResume == null) {
            return ResponseEntity.notFound().build();
        }

        Resource resource = resumeService.getResumeFile(currentResume.getId());

        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_OCTET_STREAM)
                .header(HttpHeaders.CONTENT_DISPOSITION,
                        "attachment; filename=\"" + currentResume.getOriginalName() + "\"")
                .body(resource);
    }

    /**
     * View extracted text from candidate's current resume
     */
    @GetMapping("/candidates/{id}/resume/text")
    public String viewCandidateResumeText(@PathVariable Long id, Model model) {
        User candidate = userService.getUserById(id)
                .orElseThrow(() -> new IllegalArgumentException("Candidate not found"));

        if (!candidate.isCandidate()) {
            throw new IllegalArgumentException("User is not a candidate");
        }

        Resume currentResume = candidate.getCurrentResume();

        model.addAttribute("candidate", candidate);
        model.addAttribute("resume", currentResume);

        if (currentResume == null) {
            model.addAttribute("error", "No resume found for this candidate");
        } else if (currentResume.getExtractedText() == null) {
            model.addAttribute("error", "No extracted text available for this resume");
        } else {
            model.addAttribute("resumeText", currentResume.getExtractedText());
        }

        return "hr/resume-text";
    }

    /**
     * View extracted text from specific resume version
     */
    @GetMapping("/candidates/{candidateId}/resume/text/{resumeId}")
    public String viewCandidateResumeVersionText(
            @PathVariable Long candidateId,
            @PathVariable Long resumeId,
            Model model) {

        User candidate = userService.getUserById(candidateId)
                .orElseThrow(() -> new IllegalArgumentException("Candidate not found"));

        Resume resume = resumeService.getResumeById(resumeId)
                .orElseThrow(() -> new IllegalArgumentException("Resume not found"));

        // Verify that the resume belongs to the candidate
        if (!resume.getUser().getId().equals(candidateId)) {
            throw new IllegalArgumentException("Resume does not belong to this candidate");
        }

        model.addAttribute("candidate", candidate);
        model.addAttribute("resume", resume);

        if (resume.getExtractedText() == null) {
            model.addAttribute("error", "No extracted text available for this resume");
        } else {
            model.addAttribute("resumeText", resume.getExtractedText());
        }

        return "hr/resume-text";
    }

    // ==================== AI/ML OPERATIONS ====================

    /**
     * Process a single candidate with AI
     */
    @PostMapping("/candidates/{id}/process-ai")
    public String processCandidateAI(@PathVariable Long id,
                                     @RequestParam(defaultValue = "false") boolean forceReprocess,
                                     RedirectAttributes redirectAttributes) {
        try {
            User candidate = userService.getUserById(id)
                    .orElseThrow(() -> new IllegalArgumentException("Candidate not found"));

            Resume currentResume = candidate.getCurrentResume();

            if (currentResume == null) {
                redirectAttributes.addFlashAttribute("error",
                        "Candidate has no resume to process");
                return "redirect:/hr/candidates/" + id;
            }

            // Check if already processed
            if (currentResume.isMlProcessed() && !forceReprocess) {
                redirectAttributes.addFlashAttribute("message",
                        "Candidate already processed. Use force reprocess to re-run AI analysis.");
                return "redirect:/hr/candidates/" + id;
            }

            // Call the real AI service
            aiMlIntegrationService.processResume(currentResume.getId());

            // Reload the resume to get updated scores
            currentResume = candidate.getCurrentResume();

            redirectAttributes.addFlashAttribute("message",
                    "AI analysis completed for " + candidate.getFullName() +
                            ". Score: " + String.format("%.1f", currentResume.getMlScore()) +
                            ", Confidence: " + String.format("%.1f%%", currentResume.getMlConfidence()));

        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "AI processing failed: " + e.getMessage());
        }
        return "redirect:/hr/candidates/" + id;
    }

    /**
     * Process all unprocessed candidates with AI
     */
    @PostMapping("/candidates/process-all-ai")
    @ResponseBody
    public Map<String, Object> processAllCandidatesAI() {
        List<Resume> unprocessedResumes = resumeService.getUnprocessedResumes();
        int processed = 0;

        for (Resume resume : unprocessedResumes) {
            try {
                aiMlIntegrationService.processResume(resume.getId());
                processed++;
            } catch (Exception e) {
                // Log error but continue processing others
                e.printStackTrace();
            }
        }

        return Map.of(
                "success", true,
                "message", "Processed " + processed + " candidates",
                "processed", processed,
                "total", unprocessedResumes.size()
        );
    }

    /**
     * Get AI analysis details for a candidate
     */
    @GetMapping("/candidates/{id}/ai-analysis")
    public String viewAIAnalysis(@PathVariable Long id, Model model) {
        User candidate = userService.getUserById(id)
                .orElseThrow(() -> new IllegalArgumentException("Candidate not found"));

        Resume currentResume = candidate.getCurrentResume();

        if (currentResume == null) {
            model.addAttribute("error", "Candidate has no resume");
        } else if (!currentResume.isMlProcessed()) {
            model.addAttribute("error", "Candidate has not been processed by AI yet");
        } else {
            model.addAttribute("resume", currentResume);

            // Get skill breakdown if available
            if (currentResume.getExtractedText() != null) {
                // Parse skills from extracted text (simplified for demo)
                String[] words = currentResume.getExtractedText().split("\\s+");
                model.addAttribute("wordCount", words.length);
            }
        }

        model.addAttribute("candidate", candidate);
        return "hr/ai-analysis";
    }

    /**
     * Compare multiple candidates for a job
     */
    @GetMapping("/candidates/compare")
    public String compareCandidates(@RequestParam List<Long> ids,
                                    @RequestParam(required = false) Long jobId,
                                    Model model) {

        List<User> candidates = ids.stream()
                .map(id -> userService.getUserById(id).orElse(null))
                .filter(u -> u != null && u.isCandidate())
                .collect(Collectors.toList());

        Job job = null;
        if (jobId != null) {
            job = jobService.getJobById(jobId).orElse(null);
        }

        // Calculate match scores for each candidate against the job
        if (job != null) {
            for (User candidate : candidates) {
                Resume resume = candidate.getCurrentResume();
                if (resume != null && resume.isMlProcessed()) {
                    // In production, this would call the AI matching service
                    Map<String, Object> matchResult = aiMlIntegrationService.calculateSkillMatch(job, resume);
                    candidate.setTempAttribute("matchScore", matchResult.get("score"));
                    candidate.setTempAttribute("matchDetails", matchResult);
                }
            }
        }

        model.addAttribute("candidates", candidates);
        model.addAttribute("job", job);
        return "hr/compare-candidates";
    }

    /**
     * Run fairness audit for AI Ranker
     */
    @GetMapping("/ai/fairness-audit")
    @ResponseBody
    public ResponseEntity<?> runFairnessAudit() {
        try {
            Map<String, Object> result = aiMlIntegrationService.auditFairness();
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.status(500).body(Map.of("error", e.getMessage()));
        }
    }

    /**
     * Generate interview questions using AI (JSON API — legacy)
     */
    @GetMapping("/applications/{applicationId}/questions")
    @ResponseBody
    public ResponseEntity<?> generateQuestionsApi(@PathVariable Long applicationId) {
        try {
            JobApplication application = jobService.getApplicationById(applicationId)
                    .orElseThrow(() -> new IllegalArgumentException("Application not found"));

            Long cvId = application.getResumeId();
            if (cvId == null) {
                if (application.getCandidate() != null && application.getCandidate().getCurrentResume() != null) {
                    cvId = application.getCandidate().getCurrentResume().getId();
                } else {
                    return ResponseEntity.badRequest().body(Map.of("error", "No CV found for this application"));
                }
            }

            Map<String, Object> questions = aiMlIntegrationService.generateInterviewQuestions(cvId, application.getJob().getId());
            return ResponseEntity.ok(questions);
        } catch (Exception e) {
            return ResponseEntity.status(500).body(Map.of("error", e.getMessage()));
        }
    }

    /**
     * Generate & SAVE interview questions for an application (redirects back to candidate details)
     */
    @PostMapping("/applications/{applicationId}/generate-questions")
    public String generateAndSaveQuestions(@PathVariable Long applicationId,
                                           RedirectAttributes redirectAttributes) {
        try {
            JobApplication application = jobService.getApplicationById(applicationId)
                    .orElseThrow(() -> new IllegalArgumentException("Application not found"));

            Long cvId = application.getResumeId();
            if (cvId == null && application.getCandidate() != null && application.getCandidate().getCurrentResume() != null) {
                cvId = application.getCandidate().getCurrentResume().getId();
            }
            if (cvId == null) {
                redirectAttributes.addFlashAttribute("error", "Candidate has no processed resume. Please run AI extraction first.");
                return "redirect:/hr/candidates/" + application.getCandidate().getId();
            }

            Map<String, Object> result = aiMlIntegrationService.generateInterviewQuestions(cvId, application.getJob().getId());
            Object questionsObj = result.get("questions");

            if (questionsObj == null) {
                redirectAttributes.addFlashAttribute("error", "Failed to generate questions: empty response from AI service.");
                return "redirect:/hr/candidates/" + application.getCandidate().getId();
            }

            // Serialize generated questions to JSON string and persist
            com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
            String questionsJson = mapper.writeValueAsString(questionsObj);

            application.setInterviewQuestionsJson(questionsJson);
            application.setInterviewAnswersJson(null); // Clear old answers if regenerating
            application.setQuestionsGeneratedAt(java.time.LocalDateTime.now());
            jobService.saveApplication(application);

            redirectAttributes.addFlashAttribute("message",
                    "AI interview questions generated successfully for " + application.getCandidate().getFullName() + "!");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Question generation failed: " + e.getMessage() +
                    ". Make sure the Python AI service is running and OPENROUTER_API_KEY is set.");
        }
        try {
            JobApplication app = jobService.getApplicationById(applicationId).orElseThrow();
            return "redirect:/hr/candidates/" + app.getCandidate().getId();
        } catch (Exception e) {
            return "redirect:/hr/dashboard";
        }
    }

    // ==================== JOB APPLICATION MANAGEMENT ====================

    /**
     * View all applications for a specific job
     */
    @GetMapping("/jobs/{jobId}/applications")
    public String viewJobApplications(@PathVariable Long jobId, Model model) {
        Job job = jobService.getJobById(jobId)
                .orElseThrow(() -> new IllegalArgumentException("Job not found"));

        List<JobApplication> applications = jobService.getApplicationsForJobRanked(jobId);
        Map<String, Object> stats = jobService.getApplicationStatsForJob(jobId);

        model.addAttribute("job", job);
        model.addAttribute("applications", applications);
        model.addAttribute("stats", stats);
        model.addAttribute("statuses", ApplicationStatus.values());

        return "applications/job-applications";
    }

    /**
     * View application details
     */
    @GetMapping("/applications/{applicationId}")
    public String viewApplicationDetails(@PathVariable Long applicationId, Model model) {
        JobApplication application = jobService.getApplicationById(applicationId)
                .orElseThrow(() -> new IllegalArgumentException("Application not found"));

        User candidate = application.getCandidate();
        Resume resumeUsed = application.getResumeId() != null ?
                resumeService.getResumeById(application.getResumeId()).orElse(null) :
                candidate.getCurrentResume();

        model.addAttribute("application", application);
        model.addAttribute("candidate", candidate);
        model.addAttribute("job", application.getJob());
        model.addAttribute("resume", resumeUsed);
        model.addAttribute("statuses", ApplicationStatus.values());

        // Get AI match details if available
        java.util.Optional<SkillMatchResult> matchResult = skillMatchResultService.getLatestMatchForJobAndCandidate(
                application.getJob().getId(), 
                candidate.getId()
        );

        if (matchResult.isPresent()) {
            model.addAttribute("matchResult", matchResult.get());
        }

        return "applications/application-details";
    }

    /**
     * Update application status
     */
    @PostMapping("/applications/{applicationId}/status")
    public String updateApplicationStatus(@PathVariable Long applicationId,
                                          @RequestParam ApplicationStatus status,
                                          @RequestParam(required = false) String notes,
                                          RedirectAttributes redirectAttributes) {
        try {
            JobApplication application = jobService.updateApplicationStatus(applicationId, status, notes);
            redirectAttributes.addFlashAttribute("message",
                    "Application status updated to " + status.getDisplayName());
            return "redirect:/hr/jobs/" + application.getJob().getId() + "/applications";
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to update status: " + e.getMessage());
            return "redirect:/hr/applications/" + applicationId;
        }
    }

    /**
     * Recalculate AI match score for an application
     */
    @PostMapping("/applications/{applicationId}/recalculate")
    public String recalculateMatchScore(@PathVariable Long applicationId,
                                        RedirectAttributes redirectAttributes) {
        try {
            JobApplication application = jobService.recalculateMatchScore(applicationId);

            // Also update resume AI data if needed
            if (application.getResumeId() != null) {
                Resume resume = resumeService.getResumeById(application.getResumeId()).orElse(null);
                if (resume != null && !resume.isMlProcessed()) {
                    aiMlIntegrationService.processResume(resume.getId());
                }
            }

            // Recalculate using Python AI pipeline and sync to JobApplication
            try {
                skillMatchResultService.createOrUpdateMatchResult(application.getJob().getId(), application.getCandidate().getId());
                application = jobService.getApplicationById(applicationId).get();
            } catch (Exception e) {
                System.err.println("Failed to run Python AI Match Score recalculation: " + e.getMessage());
            }

            redirectAttributes.addFlashAttribute("message",
                    "Match score recalculated: " + String.format("%.1f", application.getMatchScore()) + "%");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to recalculate: " + e.getMessage());
        }
        return "redirect:/hr/applications/" + applicationId;
    }

    /**
     * Download resume attached to an application
     */
    @GetMapping("/applications/{applicationId}/resume/download")
    public ResponseEntity<Resource> downloadApplicationResume(@PathVariable Long applicationId) throws IOException {
        JobApplication application = jobService.getApplicationById(applicationId)
                .orElseThrow(() -> new IllegalArgumentException("Application not found"));

        Long resumeId = application.getResumeId();
        if (resumeId == null) {
            // Fall back to candidate's current resume
            User candidate = application.getCandidate();
            Resume currentResume = candidate.getCurrentResume();
            if (currentResume == null) {
                throw new IllegalArgumentException("No resume found for this application");
            }
            resumeId = currentResume.getId();
        }

        Resume resume = resumeService.getResumeById(resumeId)
                .orElseThrow(() -> new IllegalArgumentException("Resume not found"));

        Resource resource = resumeService.getResumeFile(resumeId);

        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_PDF)
                .header(HttpHeaders.CONTENT_DISPOSITION,
                        "attachment; filename=\"" + resume.getOriginalName() + "\"")
                .body(resource);
    }

    // ==================== REPORTS AND ANALYTICS ====================

    /**
     * Generate candidate analytics report
     */
    @GetMapping("/reports/candidates")
    public String candidateReports(Model model) {
        List<User> allCandidates = userService.getCandidates();

        // Calculate statistics
        long totalCandidates = allCandidates.size();
        long withResume = allCandidates.stream().filter(User::hasResume).count();
        long aiProcessed = allCandidates.stream()
                .filter(u -> {
                    Resume r = u.getCurrentResume();
                    return r != null && r.isMlProcessed();
                })
                .count();

        // Score distribution
        Map<String, Long> scoreDistribution = allCandidates.stream()
                .map(User::getCurrentResume)
                .filter(r -> r != null && r.isMlProcessed())
                .collect(Collectors.groupingBy(
                        r -> {
                            double score = r.getMlScore();
                            if (score >= 90) return "90-100";
                            if (score >= 80) return "80-89";
                            if (score >= 70) return "70-79";
                            if (score >= 60) return "60-69";
                            return "Below 60";
                        },
                        Collectors.counting()
                ));

        // Skill frequency
        Map<String, Long> skillFrequency = allCandidates.stream()
                .filter(u -> u.getSkills() != null)
                .flatMap(u -> List.of(u.getSkills().split(",")).stream())
                .map(String::trim)
                .collect(Collectors.groupingBy(s -> s, Collectors.counting()));

        model.addAttribute("totalCandidates", totalCandidates);
        model.addAttribute("withResume", withResume);
        model.addAttribute("aiProcessed", aiProcessed);
        model.addAttribute("scoreDistribution", scoreDistribution);
        model.addAttribute("skillFrequency", skillFrequency.entrySet().stream()
                .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
                .limit(20)
                .collect(Collectors.toList()));

        return "hr/candidate-reports";
    }

    /**
     * Export candidates report as CSV
     */
    @GetMapping("/reports/candidates/export")
    public ResponseEntity<String> exportCandidatesReport() {
        List<User> candidates = userService.getCandidates();

        StringBuilder csv = new StringBuilder();
        csv.append("ID,Name,Email,Phone,Experience,Skills,Education,Has Resume,AI Score,AI Confidence\n");

        for (User candidate : candidates) {
            Resume resume = candidate.getCurrentResume();
            csv.append(candidate.getId()).append(",")
                    .append("\"").append(candidate.getFullName()).append("\",")
                    .append("\"").append(candidate.getEmail()).append("\",")
                    .append("\"").append(candidate.getPhone() != null ? candidate.getPhone() : "").append("\",")
                    .append(candidate.getExperienceYears() != null ? candidate.getExperienceYears() : "").append(",")
                    .append("\"").append(candidate.getSkills() != null ? candidate.getSkills() : "").append("\",")
                    .append("\"").append(candidate.getEducation() != null ? candidate.getEducation() : "").append("\",")
                    .append(resume != null ? "Yes" : "No").append(",")
                    .append(resume != null && resume.getMlScore() != null ? resume.getMlScore() : "").append(",")
                    .append(resume != null && resume.getMlConfidence() != null ? resume.getMlConfidence() : "")
                    .append("\n");
        }

        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=candidates-report.csv")
                .contentType(MediaType.parseMediaType("text/csv"))
                .body(csv.toString());
    }

    // ==================== AI SKILL MATCHING ====================

    /**
     * Run AI skill matching for a single application
     */
    @PostMapping("/applications/{applicationId}/run-match")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String runSkillMatch(@PathVariable Long applicationId, RedirectAttributes redirectAttributes) {
        try {
            JobApplication application = jobService.getApplicationById(applicationId)
                    .orElseThrow(() -> new IllegalArgumentException("Application not found"));

            Long jobId = application.getJob().getId();
            Long candidateId = application.getCandidate().getId();

            SkillMatchResult matchResult = skillMatchResultService.createOrUpdateMatchResult(jobId, candidateId);

            // Update the application's match scores from the result
            if (matchResult.getOverallScore() != null) {
                application.setMatchScore(matchResult.getOverallScore());
            }
            if (matchResult.getSkillsScore() != null) {
                application.setSkillsMatchScore(matchResult.getSkillsScore());
            }
            if (matchResult.getExperienceScore() != null) {
                application.setExperienceMatchScore(matchResult.getExperienceScore());
            }
            jobService.saveApplication(application);

            redirectAttributes.addFlashAttribute("message",
                    "AI Skill Match completed for " + application.getCandidate().getFullName()
                    + " — Score: " + String.format("%.0f%%", matchResult.getOverallScore()));
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error",
                    "Skill matching failed: " + e.getMessage() + ". Make sure the Python AI service is running.");
        }

        return "redirect:/applications/" + applicationId;
    }

    /**
     * Run AI skill matching for multiple applications (batch)
     */
    @PostMapping("/applications/batch-match")
    public String batchSkillMatch(@RequestParam List<Long> applicationIds,
                                  @RequestParam Long jobId,
                                  RedirectAttributes redirectAttributes) {
        int success = 0;
        int failed = 0;
        for (Long appId : applicationIds) {
            try {
                JobApplication application = jobService.getApplicationById(appId)
                        .orElseThrow(() -> new IllegalArgumentException("Application not found"));

                SkillMatchResult matchResult = skillMatchResultService.createOrUpdateMatchResult(
                        application.getJob().getId(), application.getCandidate().getId());

                if (matchResult.getOverallScore() != null) {
                    application.setMatchScore(matchResult.getOverallScore());
                }
                if (matchResult.getSkillsScore() != null) {
                    application.setSkillsMatchScore(matchResult.getSkillsScore());
                }
                if (matchResult.getExperienceScore() != null) {
                    application.setExperienceMatchScore(matchResult.getExperienceScore());
                }
                jobService.saveApplication(application);
                success++;
            } catch (Exception e) {
                failed++;
            }
        }
        redirectAttributes.addFlashAttribute("message",
                "Batch AI Match complete: " + success + " succeeded, " + failed + " failed.");
        return "redirect:/hr/jobs/" + jobId + "/applications";
    }
}