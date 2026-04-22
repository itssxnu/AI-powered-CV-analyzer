package com.cv.aiml_project.controller;

import com.cv.aiml_project.entity.Job;
import com.cv.aiml_project.entity.SkillMatchResult;
import com.cv.aiml_project.entity.User;
import com.cv.aiml_project.service.JobService;
import com.cv.aiml_project.service.SkillMatchResultService;
import com.cv.aiml_project.service.UserService;
import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Controller
@RequestMapping("/skill-match")
public class SkillMatchResultController {

    @Autowired
    private SkillMatchResultService skillMatchResultService;

    @Autowired
    private JobService jobService;

    @Autowired
    private UserService userService;

    // ==================== ADMIN/HR ENDPOINTS ====================

    /**
     * View skill match results for a job
     */
    // In SkillMatchResultController.java

    @GetMapping("/job/{jobId}")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    @Transactional // Add this
    public String viewJobMatchResults(@PathVariable Long jobId, Model model) {
        Job job = jobService.getJobById(jobId)
                .orElseThrow(() -> new IllegalArgumentException("Job not found"));

        List<SkillMatchResult> results = skillMatchResultService.getLatestMatchResultsForJob(jobId);

        // Force initialization of candidate data
        for (SkillMatchResult result : results) {
            if (result.getCandidate() != null) {
                // This forces Hibernate to load the candidate data
                result.getCandidate().getId();
                result.getCandidate().getFullName();
                result.getCandidate().getEmail();
            }
        }

        Map<String, Object> stats = skillMatchResultService.getMatchStatisticsForJob(jobId);

        model.addAttribute("job", job);
        model.addAttribute("results", results);
        model.addAttribute("stats", stats);

        return "skill-match/job-results";
    }

    /**
     * View specific match result details
     */
    @GetMapping("/result/{resultId}")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN', 'CANDIDATE')")
    public String viewMatchResultDetails(@PathVariable Long resultId, Model model) {
        SkillMatchResult result = skillMatchResultService.getMatchResultById(resultId)
                .orElseThrow(() -> new IllegalArgumentException("Match result not found"));

        model.addAttribute("result", result);
        model.addAttribute("job", result.getJob());
        model.addAttribute("candidate", result.getCandidate());

        return "skill-match/result-details";
    }

    /**
     * Create new match result for a job and candidate
     */
    @PostMapping("/create")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String createMatchResult(@RequestParam Long jobId,
                                    @RequestParam Long candidateId,
                                    RedirectAttributes redirectAttributes) {
        try {
            SkillMatchResult result = skillMatchResultService.createOrUpdateMatchResult(jobId, candidateId);
            redirectAttributes.addFlashAttribute("message",
                    "Match result created successfully for candidate. Score: " +
                            String.format("%.1f", result.getOverallScore()));
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to create match result: " + e.getMessage());
        }
        return "redirect:/skill-match/job/" + jobId;
    }

    /**
     * Recalculate score for a match result
     */
    @PostMapping("/recalculate/{resultId}")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String recalculateScore(@PathVariable Long resultId,
                                   RedirectAttributes redirectAttributes) {
        try {
            SkillMatchResult result = skillMatchResultService.recalculateScore(resultId);
            redirectAttributes.addFlashAttribute("message",
                    "Score recalculated successfully. New score: " +
                            String.format("%.1f", result.getOverallScore()));
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to recalculate: " + e.getMessage());
        }
        return "redirect:/skill-match/result/" + resultId;
    }

    /**
     * Remove match record (soft delete)
     */
    @PostMapping("/remove/{resultId}")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String removeMatchRecord(@PathVariable Long resultId,
                                    @RequestParam Long jobId,
                                    RedirectAttributes redirectAttributes) {
        try {
            skillMatchResultService.removeMatchRecord(resultId);
            redirectAttributes.addFlashAttribute("message", "Match record removed successfully");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to remove record: " + e.getMessage());
        }
        return "redirect:/skill-match/job/" + jobId;
    }

    /**
     * Delete match result permanently
     */
    @PostMapping("/delete/{resultId}")
    @PreAuthorize("hasAnyRole('ADMIN')")
    public String deleteMatchResult(@PathVariable Long resultId,
                                    @RequestParam Long jobId,
                                    RedirectAttributes redirectAttributes) {
        try {
            skillMatchResultService.deleteMatchResult(resultId);
            redirectAttributes.addFlashAttribute("message", "Match result deleted permanently");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to delete: " + e.getMessage());
        }
        return "redirect:/skill-match/job/" + jobId;
    }

    /**
     * Process all candidates for a job
     */
    @PostMapping("/job/{jobId}/process-all")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String processAllCandidates(@PathVariable Long jobId,
                                       RedirectAttributes redirectAttributes) {
        try {
            int processed = skillMatchResultService.processAllCandidatesForJob(jobId);
            skillMatchResultService.updateRankingForJob(jobId);
            redirectAttributes.addFlashAttribute("message",
                    "Processed " + processed + " candidates for this job");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to process candidates: " + e.getMessage());
        }
        return "redirect:/skill-match/job/" + jobId;
    }

    /**
     * Batch recalculate all scores for a job
     */
    @PostMapping("/job/{jobId}/batch-recalculate")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String batchRecalculate(@PathVariable Long jobId,
                                   RedirectAttributes redirectAttributes) {
        try {
            int recalculated = skillMatchResultService.batchRecalculateForJob(jobId);
            skillMatchResultService.updateRankingForJob(jobId);
            redirectAttributes.addFlashAttribute("message",
                    "Recalculated " + recalculated + " match results");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to recalculate: " + e.getMessage());
        }
        return "redirect:/skill-match/job/" + jobId;
    }

    /**
     * Update rankings for a job
     */
    @PostMapping("/job/{jobId}/update-ranking")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    @ResponseBody
    public ResponseEntity<?> updateRanking(@PathVariable Long jobId) {
        try {
            skillMatchResultService.updateRankingForJob(jobId);
            List<Map<String, Object>> rankedList = skillMatchResultService.getRankedCandidatesForJob(jobId);
            return ResponseEntity.ok(Map.of(
                    "success", true,
                    "message", "Ranking updated successfully",
                    "rankings", rankedList
            ));
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of(
                    "success", false,
                    "error", e.getMessage()
            ));
        }
    }

    /**
     * Get ranked candidates for a job (API endpoint)
     */
    @GetMapping("/api/job/{jobId}/rankings")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    @ResponseBody
    public ResponseEntity<?> getRankedCandidates(@PathVariable Long jobId) {
        try {
            List<Map<String, Object>> rankings = skillMatchResultService.getRankedCandidatesForJob(jobId);
            return ResponseEntity.ok(Map.of(
                    "success", true,
                    "rankings", rankings
            ));
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of(
                    "success", false,
                    "error", e.getMessage()
            ));
        }
    }

    // ==================== CANDIDATE ENDPOINTS ====================

    /**
     * View candidate's own match results
     */
    @GetMapping("/my-results")
    @PreAuthorize("hasRole('CANDIDATE')")
    public String viewMyMatchResults(Model model) {
        // Get current candidate
        User currentUser = getCurrentCandidate();
        List<SkillMatchResult> results = skillMatchResultService.getMatchResultsForCandidate(currentUser.getId());

        // Group by match level
        List<SkillMatchResult> excellentMatches = results.stream()
                .filter(r -> r.getOverallScore() != null && r.getOverallScore() >= 80)
                .collect(Collectors.toList());

        List<SkillMatchResult> goodMatches = results.stream()
                .filter(r -> r.getOverallScore() != null && r.getOverallScore() >= 60 && r.getOverallScore() < 80)
                .collect(Collectors.toList());

        List<SkillMatchResult> otherMatches = results.stream()
                .filter(r -> r.getOverallScore() == null || r.getOverallScore() < 60)
                .collect(Collectors.toList());

        model.addAttribute("excellentMatches", excellentMatches);
        model.addAttribute("goodMatches", goodMatches);
        model.addAttribute("otherMatches", otherMatches);
        model.addAttribute("totalMatches", results.size());

        return "skill-match/my-results";
    }
    /**
     * View match results for a specific candidate (HR/Admin view)
     */
    @GetMapping("/candidate/{candidateId}")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String viewCandidateMatchResults(@PathVariable Long candidateId, Model model) {
        User candidate = userService.getUserById(candidateId)
                .orElseThrow(() -> new IllegalArgumentException("Candidate not found"));

        List<SkillMatchResult> results = skillMatchResultService.getMatchResultsForCandidate(candidateId);

        model.addAttribute("candidate", candidate);
        model.addAttribute("results", results);

        return "skill-match/candidate-results";
    }

    /**
     * Get match result for specific job and candidate
     */
    @GetMapping("/result/job-candidate")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN', 'CANDIDATE')")
    public String getMatchResultByJobAndCandidate(@RequestParam Long jobId,
                                                  @RequestParam Long candidateId,
                                                  Model model) {
        SkillMatchResult result = skillMatchResultService.getLatestMatchForJobAndCandidate(jobId, candidateId)
                .orElseThrow(() -> new IllegalArgumentException("Match result not found"));

        model.addAttribute("result", result);
        model.addAttribute("job", result.getJob());
        model.addAttribute("candidate", result.getCandidate());

        return "skill-match/result-details";
    }

    /**
     * Get current candidate from security context
     */
    private User getCurrentCandidate() {
        org.springframework.security.core.Authentication auth =
                org.springframework.security.core.context.SecurityContextHolder.getContext().getAuthentication();
        return userService.getUserByUsername(auth.getName())
                .orElseThrow(() -> new RuntimeException("User not found"));
    }

    // ==================== STATISTICS ENDPOINTS ====================

    /**
     * View match statistics dashboard
     */
    @GetMapping("/dashboard")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String matchDashboard(Model model) {
        List<Job> activeJobs = jobService.getActiveJobs();
        Map<Long, Map<String, Object>> jobStats = activeJobs.stream()
                .collect(Collectors.toMap(
                        Job::getId,
                        job -> skillMatchResultService.getMatchStatisticsForJob(job.getId())
                ));

        model.addAttribute("activeJobs", activeJobs);
        model.addAttribute("jobStats", jobStats);

        return "skill-match/dashboard";
    }

    /**
     * Export match results for a job (CSV)
     */
    @GetMapping("/job/{jobId}/export")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    @ResponseBody
    public ResponseEntity<String> exportMatchResults(@PathVariable Long jobId) {
        try {
            List<SkillMatchResult> results = skillMatchResultService.getLatestMatchResultsForJob(jobId);
            Job job = jobService.getJobById(jobId).orElseThrow();

            StringBuilder csv = new StringBuilder();
            csv.append("Rank, Candidate Name, Email, Overall Score, Skills Score, Experience Score, ")
                    .append("Education Score, Matched Skills, Missing Skills, Match Level, Percentile\n");

            int rank = 1;
            for (SkillMatchResult result : results) {
                csv.append(rank++).append(",")
                        .append("\"").append(result.getCandidate().getFullName()).append("\",")
                        .append("\"").append(result.getCandidate().getEmail()).append("\",")
                        .append(result.getOverallScore() != null ? String.format("%.1f", result.getOverallScore()) : "").append(",")
                        .append(result.getSkillsScore() != null ? String.format("%.1f", result.getSkillsScore()) : "").append(",")
                        .append(result.getExperienceScore() != null ? String.format("%.1f", result.getExperienceScore()) : "").append(",")
                        .append(result.getEducationScore() != null ? String.format("%.1f", result.getEducationScore()) : "").append(",")
                        .append("\"").append(result.getMatchedSkills()).append("\",")
                        .append("\"").append(result.getMissingSkills()).append("\",")
                        .append("\"").append(result.getMatchLevel()).append("\",")
                        .append(result.getPercentile() != null ? String.format("%.1f", result.getPercentile()) : "")
                        .append("\n");
            }

            String filename = "skill_match_results_job_" + jobId + "_" + job.getTitle().replace(" ", "_") + ".csv";

            return ResponseEntity.ok()
                    .header("Content-Disposition", "attachment; filename=\"" + filename + "\"")
                    .header("Content-Type", "text/csv")
                    .body(csv.toString());

        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Error exporting data: " + e.getMessage());
        }
    }
}
