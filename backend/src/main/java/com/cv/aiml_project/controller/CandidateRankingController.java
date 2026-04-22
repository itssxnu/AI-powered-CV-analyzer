package com.cv.aiml_project.controller;

import com.cv.aiml_project.dto.RankingWeightConfig;
import com.cv.aiml_project.entity.CandidateRanking;
import com.cv.aiml_project.entity.HiringStatus;
import com.cv.aiml_project.entity.Job;
import com.cv.aiml_project.service.CandidateRankingService;
import com.cv.aiml_project.service.JobService;
import com.cv.aiml_project.service.SkillMatchResultService;
import com.cv.aiml_project.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Controller
@RequestMapping("/ranking")
public class CandidateRankingController {

    @Autowired
    private CandidateRankingService rankingService;

    @Autowired
    private JobService jobService;

    @Autowired
    private UserService userService;

    @Autowired
    private SkillMatchResultService skillMatchResultService;

    // ==================== MAIN VIEWS ====================

    /**
     * Ranking dashboard - list all jobs with rankings
     */
    @GetMapping("/dashboard")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String rankingDashboard(Model model) {
        List<Job> activeJobs = jobService.getActiveJobs();
        model.addAttribute("activeJobs", activeJobs);
        return "ranking/dashboard";
    }

    /**
     * View rankings for a specific job
     */
    @GetMapping("/job/{jobId}")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String viewJobRankings(@PathVariable Long jobId,
                                  @RequestParam(required = false) String search,
                                  Model model) {
        Job job = jobService.getJobById(jobId)
                .orElseThrow(() -> new IllegalArgumentException("Job not found"));

        List<CandidateRanking> rankings;
        if (search != null && !search.trim().isEmpty()) {
            rankings = rankingService.searchRankings(jobId, search);
            model.addAttribute("search", search);
        } else {
            rankings = rankingService.getCurrentRankingsForJob(jobId);
        }

        Map<String, Object> stats = rankingService.getRankingStatistics(jobId);
        RankingWeightConfig currentWeights = rankingService.getCurrentWeightConfig();

        model.addAttribute("job", job);
        model.addAttribute("rankings", rankings);
        model.addAttribute("stats", stats);
        model.addAttribute("currentWeights", currentWeights);
        model.addAttribute("hiringStatuses", HiringStatus.values());

        return "ranking/job-rankings";
    }

    /**
     * View ranking details
     */
    @GetMapping("/details/{rankingId}")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN', 'CANDIDATE')")
    public String viewRankingDetails(@PathVariable Long rankingId, Model model) {
        CandidateRanking ranking = rankingService.getRankingById(rankingId)
                .orElseThrow(() -> new IllegalArgumentException("Ranking not found"));

        model.addAttribute("ranking", ranking);
        model.addAttribute("job", ranking.getJob());
        model.addAttribute("candidate", ranking.getCandidate());
        model.addAttribute("hiringStatuses", HiringStatus.values());

        // Get ranking history for this candidate on this job
        List<CandidateRanking> history = rankingService.getRankingHistoryForCandidate(
                ranking.getJob().getId(), ranking.getCandidate().getId());
        model.addAttribute("history", history);

        return "ranking/ranking-details";
    }

    /**
     * View candidate's own rankings
     */
    @GetMapping("/my-rankings")
    @PreAuthorize("hasRole('CANDIDATE')")
    public String viewMyRankings(Model model) {
        // Get current candidate
        org.springframework.security.core.Authentication auth =
                org.springframework.security.core.context.SecurityContextHolder.getContext().getAuthentication();
        com.cv.aiml_project.entity.User currentUser = userService.getUserByUsername(auth.getName())
                .orElseThrow(() -> new RuntimeException("User not found"));

        // Get all rankings for this candidate
        List<CandidateRanking> rankings = rankingService.getRankingHistoryForCandidate(null, currentUser.getId());

        model.addAttribute("rankings", rankings);
        return "ranking/my-rankings";
    }

    // ==================== RANKING GENERATION ====================

    /**
     * Show ranking generation page
     */
    @GetMapping("/job/{jobId}/generate")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String showGenerateRankingPage(@PathVariable Long jobId, Model model) {
        Job job = jobService.getJobById(jobId)
                .orElseThrow(() -> new IllegalArgumentException("Job not found"));

        model.addAttribute("job", job);
        model.addAttribute("weightConfig", new RankingWeightConfig());

        return "ranking/generate-ranking";
    }

    /**
     * Generate rankings for a job
     */
    @PostMapping("/job/{jobId}/generate")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String generateRankings(@PathVariable Long jobId,
                                   @ModelAttribute RankingWeightConfig weightConfig,
                                   RedirectAttributes redirectAttributes) {
        try {
            List<CandidateRanking> rankings = rankingService.generateRankingForJob(jobId, weightConfig);
            redirectAttributes.addFlashAttribute("message",
                    "Rankings generated successfully for " + rankings.size() + " candidates");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to generate rankings: " + e.getMessage());
        }
        return "redirect:/ranking/job/" + jobId;
    }

    /**
     * Show recalculate ranking page
     */
    @GetMapping("/job/{jobId}/recalculate")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String showRecalculateRankingPage(@PathVariable Long jobId, Model model) {
        Job job = jobService.getJobById(jobId)
                .orElseThrow(() -> new IllegalArgumentException("Job not found"));

        model.addAttribute("job", job);
        model.addAttribute("weightConfig", rankingService.getCurrentWeightConfig());

        return "ranking/recalculate-ranking";
    }

    /**
     * Recalculate rankings with new weights (rerun)
     */
    @PostMapping("/job/{jobId}/recalculate")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String recalculateRankings(@PathVariable Long jobId,
                                      @ModelAttribute RankingWeightConfig weightConfig,
                                      RedirectAttributes redirectAttributes) {
        try {
            List<CandidateRanking> rankings = rankingService.recalculateRankingWithWeights(jobId, weightConfig);
            redirectAttributes.addFlashAttribute("message",
                    "Rankings recalculated successfully for " + rankings.size() + " candidates");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to recalculate rankings: " + e.getMessage());
        }
        return "redirect:/ranking/job/" + jobId;
    }

    // ==================== UPDATE OPERATIONS ====================

    /**
     * Update shortlist status
     */
    @PostMapping("/{rankingId}/shortlist")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String updateShortlistStatus(@PathVariable Long rankingId,
                                        @RequestParam boolean shortlisted,
                                        @RequestParam(required = false) String notes,
                                        RedirectAttributes redirectAttributes) {
        try {
            CandidateRanking ranking = rankingService.updateShortlistStatus(rankingId, shortlisted, notes);
            String status = shortlisted ? "shortlisted" : "removed from shortlist";
            redirectAttributes.addFlashAttribute("message",
                    "Candidate " + ranking.getCandidate().getFullName() + " " + status);
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to update shortlist status: " + e.getMessage());
        }
        return "redirect:/ranking/details/" + rankingId;
    }

    /**
     * Update hiring status
     */
    @PostMapping("/{rankingId}/hiring-status")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String updateHiringStatus(@PathVariable Long rankingId,
                                     @RequestParam HiringStatus status,
                                     @RequestParam(required = false) String notes,
                                     RedirectAttributes redirectAttributes) {
        try {
            CandidateRanking ranking = rankingService.updateHiringStatus(rankingId, status, notes);
            redirectAttributes.addFlashAttribute("message",
                    "Hiring status updated to " + status.getDisplayName());
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to update hiring status: " + e.getMessage());
        }
        return "redirect:/ranking/details/" + rankingId;
    }

    /**
     * Schedule interview
     */
    @PostMapping("/{rankingId}/schedule-interview")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String scheduleInterview(@PathVariable Long rankingId,
                                    @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime interviewDate,
                                    @RequestParam(required = false) String notes,
                                    RedirectAttributes redirectAttributes) {
        try {
            CandidateRanking ranking = rankingService.scheduleInterview(rankingId, interviewDate, notes);
            redirectAttributes.addFlashAttribute("message",
                    "Interview scheduled for " + ranking.getCandidate().getFullName());
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to schedule interview: " + e.getMessage());
        }
        return "redirect:/ranking/details/" + rankingId;
    }

    /**
     * Add interview feedback
     */
    @PostMapping("/{rankingId}/interview-feedback")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String addInterviewFeedback(@PathVariable Long rankingId,
                                       @RequestParam String feedback,
                                       RedirectAttributes redirectAttributes) {
        try {
            CandidateRanking ranking = rankingService.addInterviewFeedback(rankingId, feedback);
            redirectAttributes.addFlashAttribute("message", "Interview feedback added");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to add feedback: " + e.getMessage());
        }
        return "redirect:/ranking/details/" + rankingId;
    }

    /**
     * Update ranking notes
     */
    @PostMapping("/{rankingId}/notes")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    public String updateNotes(@PathVariable Long rankingId,
                              @RequestParam String notes,
                              RedirectAttributes redirectAttributes) {
        try {
            rankingService.updateNotes(rankingId, notes);
            redirectAttributes.addFlashAttribute("message", "Notes updated");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to update notes: " + e.getMessage());
        }
        return "redirect:/ranking/details/" + rankingId;
    }

    // ==================== DELETE OPERATIONS ====================

    /**
     * Delete ranking entry
     */
    @PostMapping("/{rankingId}/delete")
    @PreAuthorize("hasAnyRole('ADMIN')")
    public String deleteRanking(@PathVariable Long rankingId,
                                @RequestParam Long jobId,
                                RedirectAttributes redirectAttributes) {
        try {
            rankingService.deleteRanking(rankingId);
            redirectAttributes.addFlashAttribute("message", "Ranking entry deleted");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Failed to delete: " + e.getMessage());
        }
        return "redirect:/ranking/job/" + jobId;
    }

    // ==================== API ENDPOINTS ====================

    /**
     * Get rankings as JSON (for AJAX)
     */
    @GetMapping("/api/job/{jobId}")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    @ResponseBody
    public ResponseEntity<?> getRankingsApi(@PathVariable Long jobId) {
        try {
            List<CandidateRanking> rankings = rankingService.getCurrentRankingsForJob(jobId);
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

    /**
     * Validate weight configuration
     */
    @PostMapping("/api/validate-weights")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    @ResponseBody
    public ResponseEntity<?> validateWeights(@RequestBody RankingWeightConfig weights) {
        boolean isValid = rankingService.validateWeightConfig(weights);
        return ResponseEntity.ok(Map.of(
                "valid", isValid,
                "message", isValid ? "Valid configuration" : weights.getValidationMessage()
        ));
    }

    // ==================== EXPORT ====================

    /**
     * Export rankings as CSV
     */
    @GetMapping("/job/{jobId}/export")
    @PreAuthorize("hasAnyRole('HR', 'ADMIN')")
    @ResponseBody
    public ResponseEntity<String> exportRankings(@PathVariable Long jobId) {
        try {
            List<CandidateRanking> rankings = rankingService.getCurrentRankingsForJob(jobId);
            Job job = jobService.getJobById(jobId).orElseThrow();

            StringBuilder csv = new StringBuilder();
            csv.append("Rank, Candidate Name, Email, Ranking Score, Skills Score, Experience Score, ")
                    .append("Education Score, Shortlisted, Hiring Status, Percentile\n");

            for (CandidateRanking ranking : rankings) {
                csv.append(ranking.getRankPosition()).append(",")
                        .append("\"").append(ranking.getCandidate().getFullName()).append("\",")
                        .append("\"").append(ranking.getCandidate().getEmail()).append("\",")
                        .append(String.format("%.1f", ranking.getRankingScore())).append(",")
                        .append(String.format("%.1f", ranking.getWeightedSkillsScore())).append(",")
                        .append(String.format("%.1f", ranking.getWeightedExperienceScore())).append(",")
                        .append(String.format("%.1f", ranking.getWeightedEducationScore())).append(",")
                        .append(ranking.isShortlisted() ? "Yes" : "No").append(",")
                        .append(ranking.getHiringStatus() != null ? ranking.getHiringStatus().getDisplayName() : "N/A").append(",")
                        .append(String.format("%.1f", ranking.getPercentile()))
                        .append("\n");
            }

            String filename = "rankings_job_" + jobId + "_" + job.getTitle().replace(" ", "_") + ".csv";

            return ResponseEntity.ok()
                    .header("Content-Disposition", "attachment; filename=\"" + filename + "\"")
                    .header("Content-Type", "text/csv")
                    .body(csv.toString());

        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Error exporting data: " + e.getMessage());
        }
    }
}
