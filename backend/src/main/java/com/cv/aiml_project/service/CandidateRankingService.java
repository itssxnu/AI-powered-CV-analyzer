package com.cv.aiml_project.service;

import com.cv.aiml_project.dto.RankingWeightConfig;
import com.cv.aiml_project.entity.*;
import com.cv.aiml_project.repository.CandidateRankingRepository;
import com.cv.aiml_project.repository.JobRepository;
import com.cv.aiml_project.repository.SkillMatchResultRepository;
import com.cv.aiml_project.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

@Service
@Transactional
public class CandidateRankingService {

    @Autowired
    private CandidateRankingRepository rankingRepository;

    @Autowired
    private JobRepository jobRepository;

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private SkillMatchResultRepository skillMatchResultRepository;

    @Autowired
    private SkillMatchResultService skillMatchResultService;

    // Default weight configuration
    private static final RankingWeightConfig DEFAULT_WEIGHTS = new RankingWeightConfig();

    // ==================== RANKING GENERATION ====================

    /**
     * Generate ranking scores for all candidates for a job
     */
    @Transactional
    public List<CandidateRanking> generateRankingForJob(Long jobId, RankingWeightConfig weights) {
        Job job = jobRepository.findById(jobId)
                .orElseThrow(() -> new RuntimeException("Job not found with id: " + jobId));

        // Validate weights
        if (weights == null) {
            weights = DEFAULT_WEIGHTS;
        }
        if (!weights.isValid()) {
            throw new RuntimeException("Invalid weight configuration: " + weights.getValidationMessage());
        }

        // Get all skill match results for this job
        List<SkillMatchResult> matchResults = skillMatchResultService.getLatestMatchResultsForJob(jobId);

        if (matchResults.isEmpty()) {
            throw new RuntimeException("No skill match results found for this job. Please run skill matching first.");
        }

        // Set all existing rankings as not current
        rankingRepository.setAllRankingsNotCurrent(jobId);

        List<CandidateRanking> rankings = new ArrayList<>();

        for (SkillMatchResult matchResult : matchResults) {
            CandidateRanking ranking = createRankingFromMatchResult(matchResult, weights);
            rankings.add(ranking);
        }

        // Save all rankings
        rankings = rankingRepository.saveAll(rankings);

        // Calculate rank positions
        rankings = calculateRankPositions(rankings, jobId);

        return rankings;
    }

    /**
     * Create ranking from skill match result
     */
    private CandidateRanking createRankingFromMatchResult(SkillMatchResult matchResult, RankingWeightConfig weights) {
        CandidateRanking ranking = new CandidateRanking();
        ranking.setJob(matchResult.getJob());
        ranking.setCandidate(matchResult.getCandidate());
        ranking.setSkillMatchResult(matchResult);
        ranking.setCurrentRanking(true);
        ranking.setRankingDate(LocalDateTime.now());
        ranking.setRankingCriteriaVersion("v1.0");

        // Set weights used
        ranking.setSkillsWeight(weights.getSkillsWeight());
        ranking.setExperienceWeight(weights.getExperienceWeight());
        ranking.setEducationWeight(weights.getEducationWeight());
        ranking.setPersonalityWeight(weights.getPersonalityWeight());
        ranking.setCulturalFitWeight(weights.getCulturalFitWeight());

        // Calculate weighted scores
        calculateWeightedScores(ranking, matchResult, weights);

        // Calculate overall ranking score
        calculateRankingScore(ranking);

        return ranking;
    }

    /**
     * Calculate weighted scores for each component
     */
    private void calculateWeightedScores(CandidateRanking ranking, SkillMatchResult matchResult, RankingWeightConfig weights) {
        // Skills weighted score
        if (matchResult.getSkillsScore() != null && weights.getSkillsWeight() > 0) {
            ranking.setWeightedSkillsScore(matchResult.getSkillsScore() * (weights.getSkillsWeight() / 100.0));
        } else {
            ranking.setWeightedSkillsScore(0.0);
        }

        // Experience weighted score
        if (matchResult.getExperienceScore() != null && weights.getExperienceWeight() > 0) {
            ranking.setWeightedExperienceScore(matchResult.getExperienceScore() * (weights.getExperienceWeight() / 100.0));
        } else {
            ranking.setWeightedExperienceScore(0.0);
        }

        // Education weighted score
        if (matchResult.getEducationScore() != null && weights.getEducationWeight() > 0) {
            ranking.setWeightedEducationScore(matchResult.getEducationScore() * (weights.getEducationWeight() / 100.0));
        } else {
            ranking.setWeightedEducationScore(0.0);
        }

        // Personality weighted score (if available)
        if (matchResult.getPersonalityScore() != null && weights.getPersonalityWeight() != null && weights.getPersonalityWeight() > 0) {
            ranking.setWeightedPersonalityScore(matchResult.getPersonalityScore() * (weights.getPersonalityWeight() / 100.0));
        } else {
            ranking.setWeightedPersonalityScore(0.0);
        }

        // Cultural fit weighted score (if available)
        if (matchResult.getCulturalFitScore() != null && weights.getCulturalFitWeight() != null && weights.getCulturalFitWeight() > 0) {
            ranking.setWeightedCulturalFitScore(matchResult.getCulturalFitScore() * (weights.getCulturalFitWeight() / 100.0));
        } else {
            ranking.setWeightedCulturalFitScore(0.0);
        }
    }

    /**
     * Calculate overall ranking score
     */
    private void calculateRankingScore(CandidateRanking ranking) {
        double total = 0.0;

        total += ranking.getWeightedSkillsScore() != null ? ranking.getWeightedSkillsScore() : 0.0;
        total += ranking.getWeightedExperienceScore() != null ? ranking.getWeightedExperienceScore() : 0.0;
        total += ranking.getWeightedEducationScore() != null ? ranking.getWeightedEducationScore() : 0.0;
        total += ranking.getWeightedPersonalityScore() != null ? ranking.getWeightedPersonalityScore() : 0.0;
        total += ranking.getWeightedCulturalFitScore() != null ? ranking.getWeightedCulturalFitScore() : 0.0;

        ranking.setRankingScore(total);
    }

    /**
     * Calculate and assign rank positions
     */
    @Transactional
    public List<CandidateRanking> calculateRankPositions(List<CandidateRanking> rankings, Long jobId) {
        // Sort by ranking score descending
        rankings.sort((r1, r2) -> {
            if (r1.getRankingScore() == null && r2.getRankingScore() == null) return 0;
            if (r1.getRankingScore() == null) return 1;
            if (r2.getRankingScore() == null) return -1;
            return r2.getRankingScore().compareTo(r1.getRankingScore());
        });

        int total = rankings.size();

        for (int i = 0; i < rankings.size(); i++) {
            CandidateRanking ranking = rankings.get(i);
            Integer previousRank = ranking.getRankPosition();
            Integer newRank = i + 1;

            // Set rank position
            ranking.setRankPosition(newRank);
            ranking.setTotalCandidatesRanked(total);

            // Calculate rank change
            if (previousRank != null) {
                ranking.setRankChange(previousRank - newRank);
            }

            // Calculate percentile
            if (total > 0) {
                double percentile = ((total - i - 1) * 100.0) / total;
                ranking.setPercentile(percentile);
            }
        }

        return rankingRepository.saveAll(rankings);
    }

    /**
     * Recalculate ranking with new weights (rerun)
     */
    @Transactional
    public List<CandidateRanking> recalculateRankingWithWeights(Long jobId, RankingWeightConfig newWeights) {
        // Validate new weights
        if (newWeights == null || !newWeights.isValid()) {
            throw new RuntimeException("Invalid weight configuration");
        }

        // Get current skill match results
        List<SkillMatchResult> matchResults = skillMatchResultService.getLatestMatchResultsForJob(jobId);

        if (matchResults.isEmpty()) {
            throw new RuntimeException("No skill match results found for this job");
        }

        // Store previous rankings for change tracking
        Map<Long, Integer> previousRankings = new HashMap<>();
        List<CandidateRanking> existingRankings = rankingRepository.findCurrentRankingsByJob(jobId);
        for (CandidateRanking ranking : existingRankings) {
            previousRankings.put(ranking.getCandidate().getId(), ranking.getRankPosition());
        }

        // Set all existing rankings as not current
        rankingRepository.setAllRankingsNotCurrent(jobId);

        // Create new rankings with new weights
        List<CandidateRanking> newRankings = new ArrayList<>();
        for (SkillMatchResult matchResult : matchResults) {
            CandidateRanking ranking = createRankingFromMatchResult(matchResult, newWeights);

            // Set previous rank for change tracking
            Integer previousRank = previousRankings.get(matchResult.getCandidate().getId());
            ranking.setPreviousRankPosition(previousRank);

            newRankings.add(ranking);
        }

        newRankings = rankingRepository.saveAll(newRankings);

        // Calculate new rank positions (this will also calculate rank changes)
        newRankings = calculateRankPositions(newRankings, jobId);

        return newRankings;
    }

    // ==================== READ OPERATIONS ====================

    /**
     * Get current rankings for a job
     */
    public List<CandidateRanking> getCurrentRankingsForJob(Long jobId) {
        return rankingRepository.findCurrentRankingsByJob(jobId);
    }

    /**
     * Get ranking by ID
     */
    public Optional<CandidateRanking> getRankingById(Long rankingId) {
        return rankingRepository.findById(rankingId);
    }

    /**
     * Get current ranking for specific job and candidate
     */
    public Optional<CandidateRanking> getCurrentRankingForJobAndCandidate(Long jobId, Long candidateId) {
        Job job = jobRepository.getReferenceById(jobId);
        User candidate = userRepository.getReferenceById(candidateId);
        return rankingRepository.findByJobAndCandidateAndIsCurrentRankingTrue(job, candidate);
    }

    /**
     * Get top N ranked candidates for a job
     */
    public List<CandidateRanking> getTopRankedCandidates(Long jobId, int topN) {
        return rankingRepository.findTopRankedCandidates(jobId, topN);
    }

    /**
     * Get ranking history for a candidate on a specific job
     */
    public List<CandidateRanking> getRankingHistoryForCandidate(Long jobId, Long candidateId) {
        if (jobId == null) {
            // Used by candidate's "my-rankings" page — fetch across all jobs
            return rankingRepository.findAllRankingsForCandidate(candidateId);
        }
        return rankingRepository.findRankingHistoryForCandidate(candidateId, jobId);
    }

    /**
     * Get rankings by minimum score
     */
    public List<CandidateRanking> getRankingsByMinimumScore(Long jobId, Double minScore) {
        return rankingRepository.findByMinimumScore(jobId, minScore);
    }

    /**
     * Get shortlisted candidates
     */
    public List<CandidateRanking> getShortlistedCandidates(Long jobId) {
        return rankingRepository.findShortlistedForJob(jobId);
    }

    /**
     * Search rankings
     */
    public List<CandidateRanking> searchRankings(Long jobId, String keyword) {
        if (keyword == null || keyword.trim().isEmpty()) {
            return getCurrentRankingsForJob(jobId);
        }
        return rankingRepository.searchRankings(jobId, keyword);
    }

    // ==================== UPDATE OPERATIONS ====================

    /**
     * Update shortlist status
     */
    @Transactional
    public CandidateRanking updateShortlistStatus(Long rankingId, boolean shortlisted, String notes) {
        CandidateRanking ranking = rankingRepository.findById(rankingId)
                .orElseThrow(() -> new RuntimeException("Ranking not found with id: " + rankingId));

        ranking.setShortlisted(shortlisted);
        if (shortlisted) {
            ranking.setShortlistDate(LocalDateTime.now());
            ranking.setShortlistNotes(notes);
            ranking.setHiringStatus(HiringStatus.SHORTLISTED);
        } else {
            ranking.setShortlistDate(null);
            ranking.setShortlistNotes(null);
            ranking.setHiringStatus(HiringStatus.UNDER_REVIEW);
        }

        return rankingRepository.save(ranking);
    }

    /**
     * Update hiring status
     */
    @Transactional
    public CandidateRanking updateHiringStatus(Long rankingId, HiringStatus status, String notes) {
        CandidateRanking ranking = rankingRepository.findById(rankingId)
                .orElseThrow(() -> new RuntimeException("Ranking not found with id: " + rankingId));

        ranking.setHiringStatus(status);
        ranking.setHiringDecisionDate(LocalDateTime.now());

        if (status == HiringStatus.SHORTLISTED) {
            ranking.setShortlisted(true);
            ranking.setShortlistDate(LocalDateTime.now());
            ranking.setShortlistNotes(notes);
        } else if (status == HiringStatus.REJECTED || status == HiringStatus.OFFER_DECLINED) {
            ranking.setShortlisted(false);
        }

        return rankingRepository.save(ranking);
    }

    /**
     * Schedule interview
     */
    @Transactional
    public CandidateRanking scheduleInterview(Long rankingId, LocalDateTime interviewDate, String notes) {
        CandidateRanking ranking = rankingRepository.findById(rankingId)
                .orElseThrow(() -> new RuntimeException("Ranking not found with id: " + rankingId));

        ranking.setInterviewScheduled(true);
        ranking.setInterviewDate(interviewDate);
        ranking.setNotes(notes);
        ranking.setHiringStatus(HiringStatus.INTERVIEWED);

        return rankingRepository.save(ranking);
    }

    /**
     * Add interview feedback
     */
    @Transactional
    public CandidateRanking addInterviewFeedback(Long rankingId, String feedback) {
        CandidateRanking ranking = rankingRepository.findById(rankingId)
                .orElseThrow(() -> new RuntimeException("Ranking not found with id: " + rankingId));

        ranking.setInterviewFeedback(feedback);

        return rankingRepository.save(ranking);
    }

    /**
     * Update ranking notes
     */
    @Transactional
    public CandidateRanking updateNotes(Long rankingId, String notes) {
        CandidateRanking ranking = rankingRepository.findById(rankingId)
                .orElseThrow(() -> new RuntimeException("Ranking not found with id: " + rankingId));

        ranking.setNotes(notes);

        return rankingRepository.save(ranking);
    }

    // ==================== DELETE OPERATIONS ====================

    /**
     * Delete ranking entry
     */
    @Transactional
    public void deleteRanking(Long rankingId) {
        CandidateRanking ranking = rankingRepository.findById(rankingId)
                .orElseThrow(() -> new RuntimeException("Ranking not found with id: " + rankingId));

        rankingRepository.delete(ranking);
    }

    /**
     * Delete all rankings for a job
     */
    @Transactional
    public void deleteAllRankingsForJob(Long jobId) {
        rankingRepository.deleteByJobId(jobId);
    }

    // ==================== STATISTICS AND ANALYTICS ====================

    /**
     * Get ranking statistics for a job
     */
    public Map<String, Object> getRankingStatistics(Long jobId) {
        Map<String, Object> stats = new HashMap<>();

        List<CandidateRanking> rankings = rankingRepository.findCurrentRankingsByJob(jobId);

        stats.put("totalRanked", rankings.size());

        // Score statistics
        Double avgScore = rankingRepository.getAverageRankingScoreForJob(jobId);
        stats.put("averageScore", avgScore != null ? avgScore : 0.0);

        OptionalDouble maxScore = rankings.stream()
                .mapToDouble(r -> r.getRankingScore() != null ? r.getRankingScore() : 0.0)
                .max();
        stats.put("maxScore", maxScore.isPresent() ? maxScore.getAsDouble() : 0.0);

        OptionalDouble minScore = rankings.stream()
                .mapToDouble(r -> r.getRankingScore() != null ? r.getRankingScore() : 0.0)
                .min();
        stats.put("minScore", minScore.isPresent() ? minScore.getAsDouble() : 0.0);

        // Status counts
        long shortlisted = rankings.stream().filter(CandidateRanking::isShortlisted).count();
        long interviewed = rankings.stream().filter(r -> r.isInterviewScheduled()).count();
        long hired = rankings.stream()
                .filter(r -> r.getHiringStatus() == HiringStatus.HIRED)
                .count();

        stats.put("shortlisted", shortlisted);
        stats.put("interviewed", interviewed);
        stats.put("hired", hired);

        // Score distribution
        long excellent = rankings.stream()
                .filter(r -> r.getRankingScore() != null && r.getRankingScore() >= 80)
                .count();
        long good = rankings.stream()
                .filter(r -> r.getRankingScore() != null && r.getRankingScore() >= 60 && r.getRankingScore() < 80)
                .count();
        long average = rankings.stream()
                .filter(r -> r.getRankingScore() != null && r.getRankingScore() >= 40 && r.getRankingScore() < 60)
                .count();
        long poor = rankings.stream()
                .filter(r -> r.getRankingScore() != null && r.getRankingScore() < 40)
                .count();

        stats.put("excellentMatches", excellent);
        stats.put("goodMatches", good);
        stats.put("averageMatches", average);
        stats.put("poorMatches", poor);

        return stats;
    }

    /**
     * Get weight configuration summary
     */
    public RankingWeightConfig getCurrentWeightConfig() {
        return DEFAULT_WEIGHTS;
    }

    /**
     * Validate weight configuration
     */
    public boolean validateWeightConfig(RankingWeightConfig config) {
        return config != null && config.isValid();
    }
}
