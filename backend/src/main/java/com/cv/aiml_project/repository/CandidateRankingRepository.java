package com.cv.aiml_project.repository;

import com.cv.aiml_project.entity.CandidateRanking;
import com.cv.aiml_project.entity.HiringStatus;
import com.cv.aiml_project.entity.Job;
import com.cv.aiml_project.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface CandidateRankingRepository extends JpaRepository<CandidateRanking, Long> {

    // Find rankings by job
    List<CandidateRanking> findByJobOrderByRankPosition(Job job);

    @Query("SELECT cr FROM CandidateRanking cr WHERE cr.job.id = :jobId AND cr.isCurrentRanking = true ORDER BY cr.rankPosition")
    List<CandidateRanking> findCurrentRankingsByJob(@Param("jobId") Long jobId);

    @Query("SELECT cr FROM CandidateRanking cr WHERE cr.job.id = :jobId ORDER BY cr.rankingScore DESC")
    List<CandidateRanking> findByJobOrderByRankingScoreDesc(@Param("jobId") Long jobId);

    // Find rankings by candidate
    List<CandidateRanking> findByCandidateOrderByRankingDateDesc(User candidate);

    // Find specific ranking
    Optional<CandidateRanking> findByJobAndCandidateAndIsCurrentRankingTrue(Job job, User candidate);

    List<CandidateRanking> findByJobAndCandidateOrderByRankingDateDesc(Job job, User candidate);

    // Find top ranked candidates
    @Query("SELECT cr FROM CandidateRanking cr WHERE cr.job.id = :jobId AND cr.rankPosition <= :topN AND cr.isCurrentRanking = true ORDER BY cr.rankPosition")
    List<CandidateRanking> findTopRankedCandidates(@Param("jobId") Long jobId, @Param("topN") int topN);

    // Find shortlisted candidates
    @Query("SELECT cr FROM CandidateRanking cr WHERE cr.job.id = :jobId AND cr.isShortlisted = true AND cr.isCurrentRanking = true")
    List<CandidateRanking> findShortlistedForJob(@Param("jobId") Long jobId);

    // Find by hiring status
    List<CandidateRanking> findByJobAndHiringStatus(Job job, HiringStatus status);

    // Statistics
    @Query("SELECT COUNT(cr) FROM CandidateRanking cr WHERE cr.job.id = :jobId AND cr.isCurrentRanking = true")
    long countRankingsForJob(@Param("jobId") Long jobId);

    @Query("SELECT AVG(cr.rankingScore) FROM CandidateRanking cr WHERE cr.job.id = :jobId AND cr.isCurrentRanking = true")
    Double getAverageRankingScoreForJob(@Param("jobId") Long jobId);

    @Query("SELECT cr FROM CandidateRanking cr WHERE cr.job.id = :jobId AND cr.rankingScore >= :minScore")
    List<CandidateRanking> findByMinimumScore(@Param("jobId") Long jobId, @Param("minScore") Double minScore);

    // Ranking history — for a specific job
    @Query("SELECT cr FROM CandidateRanking cr WHERE cr.candidate.id = :candidateId AND cr.job.id = :jobId ORDER BY cr.rankingDate DESC")
    List<CandidateRanking> findRankingHistoryForCandidate(@Param("candidateId") Long candidateId, @Param("jobId") Long jobId);

    // Ranking history — across ALL jobs (used for candidate's my-rankings page)
    @Query("SELECT cr FROM CandidateRanking cr WHERE cr.candidate.id = :candidateId AND cr.isCurrentRanking = true ORDER BY cr.rankingDate DESC")
    List<CandidateRanking> findAllRankingsForCandidate(@Param("candidateId") Long candidateId);

    // Update operations
    @Modifying
    @Query("UPDATE CandidateRanking cr SET cr.isCurrentRanking = false WHERE cr.job.id = :jobId")
    void setAllRankingsNotCurrent(@Param("jobId") Long jobId);

    @Modifying
    @Query("UPDATE CandidateRanking cr SET cr.isCurrentRanking = false WHERE cr.job.id = :jobId AND cr.candidate.id = :candidateId")
    void setRankingsNotCurrentForCandidate(@Param("jobId") Long jobId, @Param("candidateId") Long candidateId);

    // Delete operations
    @Modifying
    @Query("DELETE FROM CandidateRanking cr WHERE cr.job.id = :jobId")
    void deleteByJobId(@Param("jobId") Long jobId);

    @Modifying
    @Query("DELETE FROM CandidateRanking cr WHERE cr.candidate.id = :candidateId")
    void deleteByCandidateId(@Param("candidateId") Long candidateId);

    // Search and filter
    @Query("SELECT cr FROM CandidateRanking cr WHERE cr.job.id = :jobId AND " +
            "(LOWER(cr.candidate.firstName) LIKE LOWER(CONCAT('%', :keyword, '%')) OR " +
            "LOWER(cr.candidate.lastName) LIKE LOWER(CONCAT('%', :keyword, '%')) OR " +
            "LOWER(cr.candidate.email) LIKE LOWER(CONCAT('%', :keyword, '%')))")
    List<CandidateRanking> searchRankings(@Param("jobId") Long jobId, @Param("keyword") String keyword);
}
