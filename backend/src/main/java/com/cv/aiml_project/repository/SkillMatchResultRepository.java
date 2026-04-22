package com.cv.aiml_project.repository;

import com.cv.aiml_project.entity.Job;
import com.cv.aiml_project.entity.SkillMatchResult;
import com.cv.aiml_project.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface SkillMatchResultRepository extends JpaRepository<SkillMatchResult, Long> {

    // Find results by job
    List<SkillMatchResult> findByJob(Job job);

    List<SkillMatchResult> findByJobOrderByOverallScoreDesc(Job job);

    @Query("SELECT smr FROM SkillMatchResult smr WHERE smr.job.id = :jobId AND smr.isLatest = true ORDER BY smr.overallScore DESC")
    List<SkillMatchResult> findLatestByJobOrderByScoreDesc(@Param("jobId") Long jobId);

    // Find results by candidate
    List<SkillMatchResult> findByCandidate(User candidate);

    List<SkillMatchResult> findByCandidateOrderByMatchDateDesc(User candidate);

    // Find specific match
    Optional<SkillMatchResult> findByJobAndCandidateAndIsLatestTrue(Job job, User candidate);

    @Query("SELECT smr FROM SkillMatchResult smr WHERE smr.job.id = :jobId AND smr.candidate.id = :candidateId ORDER BY smr.matchDate DESC")
    List<SkillMatchResult> findByJobAndCandidateOrderByMatchDateDesc(@Param("jobId") Long jobId, @Param("candidateId") Long candidateId);

    // Find top matches for a job
    @Query("SELECT smr FROM SkillMatchResult smr WHERE smr.job.id = :jobId AND smr.isLatest = true AND smr.overallScore >= :minScore ORDER BY smr.overallScore DESC")
    List<SkillMatchResult> findTopMatchesForJob(@Param("jobId") Long jobId, @Param("minScore") Double minScore);

    // Statistics
    @Query("SELECT COUNT(smr) FROM SkillMatchResult smr WHERE smr.job.id = :jobId")
    long countByJobId(@Param("jobId") Long jobId);

    @Query("SELECT AVG(smr.overallScore) FROM SkillMatchResult smr WHERE smr.job.id = :jobId AND smr.overallScore IS NOT NULL")
    Double getAverageScoreForJob(@Param("jobId") Long jobId);

    @Query("SELECT MAX(smr.overallScore) FROM SkillMatchResult smr WHERE smr.job.id = :jobId")
    Double getMaxScoreForJob(@Param("jobId") Long jobId);

    @Query("SELECT MIN(smr.overallScore) FROM SkillMatchResult smr WHERE smr.job.id = :jobId AND smr.overallScore IS NOT NULL")
    Double getMinScoreForJob(@Param("jobId") Long jobId);

    // Find results with scores in range
    @Query("SELECT smr FROM SkillMatchResult smr WHERE smr.job.id = :jobId AND smr.overallScore BETWEEN :minScore AND :maxScore")
    List<SkillMatchResult> findByScoreRange(@Param("jobId") Long jobId, @Param("minScore") Double minScore, @Param("maxScore") Double maxScore);

    // Find processed/unprocessed results
    List<SkillMatchResult> findByAiProcessed(boolean aiProcessed);

    @Query("SELECT smr FROM SkillMatchResult smr WHERE smr.job.id = :jobId AND smr.aiProcessed = false")
    List<SkillMatchResult> findUnprocessedForJob(@Param("jobId") Long jobId);

    // Delete operations
    @Modifying
    @Query("UPDATE SkillMatchResult smr SET smr.isActive = false WHERE smr.job.id = :jobId")
    void deactivateAllForJob(@Param("jobId") Long jobId);

    @Modifying
    @Query("DELETE FROM SkillMatchResult smr WHERE smr.job.id = :jobId")
    void deleteByJobId(@Param("jobId") Long jobId);

    @Modifying
    @Query("DELETE FROM SkillMatchResult smr WHERE smr.candidate.id = :candidateId")
    void deleteByCandidateId(@Param("candidateId") Long candidateId);

    @Modifying
    @Query("UPDATE SkillMatchResult smr SET smr.isLatest = false WHERE smr.job.id = :jobId AND smr.candidate.id = :candidateId")
    void setNotLatestForJobAndCandidate(@Param("jobId") Long jobId, @Param("candidateId") Long candidateId);
}
