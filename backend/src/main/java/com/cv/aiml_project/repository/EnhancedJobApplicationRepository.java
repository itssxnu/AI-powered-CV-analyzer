package com.cv.aiml_project.repository;

import com.cv.aiml_project.entity.ApplicationStatus;
import com.cv.aiml_project.entity.JobApplication;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Repository
public interface EnhancedJobApplicationRepository extends JpaRepository<JobApplication, Long> {

    // Advanced filtering
    @Query("SELECT ja FROM JobApplication ja WHERE " +
            "(:jobId IS NULL OR ja.job.id = :jobId) AND " +
            "(:status IS NULL OR ja.status = :status) AND " +
            "(:fromDate IS NULL OR ja.appliedDate >= :fromDate) AND " +
            "(:toDate IS NULL OR ja.appliedDate <= :toDate) AND " +
            "(:minScore IS NULL OR ja.matchScore >= :minScore) AND " +
            "(:shortlistedOnly IS NULL OR :shortlistedOnly = false OR ja.status = 'SHORTLISTED') AND " +
            "(:interviewedOnly IS NULL OR :interviewedOnly = false OR ja.interviewScheduled = true) AND " +
            "(:searchKeyword IS NULL OR LOWER(ja.candidate.firstName) LIKE LOWER(CONCAT('%', :searchKeyword, '%')) OR " +
            "LOWER(ja.candidate.lastName) LIKE LOWER(CONCAT('%', :searchKeyword, '%')) OR " +
            "LOWER(ja.candidate.email) LIKE LOWER(CONCAT('%', :searchKeyword, '%')))")
    List<JobApplication> findApplicationsByFilters(
            @Param("jobId") Long jobId,
            @Param("status") ApplicationStatus status,
            @Param("fromDate") LocalDateTime fromDate,
            @Param("toDate") LocalDateTime toDate,
            @Param("minScore") Double minScore,
            @Param("shortlistedOnly") Boolean shortlistedOnly,
            @Param("interviewedOnly") Boolean interviewedOnly,
            @Param("searchKeyword") String searchKeyword);

    // Paginated version
    @Query("SELECT ja FROM JobApplication ja WHERE " +
            "(:jobId IS NULL OR ja.job.id = :jobId) AND " +
            "(:status IS NULL OR ja.status = :status)")
    Page<JobApplication> findApplicationsByJobAndStatus(
            @Param("jobId") Long jobId,
            @Param("status") ApplicationStatus status,
            Pageable pageable);

    // Shortlist operations
    @Query("SELECT ja FROM JobApplication ja WHERE ja.job.id = :jobId AND ja.status = 'SHORTLISTED' ORDER BY ja.matchScore DESC")
    List<JobApplication> findShortlistedForJob(@Param("jobId") Long jobId);

    @Query("SELECT ja FROM JobApplication ja WHERE ja.candidate.id = :candidateId AND ja.status = 'SHORTLISTED'")
    List<JobApplication> findShortlistedByCandidate(@Param("candidateId") Long candidateId);

    // Interview operations
    @Query("SELECT ja FROM JobApplication ja WHERE ja.interviewScheduled = true AND ja.interviewDate BETWEEN :startDate AND :endDate")
    List<JobApplication> findInterviewsInDateRange(@Param("startDate") LocalDateTime startDate, @Param("endDate") LocalDateTime endDate);

    @Query("SELECT ja FROM JobApplication ja WHERE ja.interviewScheduled = true AND ja.interviewerEmail = :interviewerEmail")
    List<JobApplication> findInterviewsByInterviewer(@Param("interviewerEmail") String interviewerEmail);

    @Query("SELECT ja FROM JobApplication ja WHERE ja.interviewScheduled = true AND ja.interviewDate < :currentDate")
    List<JobApplication> findPastInterviews(@Param("currentDate") LocalDateTime currentDate);

    @Query("SELECT ja FROM JobApplication ja WHERE ja.interviewScheduled = true AND ja.interviewDate > :currentDate ORDER BY ja.interviewDate")
    List<JobApplication> findUpcomingInterviews(@Param("currentDate") LocalDateTime currentDate);

    // Statistics and analytics
    @Query("SELECT ja.status, COUNT(ja) FROM JobApplication ja WHERE ja.job.id = :jobId GROUP BY ja.status")
    List<Object[]> getApplicationStatusCounts(@Param("jobId") Long jobId);

    @Query("SELECT DATE(ja.appliedDate), COUNT(ja) FROM JobApplication ja WHERE ja.job.id = :jobId AND ja.appliedDate BETWEEN :startDate AND :endDate GROUP BY DATE(ja.appliedDate)")
    List<Object[]> getApplicationTrends(@Param("jobId") Long jobId, @Param("startDate") LocalDateTime startDate, @Param("endDate") LocalDateTime endDate);

    @Query("SELECT AVG(ja.matchScore) FROM JobApplication ja WHERE ja.job.id = :jobId AND ja.matchScore IS NOT NULL")
    Double getAverageMatchScoreForJob(@Param("jobId") Long jobId);

    @Query("SELECT ja.status, AVG(ja.matchScore) FROM JobApplication ja WHERE ja.job.id = :jobId GROUP BY ja.status")
    List<Object[]> getAverageScoreByStatus(@Param("jobId") Long jobId);

    // Bulk operations
    @Modifying
    @Query("UPDATE JobApplication ja SET ja.status = :newStatus WHERE ja.id IN :applicationIds")
    int bulkUpdateStatus(@Param("applicationIds") List<Long> applicationIds, @Param("newStatus") ApplicationStatus newStatus);

    @Modifying
    @Query("UPDATE JobApplication ja SET ja.isActive = false WHERE ja.job.id = :jobId AND ja.status = 'REJECTED'")
    int removeRejectedApplications(@Param("jobId") Long jobId);

    // Time-based queries
    @Query("SELECT ja FROM JobApplication ja WHERE ja.status = 'PENDING' AND ja.appliedDate < :thresholdDate")
    List<JobApplication> findStaleApplications(@Param("thresholdDate") LocalDateTime thresholdDate);

    @Query("SELECT ja FROM JobApplication ja WHERE ja.job.id = :jobId AND ja.status IN ('SHORTLISTED', 'INTERVIEW_SCHEDULED')")
    List<JobApplication> findActiveCandidatesForJob(@Param("jobId") Long jobId);

    // Candidate history
    @Query("SELECT ja FROM JobApplication ja WHERE ja.candidate.id = :candidateId AND ja.status != 'WITHDRAWN' ORDER BY ja.appliedDate DESC")
    List<JobApplication> findActiveApplicationsByCandidate(@Param("candidateId") Long candidateId);

    // Job statistics
    @Query("SELECT j.id, j.title, COUNT(ja) FROM Job j LEFT JOIN JobApplication ja ON j.id = ja.job.id WHERE j.isActive = true GROUP BY j.id, j.title")
    List<Object[]> getApplicationCountsByJob();
}