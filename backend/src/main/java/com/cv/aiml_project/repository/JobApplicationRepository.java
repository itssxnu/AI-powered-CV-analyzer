package com.cv.aiml_project.repository;

import com.cv.aiml_project.entity.ApplicationStatus;
import com.cv.aiml_project.entity.Job;
import com.cv.aiml_project.entity.JobApplication;
import com.cv.aiml_project.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface JobApplicationRepository extends JpaRepository<JobApplication, Long> {

    List<JobApplication> findByJob(Job job);

    List<JobApplication> findByCandidate(User candidate);

    Optional<JobApplication> findByJobAndCandidate(Job job, User candidate);

    List<JobApplication> findByJobAndStatus(Job job, ApplicationStatus status);

    List<JobApplication> findByCandidateAndStatus(User candidate, ApplicationStatus status);

    @Query("SELECT ja FROM JobApplication ja WHERE ja.job.id = :jobId AND ja.isActive = true " +
            "ORDER BY ja.matchScore DESC NULLS LAST")
    List<JobApplication> findByJobOrderByMatchScoreDesc(@Param("jobId") Long jobId);

    @Query("SELECT ja FROM JobApplication ja WHERE ja.candidate.id = :candidateId " +
            "ORDER BY ja.appliedDate DESC")
    List<JobApplication> findByCandidateOrderByAppliedDateDesc(@Param("candidateId") Long candidateId);

    @Query("SELECT COUNT(ja) FROM JobApplication ja WHERE ja.job.id = :jobId AND ja.isActive = true")
    long countByJobId(@Param("jobId") Long jobId);

    @Query("SELECT COUNT(ja) FROM JobApplication ja WHERE ja.job.id = :jobId AND ja.status = :status AND ja.isActive = true")
    long countByJobIdAndStatus(@Param("jobId") Long jobId, @Param("status") ApplicationStatus status);

    @Query("SELECT ja FROM JobApplication ja WHERE ja.candidate = :candidate AND ja.isActive = true")
    List<JobApplication> findActiveApplicationsByCandidate(@Param("candidate") User candidate);


    @Query("SELECT ja FROM JobApplication ja WHERE ja.job.id = :jobId AND ja.isActive = true ORDER BY ja.matchScore DESC")
    List<JobApplication> findActiveApplicationsByJobOrderByScoreDesc(@Param("jobId") Long jobId);

    @Query("SELECT ja FROM JobApplication ja WHERE ja.job.id = :jobId AND ja.status = :status ORDER BY ja.matchScore DESC")
    List<JobApplication> findByJobAndStatusOrderByMatchScoreDesc(@Param("jobId") Long jobId, @Param("status") ApplicationStatus status);

    boolean existsByJobAndCandidate(Job job, User candidate);

    @Query("SELECT ja FROM JobApplication ja WHERE ja.candidate.id = :candidateId AND ja.interviewQuestionsJson IS NOT NULL ORDER BY ja.questionsGeneratedAt DESC")
    List<JobApplication> findApplicationsWithQuestionsByCandidate(@Param("candidateId") Long candidateId);
}
