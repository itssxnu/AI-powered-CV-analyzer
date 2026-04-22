package com.cv.aiml_project.repository;

import com.cv.aiml_project.entity.Job;
import com.cv.aiml_project.entity.JobType;
import com.cv.aiml_project.entity.User;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface JobRepository extends JpaRepository<Job, Long> {

    List<Job> findByPostedBy(User postedBy);

    List<Job> findByIsActive(boolean isActive);

    List<Job> findByDepartment(String department);

    List<Job> findByJobType(JobType jobType);

    List<Job> findByExpiryDateBeforeAndIsActive(LocalDateTime date, boolean isActive);

    @Query("SELECT j FROM Job j WHERE " +
            "LOWER(j.title) LIKE LOWER(CONCAT('%', :keyword, '%')) OR " +
            "LOWER(j.description) LIKE LOWER(CONCAT('%', :keyword, '%')) OR " +
            "LOWER(j.department) LIKE LOWER(CONCAT('%', :keyword, '%')) OR " +
            "LOWER(j.location) LIKE LOWER(CONCAT('%', :keyword, '%')) OR " +
            "LOWER(j.requiredSkills) LIKE LOWER(CONCAT('%', :keyword, '%'))")
    List<Job> searchJobs(@Param("keyword") String keyword);

    @Query("SELECT j FROM Job j WHERE j.isActive = true AND " +
            "(j.expiryDate IS NULL OR j.expiryDate > :currentDate)")
    List<Job> findActiveJobs(@Param("currentDate") LocalDateTime currentDate);

    @Query("SELECT j FROM Job j WHERE j.isActive = true AND j.department = :department " +
            "AND (j.expiryDate IS NULL OR j.expiryDate > :currentDate)")
    List<Job> findActiveJobsByDepartment(@Param("department") String department,
                                         @Param("currentDate") LocalDateTime currentDate);

    @Query("SELECT COUNT(j) FROM Job j WHERE j.isActive = true " +
            "AND (j.expiryDate IS NULL OR j.expiryDate > :currentDate)")
    long countActiveJobs(@Param("currentDate") LocalDateTime currentDate);

    @Query("SELECT j.department, COUNT(j) FROM Job j " +
            "WHERE j.isActive = true GROUP BY j.department")
    List<Object[]> countJobsByDepartment();

    @Query("SELECT j FROM Job j ORDER BY j.postedDate DESC")
    Page<Job> findRecentJobs(Pageable pageable);
}
