package com.cv.aiml_project.repository;

import com.cv.aiml_project.entity.Resume;
import com.cv.aiml_project.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface ResumeRepository extends JpaRepository<Resume, Long> {

    List<Resume> findByUser(User user);

    List<Resume> findByUserOrderByUploadDateDesc(User user);

    Optional<Resume> findByUserAndIsCurrentTrue(User user);

    @Query("SELECT r FROM Resume r WHERE r.user.id = :userId AND r.isCurrent = true")
    Optional<Resume> findCurrentResumeByUserId(@Param("userId") Long userId);

    @Query("SELECT r FROM Resume r WHERE r.user.id = :userId ORDER BY r.uploadDate DESC")
    List<Resume> findAllByUserIdOrderByUploadDateDesc(@Param("userId") Long userId);

    @Query("SELECT r FROM Resume r WHERE r.mlProcessed = :processed")
    List<Resume> findByMlProcessed(@Param("processed") boolean processed);

    @Query("SELECT COUNT(r) FROM Resume r WHERE r.mlProcessed = true")
    long countProcessedResumes();

    @Modifying
    @Query("UPDATE Resume r SET r.isCurrent = false WHERE r.user.id = :userId AND r.id != :currentResumeId")
    void setOtherResumesAsNotCurrent(@Param("userId") Long userId, @Param("currentResumeId") Long currentResumeId);

    @Modifying
    @Query("UPDATE Resume r SET r.isCurrent = false WHERE r.user.id = :userId")
    void setAllResumesNotCurrent(@Param("userId") Long userId);

    boolean existsByUserAndIsCurrentTrue(User user);
}
