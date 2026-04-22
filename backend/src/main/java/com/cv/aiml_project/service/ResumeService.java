package com.cv.aiml_project.service;

import com.cv.aiml_project.entity.Resume;
import com.cv.aiml_project.entity.User;
import com.cv.aiml_project.repository.ResumeRepository;
import com.cv.aiml_project.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.net.MalformedURLException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

@Service
@Transactional
public class ResumeService {

    @Value("${file.upload.resume-dir:./uploads/resumes}")
    private String uploadDir;

    @Autowired
    private ResumeRepository resumeRepository;

    @Autowired
    private UserRepository userRepository;

    @PersistenceContext
    private EntityManager entityManager;

    /**
     * Upload a new resume for a user
     */
    public Resume uploadResume(Long userId, MultipartFile file) throws IOException {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found"));

        // Create upload directory if not exists
        Path uploadPath = Paths.get(uploadDir, user.getUsername());
        if (!Files.exists(uploadPath)) {
            Files.createDirectories(uploadPath);
        }

        // Generate unique filename
        String originalFilename = file.getOriginalFilename();
        String fileExtension = originalFilename.substring(originalFilename.lastIndexOf("."));
        String uniqueFileName = UUID.randomUUID().toString() + fileExtension;

        // Save file
        Path filePath = uploadPath.resolve(uniqueFileName);
        Files.copy(file.getInputStream(), filePath, StandardCopyOption.REPLACE_EXISTING);

        // Create new resume entity
        Resume resume = new Resume();
        resume.setUser(user);
        resume.setFileName(uniqueFileName);
        resume.setOriginalName(originalFilename);
        resume.setContentType(file.getContentType());
        resume.setFileSize(file.getSize());
        resume.setFilePath(filePath.toString());
        resume.setUploadDate(LocalDateTime.now());
        resume.setCurrent(true); // This becomes the current resume
        resume.setVersion(getNextVersionNumber(userId));

        // Set all other resumes as not current
        resumeRepository.setAllResumesNotCurrent(userId);

        // Save the new resume
        Resume savedResume = resumeRepository.save(resume);


        return savedResume;
    }

    /**
     * Get the next version number for a user's resume
     */
    private Integer getNextVersionNumber(Long userId) {
        List<Resume> userResumes = resumeRepository.findAllByUserIdOrderByUploadDateDesc(userId);
        if (userResumes.isEmpty()) {
            return 1;
        }
        return userResumes.get(0).getVersion() + 1;
    }



    /**
     * Get the current resume for a user
     */
    public Optional<Resume> getCurrentResume(Long userId) {
        return resumeRepository.findCurrentResumeByUserId(userId);
    }

    /**
     * Get a specific resume by ID
     */
    public Optional<Resume> getResumeById(Long resumeId) {
        return resumeRepository.findById(resumeId);
    }

    /**
     * Get all resumes for a user
     */
    public List<Resume> getAllUserResumes(Long userId) {
        return resumeRepository.findAllByUserIdOrderByUploadDateDesc(userId);
    }

    /**
     * Get resume file as Resource
     */
    public Resource getResumeFile(Long resumeId) throws MalformedURLException {
        Resume resume = resumeRepository.findById(resumeId)
                .orElseThrow(() -> new RuntimeException("Resume not found"));

        Path filePath = Paths.get(resume.getFilePath());
        Resource resource = new UrlResource(filePath.toUri());

        if (resource.exists() && resource.isReadable()) {
            return resource;
        } else {
            throw new RuntimeException("Could not read the file!");
        }
    }

    /**
     * Delete a resume
     */
    public void deleteResume(Long resumeId) throws IOException {
        Resume resume = resumeRepository.findById(resumeId)
                .orElseThrow(() -> new RuntimeException("Resume not found"));

        Long userId = resume.getUser().getId();

        // Delete file from storage
        Path filePath = Paths.get(resume.getFilePath());
        Files.deleteIfExists(filePath);

        // Sever foreign key links in related tables before deletion
        entityManager.createQuery("UPDATE JobApplication ja SET ja.resume = null WHERE ja.resume = :resume")
                .setParameter("resume", resume)
                .executeUpdate();

        entityManager.createQuery("UPDATE SkillMatchResult smr SET smr.resume = null WHERE smr.resume = :resume")
                .setParameter("resume", resume)
                .executeUpdate();

        // Remove from user collection to prevent CascadeType.ALL re-saving issues
        User user = resume.getUser();
        if (user != null && user.getResumes() != null) {
            user.getResumes().remove(resume);
        }

        // Delete from database and flush immediately
        resumeRepository.delete(resume);
        resumeRepository.flush();

        // If this was the current resume, set another resume as current
        if (resume.isCurrent()) {
            List<Resume> remainingResumes = resumeRepository.findAllByUserIdOrderByUploadDateDesc(userId);
            // Ensure we don't pick the one we just deleted (in case flush didn't completely evict from cache)
            remainingResumes.removeIf(r -> r.getId().equals(resumeId));
            
            if (!remainingResumes.isEmpty()) {
                Resume newCurrent = remainingResumes.get(0);
                newCurrent.setCurrent(true);
                resumeRepository.save(newCurrent);
            }
        }
    }

    /**
     * Update AI processing results for a resume
     */
    public Resume updateAIResults(Long resumeId, Double score, Double confidence,
                                  String extractedText, String rawResponse) {
        Resume resume = resumeRepository.findById(resumeId)
                .orElseThrow(() -> new RuntimeException("Resume not found"));

        resume.setMlProcessed(true);
        resume.setMlScore(score);
        resume.setMlConfidence(confidence);
        resume.setMlProcessedDate(LocalDateTime.now());

        if (extractedText != null) {
            resume.setExtractedText(extractedText);
        }

        if (rawResponse != null) {
            resume.setMlRawResponse(rawResponse);
        }

        return resumeRepository.save(resume);
    }

    /**
     * Check if user has a resume
     */
    public boolean hasResume(Long userId) {
        return resumeRepository.existsByUserAndIsCurrentTrue(
                userRepository.getReferenceById(userId));
    }

    /**
     * Get count of processed resumes
     */
    public long getProcessedResumeCount() {
        return resumeRepository.countProcessedResumes();
    }

    /**
     * Get all unprocessed resumes
     */
    public List<Resume> getUnprocessedResumes() {
        return resumeRepository.findByMlProcessed(false);
    }

    public long getTotalResumeCount() {
        return 0;
    }
}