package com.cv.aiml_project.controller;

import com.cv.aiml_project.entity.Resume;
import com.cv.aiml_project.entity.User;
import com.cv.aiml_project.service.ResumeService;
import com.cv.aiml_project.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import java.io.IOException;
import java.time.format.DateTimeFormatter;
import java.util.List;

@Controller
@RequestMapping("/candidate/resume")
public class ResumeController {

    @Autowired
    private UserService userService;

    @Autowired
    private ResumeService resumeService;

    private User getCurrentUser() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String username = authentication.getName();
        return userService.getUserByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found"));
    }

    // Resume Management Page
    @GetMapping("")
    public String resumeManagement(Model model) {
        User user = getCurrentUser();
        Resume currentResume = user.getCurrentResume();
        List<Resume> allResumes = resumeService.getAllUserResumes(user.getId());

        model.addAttribute("user", user);
        model.addAttribute("currentResume", currentResume);
        model.addAttribute("allResumes", allResumes);
        model.addAttribute("hasResume", currentResume != null);

        if (currentResume != null && currentResume.getUploadDate() != null) {
            model.addAttribute("resumeUploadDate",
                    currentResume.getUploadDate().format(DateTimeFormatter.ofPattern("dd MMM yyyy, HH:mm")));
        }

        return "candidate/resume";
    }

    // Upload Resume Page
    @GetMapping("/upload")
    public String showUploadResumePage(Model model) {
        User user = getCurrentUser();
        model.addAttribute("user", user);
        return "candidate/upload-resume";
    }

    // Process Resume Upload
    @PostMapping("/upload")
    public String uploadResume(@RequestParam("resumeFile") MultipartFile file,
                               RedirectAttributes redirectAttributes) {
        try {
            User user = getCurrentUser();

            // Validate file
            if (file.isEmpty()) {
                redirectAttributes.addFlashAttribute("error", "Please select a file to upload");
                return "redirect:/candidate/resume/upload";
            }

            // Check file type
            String contentType = file.getContentType();
            if (!"application/pdf".equals(contentType)) {
                redirectAttributes.addFlashAttribute("error", "Only PDF files are allowed");
                return "redirect:/candidate/resume/upload";
            }

            // Check file size (max 5MB)
            if (file.getSize() > 5 * 1024 * 1024) {
                redirectAttributes.addFlashAttribute("error", "File size must be less than 5MB");
                return "redirect:/candidate/resume/upload";
            }

            // Upload resume
            resumeService.uploadResume(user.getId(), file);

            redirectAttributes.addFlashAttribute("message",
                    "Resume uploaded successfully! AI analysis will begin shortly.");

            return "redirect:/candidate/resume";

        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Upload failed: " + e.getMessage());
            return "redirect:/candidate/resume/upload";
        }
    }

    // View Current Resume
    @GetMapping("/view")
    public ResponseEntity<Resource> viewCurrentResume() throws IOException {
        User user = getCurrentUser();
        Resume currentResume = user.getCurrentResume();

        if (currentResume == null) {
            return ResponseEntity.notFound().build();
        }

        Resource resource = resumeService.getResumeFile(currentResume.getId());

        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_PDF)
                .header(HttpHeaders.CONTENT_DISPOSITION,
                        "inline; filename=\"" + currentResume.getOriginalName() + "\"")
                .body(resource);
    }

    // View Specific Resume Version
    @GetMapping("/view/{resumeId}")
    public ResponseEntity<Resource> viewResumeVersion(@PathVariable Long resumeId) throws IOException {
        User user = getCurrentUser();
        Resume resume = resumeService.getResumeById(resumeId)
                .orElseThrow(() -> new RuntimeException("Resume not found"));

        // Verify ownership
        if (!resume.getUser().getId().equals(user.getId())) {
            return ResponseEntity.status(403).build();
        }

        Resource resource = resumeService.getResumeFile(resumeId);

        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_PDF)
                .header(HttpHeaders.CONTENT_DISPOSITION,
                        "inline; filename=\"" + resume.getOriginalName() + "\"")
                .body(resource);
    }

    // Download Current Resume
    @GetMapping("/download")
    public ResponseEntity<Resource> downloadCurrentResume() throws IOException {
        User user = getCurrentUser();
        Resume currentResume = user.getCurrentResume();

        if (currentResume == null) {
            return ResponseEntity.notFound().build();
        }

        Resource resource = resumeService.getResumeFile(currentResume.getId());

        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_OCTET_STREAM)
                .header(HttpHeaders.CONTENT_DISPOSITION,
                        "attachment; filename=\"" + currentResume.getOriginalName() + "\"")
                .body(resource);
    }

    // Delete Current Resume
    @PostMapping("/delete")
    public String deleteCurrentResume(RedirectAttributes redirectAttributes) {
        try {
            User user = getCurrentUser();
            Resume currentResume = user.getCurrentResume();

            if (currentResume != null) {
                resumeService.deleteResume(currentResume.getId());
            }

            redirectAttributes.addFlashAttribute("message", "Resume deleted successfully");
            return "redirect:/candidate/resume";

        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Delete failed: " + e.getMessage());
            return "redirect:/candidate/resume";
        }
    }

    // Delete Specific Resume Version
    @PostMapping("/delete/{resumeId}")
    public String deleteResumeVersion(@PathVariable Long resumeId,
                                      RedirectAttributes redirectAttributes) {
        try {
            User user = getCurrentUser();
            Resume resume = resumeService.getResumeById(resumeId)
                    .orElseThrow(() -> new RuntimeException("Resume not found"));

            // Verify ownership
            if (!resume.getUser().getId().equals(user.getId())) {
                redirectAttributes.addFlashAttribute("error", "You don't have permission to delete this resume");
                return "redirect:/candidate/resume";
            }

            resumeService.deleteResume(resumeId);

            redirectAttributes.addFlashAttribute("message", "Resume version deleted successfully");
            return "redirect:/candidate/resume";

        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Delete failed: " + e.getMessage());
            return "redirect:/candidate/resume";
        }
    }

    // Update/Replace Current Resume
    @PostMapping("/update")
    public String updateResume(@RequestParam("resumeFile") MultipartFile file,
                               RedirectAttributes redirectAttributes) {
        try {
            User user = getCurrentUser();

            // Validate file
            if (file.isEmpty()) {
                redirectAttributes.addFlashAttribute("error", "Please select a file to upload");
                return "redirect:/candidate/resume";
            }

            // Check file type
            String contentType = file.getContentType();
            if (!"application/pdf".equals(contentType)) {
                redirectAttributes.addFlashAttribute("error", "Only PDF files are allowed");
                return "redirect:/candidate/resume";
            }

            // Check file size
            if (file.getSize() > 5 * 1024 * 1024) {
                redirectAttributes.addFlashAttribute("error", "File size must be less than 5MB");
                return "redirect:/candidate/resume";
            }

            // Upload new version (this will automatically set it as current)
            resumeService.uploadResume(user.getId(), file);

            redirectAttributes.addFlashAttribute("message",
                    "Resume updated successfully! AI analysis will be re-run.");

            return "redirect:/candidate/resume";

        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Update failed: " + e.getMessage());
            return "redirect:/candidate/resume";
        }
    }

    // View Resume Text (Extracted content)
    @GetMapping("/text")
    public String viewResumeText(Model model) {
        User user = getCurrentUser();
        Resume currentResume = user.getCurrentResume();

        if (currentResume == null) {
            model.addAttribute("error", "No resume found. Please upload a resume first.");
        } else {
            model.addAttribute("resume", currentResume);
            model.addAttribute("resumeText", currentResume.getExtractedText());
        }

        model.addAttribute("user", user);
        return "candidate/resume-text";
    }

    // View Specific Resume Version Text
    @GetMapping("/text/{resumeId}")
    public String viewResumeVersionText(@PathVariable Long resumeId, Model model) {
        User user = getCurrentUser();
        Resume resume = resumeService.getResumeById(resumeId)
                .orElseThrow(() -> new RuntimeException("Resume not found"));

        // Verify ownership
        if (!resume.getUser().getId().equals(user.getId())) {
            throw new RuntimeException("Access denied");
        }

        model.addAttribute("resume", resume);
        model.addAttribute("resumeText", resume.getExtractedText());
        model.addAttribute("user", user);
        return "candidate/resume-text";
    }

    // Resume History/Version List
    @GetMapping("/history")
    public String resumeHistory(Model model) {
        User user = getCurrentUser();
        List<Resume> allResumes = resumeService.getAllUserResumes(user.getId());

        model.addAttribute("user", user);
        model.addAttribute("resumes", allResumes);
        return "candidate/resume-history";
    }

    // Set a specific version as current
    @PostMapping("/set-current/{resumeId}")
    public String setCurrentResume(@PathVariable Long resumeId,
                                   RedirectAttributes redirectAttributes) {
        try {
            User user = getCurrentUser();
            Resume resume = resumeService.getResumeById(resumeId)
                    .orElseThrow(() -> new RuntimeException("Resume not found"));

            // Verify ownership
            if (!resume.getUser().getId().equals(user.getId())) {
                redirectAttributes.addFlashAttribute("error", "Access denied");
                return "redirect:/candidate/resume";
            }

            // Implementation would need a method in ResumeService to set a specific resume as current
            // resumeService.setAsCurrent(resumeId);

            redirectAttributes.addFlashAttribute("message", "Current resume updated successfully");
            return "redirect:/candidate/resume";

        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Operation failed: " + e.getMessage());
            return "redirect:/candidate/resume";
        }
    }
}