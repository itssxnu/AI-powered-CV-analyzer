package com.cv.aiml_project.controller;

import com.cv.aiml_project.entity.Resume;
import com.cv.aiml_project.entity.Role;
import com.cv.aiml_project.entity.User;
import com.cv.aiml_project.service.ResumeService;
import com.cv.aiml_project.service.UserService;
import com.cv.aiml_project.service.AIMLIntegrationService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import java.util.List;
import java.util.stream.Collectors;

@Controller
@RequestMapping("/admin")
@PreAuthorize("hasRole('ADMIN')")
public class AdminController {

    @Autowired
    private UserService userService;

    @Autowired
    private ResumeService resumeService;

    @Autowired
    private AIMLIntegrationService aimlIntegrationService;

    // Admin Dashboard
    @GetMapping("/dashboard")
    public String adminDashboard(Model model, HttpServletRequest request, HttpSession session) {
        model.addAttribute("currentPath", request.getRequestURI());
        long totalUsers = userService.getTotalUsers();
        long totalCandidates = userService.getTotalCandidates();
        long totalHR = userService.getTotalHR();
        long activeCandidates = userService.getActiveCandidates();
        long mlProcessed = resumeService.getProcessedResumeCount();  // Use resumeService instead

        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        User user = userService.getUserByUsername(auth.getName()).orElse(null);
        if (user != null) {
            session.setAttribute("user", user);
        }

        model.addAttribute("totalUsers", totalUsers);
        model.addAttribute("totalCandidates", totalCandidates);
        model.addAttribute("totalHR", totalHR);
        model.addAttribute("activeCandidates", activeCandidates);
        model.addAttribute("mlProcessed", mlProcessed);

        return "admin/dashboard";
    }

    // User Management
    @GetMapping("/users")
    public String manageUsers(Model model,
                              @RequestParam(required = false) String role,
                              @RequestParam(required = false) String search) {

        List<User> users;
        if (search != null && !search.trim().isEmpty()) {
            users = userService.searchUsers(search);
            model.addAttribute("search", search);
        } else if (role != null && !role.trim().isEmpty()) {
            try {
                users = userService.getUsersByRole(Role.valueOf(role.toUpperCase()));
                model.addAttribute("selectedRole", role);
            } catch (IllegalArgumentException e) {
                users = userService.getAllUsers();
            }
        } else {
            users = userService.getAllUsers();
        }

        // Force load resumes to avoid LazyInitializationException in template
        users.forEach(user -> {
            if (user.getResumes() != null) {
                user.getResumes().size();  // This loads the resumes
            }
        });

        model.addAttribute("users", users);
        return "admin/users";
    }

    @GetMapping("/users/new")
    public String showUserForm(Model model) {
        model.addAttribute("user", new User());
        return "admin/user-form";
    }

    @PostMapping("/users/new")
    public String createUser(@ModelAttribute User user,
                             @RequestParam String password,
                             @RequestParam String role,
                             RedirectAttributes redirectAttributes) {
        try {
            // Set role
            try {
                user.setRole(Role.valueOf(role.toUpperCase()));
            } catch (IllegalArgumentException e) {
                user.setRole(Role.CANDIDATE);
            }

            // Create user
            userService.createUser(user, password);
            redirectAttributes.addFlashAttribute("message", "User created successfully!");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Error creating user: " + e.getMessage());
        }
        return "redirect:/admin/users";
    }

    @GetMapping("/users/{id}/edit")
    public String showEditUserForm(@PathVariable Long id, Model model) {
        User user = userService.getUserById(id)
                .orElseThrow(() -> new IllegalArgumentException("User not found"));
        // Load resumes
        if (user.getResumes() != null) {
            user.getResumes().size();
        }
        model.addAttribute("user", user);
        return "admin/user-edit";
    }

    @PostMapping("/users/{id}/edit")
    public String updateUser(@PathVariable Long id,
                             @ModelAttribute User user,
                             RedirectAttributes redirectAttributes) {
        try {
            userService.updateUser(id, user);
            redirectAttributes.addFlashAttribute("message", "User updated successfully!");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Error updating user: " + e.getMessage());
        }
        return "redirect:/admin/users";
    }

    @PostMapping("/users/{id}/role")
    public String changeUserRole(@PathVariable Long id,
                                 @RequestParam String role,
                                 RedirectAttributes redirectAttributes) {
        try {
            userService.updateRole(id, Role.valueOf(role.toUpperCase()));
            redirectAttributes.addFlashAttribute("message", "User role updated successfully!");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Error updating role: " + e.getMessage());
        }
        return "redirect:/admin/users";
    }

    @PostMapping("/users/{id}/activate")
    public String activateUser(@PathVariable Long id, RedirectAttributes redirectAttributes) {
        try {
            userService.activateUser(id);
            redirectAttributes.addFlashAttribute("message", "User activated successfully!");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Error activating user: " + e.getMessage());
        }
        return "redirect:/admin/users";
    }

    @PostMapping("/users/{id}/deactivate")
    public String deactivateUser(@PathVariable Long id, RedirectAttributes redirectAttributes) {
        try {
            userService.deactivateUser(id);
            redirectAttributes.addFlashAttribute("message", "User deactivated successfully!");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Error deactivating user: " + e.getMessage());
        }
        return "redirect:/admin/users";
    }

    @PostMapping("/users/{id}/delete")
    public String deleteUser(@PathVariable Long id, RedirectAttributes redirectAttributes) {
        System.out.println("DELETE request received for user ID: " + id); // Add this
        try {
            userService.deleteUser(id);
            System.out.println("User deleted successfully: " + id); // Add this
            redirectAttributes.addFlashAttribute("message", "User deleted successfully!");
        } catch (Exception e) {
            System.out.println("Error deleting user: " + e.getMessage()); // Add this
            e.printStackTrace(); // Add this
            redirectAttributes.addFlashAttribute("error", "Error deleting user: " + e.getMessage());
        }
        return "redirect:/admin/users";
    }

    // AI/ML Management
    @GetMapping("/ai-management")
    public String aiManagement(Model model) {
        List<User> allCandidates = userService.getCandidates();

        // Load resumes for all candidates
        allCandidates.forEach(candidate -> {
            if (candidate.getResumes() != null) {
                candidate.getResumes().size();
            }
        });

        List<User> unprocessedCandidates = allCandidates.stream()
                .filter(u -> u.getCurrentResume() == null || !u.getCurrentResume().isMlProcessed())
                .collect(Collectors.toList());

        List<User> processedCandidates = allCandidates.stream()
                .filter(u -> u.getCurrentResume() != null && u.getCurrentResume().isMlProcessed())
                .collect(Collectors.toList());

        model.addAttribute("unprocessedCandidates", unprocessedCandidates);
        model.addAttribute("processedCandidates", processedCandidates);

        return "admin/ai-management";
    }

    @PostMapping("/ai/process/{userId}")
    public String processCandidateAI(@PathVariable Long userId,
                                     RedirectAttributes redirectAttributes) {
        try {
            User user = userService.getUserById(userId)
                    .orElseThrow(() -> new IllegalArgumentException("User not found"));

            // Check if user has a resume
            Resume currentResume = user.getCurrentResume();
            if (currentResume == null) {
                redirectAttributes.addFlashAttribute("error",
                        "User has no resume to process. Please upload a resume first.");
                return "redirect:/admin/ai-management";
            }

            // Perform actual AI processing via Python FastAPI integration
            aimlIntegrationService.processResume(currentResume.getId());

            // Re-fetch the resume from DB to get the new ML scores and text
            currentResume = resumeService.getCurrentResume(userId).orElse(currentResume);

            redirectAttributes.addFlashAttribute("message",
                    "AI processing completed for " + user.getFullName() +
                            " with score: " + String.format("%.1f", currentResume.getMlScore() != null ? currentResume.getMlScore() : 0.0));
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "AI processing failed: " + e.getMessage());
        }
        return "redirect:/admin/ai-management";
    }

    /**
     * Run the bias/fairness audit via Python AI service and render results on the admin panel
     */
    @GetMapping("/ai/fairness-audit")
    public String runFairnessAudit(Model model, RedirectAttributes redirectAttributes,
                                   HttpServletRequest request) {
        model.addAttribute("currentPath", request.getRequestURI());

        // Reload AI management page data
        List<User> allCandidates = userService.getCandidates();
        List<User> unprocessed = allCandidates.stream()
                .filter(u -> u.getCurrentResume() == null || !u.getCurrentResume().isMlProcessed())
                .collect(Collectors.toList());
        List<User> processed = allCandidates.stream()
                .filter(u -> u.getCurrentResume() != null && u.getCurrentResume().isMlProcessed())
                .collect(Collectors.toList());

        model.addAttribute("unprocessedCandidates", unprocessed);
        model.addAttribute("processedCandidates", processed);

        try {
            java.util.Map<String, Object> auditResult = aimlIntegrationService.auditFairness();
            model.addAttribute("auditResult", auditResult);
            model.addAttribute("auditRan", true);
        } catch (Exception e) {
            model.addAttribute("error", "Bias audit failed: " + e.getMessage() +
                    ". Make sure the Python AI service is running.");
        }

        return "admin/ai-management";
    }
}