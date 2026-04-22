package com.cv.aiml_project.controller;

import com.cv.aiml_project.entity.Resume;
import com.cv.aiml_project.entity.User;
import com.cv.aiml_project.service.UserService;
import jakarta.servlet.http.HttpSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;
import org.springframework.web.multipart.MultipartFile;
import java.io.IOException;

@Controller
@RequestMapping("/candidate")
public class UserController {

    @Autowired
    private UserService userService;

    private User getCurrentUser() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String username = authentication.getName();
        return userService.getUserByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found"));
    }

    @GetMapping("/dashboard")
    public String candidateDashboard(Model model, HttpSession session) {
        User user = getCurrentUser();
        Resume currentResume = user.getCurrentResume();

        model.addAttribute("user", user);
        model.addAttribute("currentResume", currentResume);
        model.addAttribute("mlProcessed", currentResume != null ? currentResume.isMlProcessed() : false);

        return "candidate/dashboard";
    }

    // Profile Management
    @GetMapping("/profile")
    public String showProfile(Model model) {
        User user = getCurrentUser();
        model.addAttribute("user", user);
        return "candidate/profile";
    }

    @PostMapping("/profile/update")
    public String updateProfile(@RequestParam String firstName,
                                @RequestParam String lastName,
                                @RequestParam String email,
                                @RequestParam String phone,
                                @RequestParam(required = false) String skills,
                                @RequestParam(required = false) Integer experienceYears,
                                @RequestParam(required = false) String education,
                                RedirectAttributes redirectAttributes) {

        User currentUser = getCurrentUser();
        Long userId = currentUser.getId();

        // Check if email is already taken by another user
        if (!currentUser.getEmail().equals(email) && userService.emailExists(email)) {
            redirectAttributes.addFlashAttribute("error", "Email already registered with another account");
            return "redirect:/candidate/profile";
        }

        // Update user
        User updatedUser = new User();
        updatedUser.setFirstName(firstName);
        updatedUser.setLastName(lastName);
        updatedUser.setEmail(email);
        updatedUser.setPhone(phone);
        updatedUser.setSkills(skills);
        updatedUser.setExperienceYears(experienceYears);
        updatedUser.setEducation(education);

        userService.updateUser(userId, updatedUser);

        redirectAttributes.addFlashAttribute("message", "Profile updated successfully");
        return "redirect:/candidate/profile";
    }

    // Change Password
    @PostMapping("/change-password")
    public String changePassword(@RequestParam String currentPassword,
                                 @RequestParam String newPassword,
                                 @RequestParam String confirmPassword,
                                 RedirectAttributes redirectAttributes) {

        User user = getCurrentUser();

        // Verify current password
        if (!userService.validatePassword(currentPassword, user.getPassword())) {
            redirectAttributes.addFlashAttribute("error", "Current password is incorrect");
            return "redirect:/candidate/profile#password";
        }

        // Check if new passwords match
        if (!newPassword.equals(confirmPassword)) {
            redirectAttributes.addFlashAttribute("error", "New passwords do not match");
            return "redirect:/candidate/profile#password";
        }

        // Update password
        userService.updatePassword(user.getId(), newPassword);

        redirectAttributes.addFlashAttribute("message", "Password updated successfully");
        return "redirect:/candidate/profile#password";
    }

    // Delete Account
    @GetMapping("/delete-account")
    public String showDeleteAccountPage(Model model) {
        User user = getCurrentUser();
        model.addAttribute("user", user);
        return "candidate/delete-account";
    }

    @PostMapping("/delete-account")
    public String deleteAccount(@RequestParam String password,
                                RedirectAttributes redirectAttributes) {

        User user = getCurrentUser();

        // Verify password
        if (!userService.validatePassword(password, user.getPassword())) {
            redirectAttributes.addFlashAttribute("error", "Incorrect password. Account deletion failed.");
            return "redirect:/candidate/delete-account";
        }

        // Deactivate account instead of deleting
        userService.deactivateUser(user.getId());

        // Logout the user
        SecurityContextHolder.clearContext();

        redirectAttributes.addFlashAttribute("message", "Your account has been deactivated.");
        return "redirect:/auth/login";
    }





    @GetMapping("/ai-analysis")
    public String viewAIAnalysis(Model model) {
        User user = getCurrentUser();
        Resume currentResume = user.getCurrentResume();

        if (currentResume == null) {
            model.addAttribute("message", "No resume found. Please upload your resume first.");
        } else if (!currentResume.isMlProcessed()) {
            model.addAttribute("message", "Your resume has not been processed by AI yet.");
        }

        model.addAttribute("user", user);
        model.addAttribute("currentResume", currentResume);
        return "candidate/ai-analysis";
    }
}
