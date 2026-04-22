package com.cv.aiml_project.controller;

import com.cv.aiml_project.entity.Role;
import com.cv.aiml_project.entity.User;
import com.cv.aiml_project.service.UserService;
import jakarta.servlet.http.HttpSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

@Controller
@RequestMapping("/auth")
public class AuthController {

    @Autowired
    private UserService userService;

    // Show registration form
    @GetMapping("/register")
    public String showRegistrationForm(Model model) {
        model.addAttribute("user", new User());
        return "auth/register";
    }

    // Process registration
    @PostMapping("/register")
    public String registerUser(@ModelAttribute User user,
                               @RequestParam String confirmPassword,
                               @RequestParam(defaultValue = "CANDIDATE") String role,
                               @RequestParam String phoneNumber,
                               RedirectAttributes redirectAttributes) {

        // Check if passwords match
        if (!user.getPassword().equals(confirmPassword)) {
            redirectAttributes.addFlashAttribute("error", "Passwords do not match");
            return "redirect:/auth/register";
        }

        // Check if username already exists
        if (userService.usernameExists(user.getUsername())) {
            redirectAttributes.addFlashAttribute("error", "Username already taken");
            return "redirect:/auth/register";
        }

        // Check if email already exists
        if (userService.emailExists(user.getEmail())) {
            redirectAttributes.addFlashAttribute("error", "Email already registered");
            return "redirect:/auth/register";
        }

        user.setPhone(phoneNumber);

        // Set role (default to CANDIDATE)
        try {
            user.setRole(Role.valueOf(role.toUpperCase()));
        } catch (IllegalArgumentException e) {
            user.setRole(Role.CANDIDATE);
        }

        // Register the user
        userService.registerUser(user);
        redirectAttributes.addFlashAttribute("message", "Registration successful! Please login.");
        return "redirect:/auth/login";
    }

    // Show login form
    @GetMapping("/login")
    public String showLoginForm(@RequestParam(value = "error", required = false) String error,
                                @RequestParam(value = "logout", required = false) String logout,
                                Model model) {

        if (error != null) {
            model.addAttribute("error", "Invalid username or password");
        }

        if (logout != null) {
            model.addAttribute("message", "You have been logged out successfully");
        }

        return "auth/login";
    }

    // Access denied page
    @GetMapping("/access-denied")
    public String accessDenied() {
        return "auth/access-denied";
    }
}
