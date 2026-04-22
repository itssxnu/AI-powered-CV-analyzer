package com.cv.aiml_project.config;

import com.cv.aiml_project.entity.Role;
import com.cv.aiml_project.entity.User;
import com.cv.aiml_project.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

@Component
public class DataLoader implements CommandLineRunner {

    @Autowired
    private UserService userService;

    @Autowired
    private PasswordEncoder passwordEncoder;

    @Value("${file.upload.resume-dir:./uploads/resumes}")
    private String resumeUploadDir;

    @Value("${file.upload.application-dir:./uploads/applications}")
    private String applicationUploadDir;

    @Override
    public void run(String... args) throws Exception {
        // Create upload directories
        createUploadDirectories();

        // Create users (without resume data)
        createUsers();
    }

    private void createUploadDirectories() throws Exception {
        Path resumePath = Paths.get(resumeUploadDir);
        if (!Files.exists(resumePath)) {
            Files.createDirectories(resumePath);
            System.out.println("Created resume upload directory: " + resumePath.toAbsolutePath());
        }

        Path applicationPath = Paths.get(applicationUploadDir);
        if (!Files.exists(applicationPath)) {
            Files.createDirectories(applicationPath);
            System.out.println("Created application upload directory: " + applicationPath.toAbsolutePath());
        }
    }

    private void createUsers() {
        // Create admin user
        if (userService.getUserByUsername("admin").isEmpty()) {
            User admin = new User();
            admin.setUsername("admin");
            admin.setEmail("admin@recruitment.com");
            admin.setPassword(passwordEncoder.encode("admin123"));
            admin.setFirstName("System");
            admin.setLastName("Administrator");
            admin.setPhone("9999999999");
            admin.setRole(Role.ADMIN);
            admin.setActive(true);

            userService.createUser(admin, "admin123");
            System.out.println("Admin user created: admin/admin123");
        }

        // Create HR user
        if (userService.getUserByUsername("hr1").isEmpty()) {
            User hr = new User();
            hr.setUsername("hr1");
            hr.setEmail("hr@recruitment.com");
            hr.setPassword(passwordEncoder.encode("hr123"));
            hr.setFirstName("HR");
            hr.setLastName("Manager");
            hr.setPhone("8888888888");
            hr.setRole(Role.HR);
            hr.setActive(true);

            userService.createUser(hr, "hr123");
            System.out.println("HR user created: hr1/hr123");
        }

        // Create candidate - John Doe
        if (userService.getUserByUsername("john_doe").isEmpty()) {
            User candidate = new User();
            candidate.setUsername("john_doe");
            candidate.setEmail("john@example.com");
            candidate.setPassword(passwordEncoder.encode("candidate123"));
            candidate.setFirstName("John");
            candidate.setLastName("Doe");
            candidate.setPhone("7777777777");
            candidate.setRole(Role.CANDIDATE);
            candidate.setSkills("Java, Spring Boot, SQL, Machine Learning");
            candidate.setExperienceYears(3);
            candidate.setEducation("B.Tech Computer Science");
            candidate.setActive(true);

            userService.createUser(candidate, "candidate123");
            System.out.println("Candidate user created: john_doe/candidate123");
        }

        // Create candidate - Jane Smith
        if (userService.getUserByUsername("jane_smith").isEmpty()) {
            User candidate = new User();
            candidate.setUsername("jane_smith");
            candidate.setEmail("jane@example.com");
            candidate.setPassword(passwordEncoder.encode("candidate123"));
            candidate.setFirstName("Jane");
            candidate.setLastName("Smith");
            candidate.setPhone("6666666666");
            candidate.setRole(Role.CANDIDATE);
            candidate.setSkills("Python, Django, React, Data Science");
            candidate.setExperienceYears(2);
            candidate.setEducation("M.Sc Data Science");
            candidate.setActive(true);

            userService.createUser(candidate, "candidate123");
            System.out.println("Candidate user created: jane_smith/candidate123");
        }

        // Create candidate - Bob Johnson
        if (userService.getUserByUsername("bob_johnson").isEmpty()) {
            User candidate = new User();
            candidate.setUsername("bob_johnson");
            candidate.setEmail("bob@example.com");
            candidate.setPassword(passwordEncoder.encode("candidate123"));
            candidate.setFirstName("Bob");
            candidate.setLastName("Johnson");
            candidate.setPhone("5555555555");
            candidate.setRole(Role.CANDIDATE);
            candidate.setSkills("JavaScript, Node.js, AWS, Docker");
            candidate.setExperienceYears(4);
            candidate.setEducation("B.E Computer Engineering");
            candidate.setActive(true);

            userService.createUser(candidate, "candidate123");
            System.out.println("Candidate user created: bob_johnson/candidate123");
        }
    }
}