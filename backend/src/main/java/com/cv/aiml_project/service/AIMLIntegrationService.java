package com.cv.aiml_project.service;

import com.cv.aiml_project.entity.Job;
import com.cv.aiml_project.entity.Resume;
import com.cv.aiml_project.repository.ResumeRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

@Service
public class AIMLIntegrationService {

    @Value("${ai.api.url:http://localhost:5000/api}")
    private String aiApiUrl;

    @Value("${ai.api.key:}")
    private String aiApiKey;

    @Autowired
    private ResumeRepository resumeRepository;  // Use repository directly instead of service

    @Autowired
    private com.cv.aiml_project.repository.JobRepository jobRepository;

    @Autowired
    private com.cv.aiml_project.repository.UserRepository userRepository;

    private final RestTemplate restTemplate = new RestTemplate();

    /**
     * Process a resume with the AI/ML API
     */
    public void processResume(Long resumeId) {
        try {
            Resume resume = resumeRepository.findById(resumeId)
                    .orElseThrow(() -> new RuntimeException("Resume not found"));

            Path filePath = Paths.get(resume.getFilePath());
            if (!Files.exists(filePath)) {
                throw new RuntimeException("Resume file does not exist at path: " + filePath);
            }

            // Prepare API request
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            if (aiApiKey != null && !aiApiKey.isEmpty()) {
                headers.set("X-API-Key", aiApiKey);
            }

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("resume", new FileSystemResource(filePath.toFile()));

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            // Call AI API
            String apiUrl = aiApiUrl + "/analyze-resume";
            ResponseEntity<Map> response = restTemplate.exchange(
                    apiUrl,
                    HttpMethod.POST,
                    requestEntity,
                    Map.class
            );

            // Process response
            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                Map<String, Object> result = response.getBody();

                String rawResponse = "{}";
                try {
                    com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
                    rawResponse = mapper.writeValueAsString(result);
                } catch (Exception e) {
                    System.err.println("Failed to serialize json response");
                }

                Double confidence = 0.0;
                String extractedText = "";
                String extractedSkills = "";
                String extractedEducation = null;
                String extractedCertifications = null;
                String extractedLanguages = null;
                String extractedProjects = null;
                Integer experienceYears = null;

                // --- meta ---
                if (result.containsKey("meta") && result.get("meta") instanceof Map) {
                    Map<String, Object> meta = (Map<String, Object>) result.get("meta");
                    if (meta.containsKey("confidence") && meta.get("confidence") != null) {
                        confidence = ((Number) meta.get("confidence")).doubleValue();
                    }
                }

                // --- raw text ---
                if (result.containsKey("raw") && result.get("raw") instanceof Map) {
                    Map<String, Object> raw = (Map<String, Object>) result.get("raw");
                    if (raw.containsKey("text") && raw.get("text") != null) {
                        extractedText = (String) raw.get("text");
                    }
                }

                // --- skills ---
                if (result.containsKey("skills") && result.get("skills") instanceof java.util.List) {
                    java.util.List<String> skillsList = (java.util.List<String>) result.get("skills");
                    if (!skillsList.isEmpty()) {
                        extractedSkills = String.join(", ", skillsList);
                    }
                }

                // --- education ---
                if (result.containsKey("education") && result.get("education") instanceof java.util.List) {
                    java.util.List<Map<String, Object>> eduList = (java.util.List<Map<String, Object>>) result.get("education");
                    java.util.List<String> eduStrings = new java.util.ArrayList<>();
                    for (Map<String, Object> edu : eduList) {
                        String degree = edu.get("degree") != null ? String.valueOf(edu.get("degree")) : "";
                        String institute = edu.get("institute") != null ? String.valueOf(edu.get("institute")) : "";
                        if (!degree.isEmpty() || !institute.isEmpty()) {
                            eduStrings.add(degree + (institute.isEmpty() ? "" : " - " + institute));
                        }
                    }
                    if (!eduStrings.isEmpty()) extractedEducation = String.join("; ", eduStrings);
                }

                // --- certifications ---
                if (result.containsKey("certifications") && result.get("certifications") instanceof java.util.List) {
                    java.util.List<Map<String, Object>> certList = (java.util.List<Map<String, Object>>) result.get("certifications");
                    java.util.List<String> certNames = new java.util.ArrayList<>();
                    for (Map<String, Object> cert : certList) {
                        if (cert.get("name") != null) certNames.add(String.valueOf(cert.get("name")));
                    }
                    if (!certNames.isEmpty()) extractedCertifications = String.join(", ", certNames);
                }

                // --- languages ---
                if (result.containsKey("languages") && result.get("languages") instanceof java.util.List) {
                    java.util.List<Map<String, Object>> langList = (java.util.List<Map<String, Object>>) result.get("languages");
                    java.util.List<String> langStrings = new java.util.ArrayList<>();
                    for (Map<String, Object> lang : langList) {
                        String name = lang.get("name") != null ? String.valueOf(lang.get("name")) : "";
                        String level = lang.get("level") != null ? " (" + lang.get("level") + ")" : "";
                        if (!name.isEmpty()) langStrings.add(name + level);
                    }
                    if (!langStrings.isEmpty()) extractedLanguages = String.join(", ", langStrings);
                }

                // --- experience (for years and projects/roles list) ---
                if (result.containsKey("experience") && result.get("experience") instanceof java.util.List) {
                    java.util.List<Map<String, Object>> expList = (java.util.List<Map<String, Object>>) result.get("experience");
                    java.util.List<String> roleStrings = new java.util.ArrayList<>();
                    for (Map<String, Object> exp : expList) {
                        String role = exp.get("role") != null ? String.valueOf(exp.get("role")) : "";
                        String company = exp.get("company") != null ? String.valueOf(exp.get("company")) : "";
                        if (!role.isEmpty()) roleStrings.add(role + (company.isEmpty() ? "" : " at " + company));
                    }
                    if (!roleStrings.isEmpty()) extractedProjects = String.join("; ", roleStrings);
                }

                // --- profile (career level / years of experience) ---
                if (result.containsKey("profile") && result.get("profile") instanceof Map) {
                    Map<String, Object> profile = (Map<String, Object>) result.get("profile");
                    if (profile.containsKey("years_experience") && profile.get("years_experience") != null) {
                        experienceYears = ((Number) profile.get("years_experience")).intValue();
                    }
                }

                // Update resume and candidate user with all AI results
                updateResumeWithAIResults(resumeId, confidence, extractedText, rawResponse,
                        extractedSkills, extractedEducation, extractedCertifications,
                        extractedLanguages, extractedProjects, experienceYears);
            }

        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("AI processing failed: " + e.getMessage());
        }
    }

    /**
     * Update resume and candidate User with all AI analysis results
     */
    private void updateResumeWithAIResults(Long resumeId, Double confidence,
                                           String extractedText, String rawResponse,
                                           String extractedSkills, String extractedEducation,
                                           String extractedCertifications, String extractedLanguages,
                                           String extractedProjects, Integer experienceYears) {
        Resume resume = resumeRepository.findById(resumeId)
                .orElseThrow(() -> new RuntimeException("Resume not found"));

        resume.setMlProcessed(true);
        resume.setMlConfidence(confidence);
        resume.setMlProcessedDate(LocalDateTime.now());

        if (extractedText != null && !extractedText.isEmpty()) {
            resume.setExtractedText(extractedText);
        }

        if (rawResponse != null) {
            resume.setMlRawResponse(rawResponse);
        }

        // Update the candidate User with all AI-extracted profile data
        com.cv.aiml_project.entity.User user = resume.getUser();
        if (user != null) {
            if (extractedSkills != null && !extractedSkills.isEmpty()) user.setSkills(extractedSkills);
            if (extractedEducation != null && !extractedEducation.isEmpty()) user.setEducation(extractedEducation);
            if (experienceYears != null) user.setExperienceYears(experienceYears);
            userRepository.save(user);
        }

        resumeRepository.save(resume);
    }

    /**
     * Call Python AI to calculate match scores and return details.
     */
    public Map<String, Object> calculateSkillMatch(Job job, Resume resume) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            if (aiApiKey != null && !aiApiKey.isEmpty()) {
                headers.set("X-API-Key", aiApiKey);
            }

            Map<String, Object> jobData = new HashMap<>();
            jobData.put("id", job.getId());
            jobData.put("title", job.getTitle());
            jobData.put("department", job.getDepartment());
            
            // parse experience string like "3-5 years" -> 3
            int minYears = 0;
            if (job.getExperienceRequired() != null) {
                String expStr = job.getExperienceRequired().replaceAll("[^0-9]", "");
                if (expStr.length() > 0) {
                    minYears = Character.getNumericValue(expStr.charAt(0));
                }
            }
            jobData.put("min_years", minYears);

            // Job skills to send — trim each skill to avoid whitespace breaking semantic matching
            String[] reqSkills = job.getRequiredSkills() != null
                    ? Arrays.stream(job.getRequiredSkills().split(","))
                            .map(String::trim).filter(s -> !s.isEmpty()).toArray(String[]::new)
                    : new String[]{};
            String[] prefSkills = job.getPreferredSkills() != null
                    ? Arrays.stream(job.getPreferredSkills().split(","))
                            .map(String::trim).filter(s -> !s.isEmpty()).toArray(String[]::new)
                    : new String[]{};
            jobData.put("required_skills", reqSkills);
            jobData.put("preferred_skills", prefSkills);

            Map<String, Object> requestBody = new HashMap<>();
            // Prefer structured JSON parsed by model if available, fallback to raw extractedText
            if (resume.getMlRawResponse() != null && !resume.getMlRawResponse().isEmpty()) {
                requestBody.put("cv_text", resume.getMlRawResponse());
            } else {
                requestBody.put("cv_text", resume.getExtractedText());
            }

            requestBody.put("job_id", job.getId());
            requestBody.put("job_data", jobData);

            HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(requestBody, headers);
            String apiUrl = aiApiUrl + "/match";

            ResponseEntity<Map> response = restTemplate.exchange(
                    apiUrl,
                    HttpMethod.POST,
                    requestEntity,
                    Map.class
            );

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                System.out.println("========== AI MATCH RESPONSE START ==========");
                System.out.println(response.getBody());
                System.out.println("========== AI MATCH RESPONSE END ==========");
                return response.getBody();
            } else {
                System.err.println("AI MATCH FAILED: Status " + response.getStatusCode());
                throw new RuntimeException("Failed to get match score from AI service");
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("AI matching failed: " + e.getMessage());
        }
    }

    /**
     * Process all unprocessed resumes
     */
    public void processAllUnprocessedResumes() {
        // This method would need to be called from a scheduler with access to the files
        // You might want to implement this differently
    }

    /**
     * Run fairness audit on the AI ranker
     */
    public Map<String, Object> auditFairness() {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            if (aiApiKey != null && !aiApiKey.isEmpty()) {
                headers.set("X-API-Key", aiApiKey);
            }

            HttpEntity<String> requestEntity = new HttpEntity<>("{}", headers);

            String apiUrl = aiApiUrl + "/fairness-audit";
            ResponseEntity<Map> response = restTemplate.exchange(
                    apiUrl,
                    HttpMethod.POST,
                    requestEntity,
                    Map.class
            );

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                return response.getBody();
            } else {
                throw new RuntimeException("Failed to get fairness audit from AI service");
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Fairness audit failed: " + e.getMessage());
        }
    }

    /**
     * Generate interview questions for a candidate and job
     */
    public Map<String, Object> generateInterviewQuestions(Long cvId, Long jobId) {
        try {
            com.cv.aiml_project.entity.Job job = jobRepository.findById(jobId)
                    .orElseThrow(() -> new RuntimeException("Job not found"));
            Resume resume = resumeRepository.findById(cvId)
                    .orElseThrow(() -> new RuntimeException("Resume not found"));

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            if (aiApiKey != null && !aiApiKey.isEmpty()) {
                headers.set("X-API-Key", aiApiKey);
            }

            Map<String, Object> jobData = new HashMap<>();
            jobData.put("title", job.getTitle());
            jobData.put("department", job.getDepartment());

            Map<String, Object> requestBody = new HashMap<>();
            
            if (resume.getMlRawResponse() != null && !resume.getMlRawResponse().isEmpty()) {
                requestBody.put("cv_text", resume.getMlRawResponse());
            } else {
                requestBody.put("cv_text", resume.getExtractedText());
            }

            requestBody.put("job_data", jobData);
            requestBody.put("match_details", new HashMap<>());

            HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(requestBody, headers);

            String apiUrl = aiApiUrl + "/generate-questions";
            ResponseEntity<Map> response = restTemplate.exchange(
                    apiUrl,
                    HttpMethod.POST,
                    requestEntity,
                    Map.class
            );

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                return response.getBody();
            } else {
                throw new RuntimeException("Failed to generate interview questions from AI service");
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Interview question generation failed: " + e.getMessage());
        }
    }
}