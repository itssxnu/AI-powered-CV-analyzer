package com.cv.aiml_project.service;

import com.cv.aiml_project.entity.Job;
import com.cv.aiml_project.entity.User;
import org.springframework.stereotype.Service;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

@Service
public class AIMatchingService {

    /**
     * Calculate overall job match score for a candidate
     * This is a mock implementation - will be replaced with actual AI/ML model
     */
    public Double calculateJobMatchScore(Job job, User candidate) {
        double totalScore = 0.0;
        int weightCount = 0;

        // Skills match (50% weight)
        if (job.getRequiredSkills() != null && candidate.getSkills() != null) {
            double skillsScore = calculateSkillsMatchScore(job.getRequiredSkills(), candidate.getSkills());
            totalScore += skillsScore * 50;
            weightCount += 50;
        }

        // Experience match (30% weight)
        if (job.getExperienceRequired() != null && candidate.getExperienceYears() != null) {
            double experienceScore = calculateExperienceMatchScore(job.getExperienceRequired(), candidate.getExperienceYears());
            totalScore += experienceScore * 30;
            weightCount += 30;
        }

        // Education match (20% weight)
        if (job.getEducationRequirement() != null && candidate.getEducation() != null) {
            double educationScore = calculateEducationMatchScore(job.getEducationRequirement(), candidate.getEducation());
            totalScore += educationScore * 20;
            weightCount += 20;
        }

        if (weightCount == 0) return 0.0;

        // Normalize score
        double normalizedScore = totalScore / weightCount;

        // Add some randomness for demo purposes
        double randomFactor = 0.9 + (Math.random() * 0.2); // 0.9 to 1.1

        return Math.min(100.0, Math.max(0.0, normalizedScore * randomFactor));
    }

    /**
     * Calculate individual component scores
     */
    public Map<String, Double> calculateComponentScores(Job job, User candidate) {
        Map<String, Double> scores = new HashMap<>();

        // Skills score
        if (job.getRequiredSkills() != null && candidate.getSkills() != null) {
            scores.put("skills", calculateSkillsMatchScore(job.getRequiredSkills(), candidate.getSkills()));
        } else {
            scores.put("skills", 50.0); // Default score
        }

        // Experience score
        if (job.getExperienceRequired() != null && candidate.getExperienceYears() != null) {
            scores.put("experience", calculateExperienceMatchScore(job.getExperienceRequired(), candidate.getExperienceYears()));
        } else {
            scores.put("experience", 50.0);
        }

        // Education score
        if (job.getEducationRequirement() != null && candidate.getEducation() != null) {
            scores.put("education", calculateEducationMatchScore(job.getEducationRequirement(), candidate.getEducation()));
        } else {
            scores.put("education", 50.0);
        }

        return scores;
    }

    /**
     * Calculate skills match percentage
     */
    private double calculateSkillsMatchScore(String requiredSkills, String candidateSkills) {
        Set<String> required = Arrays.stream(requiredSkills.split(","))
                .map(String::trim)
                .map(String::toLowerCase)
                .collect(Collectors.toSet());

        Set<String> candidate = Arrays.stream(candidateSkills.split(","))
                .map(String::trim)
                .map(String::toLowerCase)
                .collect(Collectors.toSet());

        if (required.isEmpty()) return 100.0;

        long matched = required.stream()
                .filter(candidate::contains)
                .count();

        return (matched * 100.0) / required.size();
    }

    /**
     * Calculate experience match score
     */
    private double calculateExperienceMatchScore(String requiredExp, int candidateYears) {
        // Parse required experience (e.g., "3-5 years", "2+ years", "5 years")
        int requiredMin = 0;
        int requiredMax = 10;

        try {
            String expStr = requiredExp.toLowerCase().replaceAll("[^0-9\\-\\+]", "");
            if (expStr.contains("-")) {
                String[] parts = expStr.split("-");
                requiredMin = Integer.parseInt(parts[0].trim());
                requiredMax = Integer.parseInt(parts[1].trim());
            } else if (expStr.contains("+")) {
                requiredMin = Integer.parseInt(expStr.replace("+", ""));
                requiredMax = 20;
            } else {
                requiredMin = Integer.parseInt(expStr);
                requiredMax = requiredMin + 2;
            }
        } catch (Exception e) {
            // Default values if parsing fails
            requiredMin = 0;
            requiredMax = 10;
        }

        if (candidateYears < requiredMin) {
            // Less than minimum: (candidateYears / requiredMin) * 70
            return (candidateYears * 70.0) / requiredMin;
        } else if (candidateYears <= requiredMax) {
            // Within range: 100
            return 100.0;
        } else {
            // More than required: 100 - (excess * 10) but not less than 70
            int excess = candidateYears - requiredMax;
            return Math.max(70, 100 - (excess * 10));
        }
    }

    /**
     * Calculate education match score
     */
    private double calculateEducationMatchScore(String requiredEdu, String candidateEdu) {
        String required = requiredEdu.toLowerCase();
        String candidate = candidateEdu.toLowerCase();

        // Simple keyword matching
        if (candidate.contains(required)) {
            return 100.0;
        }

        if (required.contains("bachelor") && candidate.contains("master")) {
            return 90.0;
        }

        if (required.contains("master") && candidate.contains("bachelor")) {
            return 70.0;
        }

        if (required.contains("phd") && candidate.contains("master")) {
            return 80.0;
        }

        return 50.0;
    }
}
