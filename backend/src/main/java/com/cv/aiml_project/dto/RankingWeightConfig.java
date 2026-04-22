package com.cv.aiml_project.dto;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;

public class RankingWeightConfig {

    @NotNull
    @Min(0)
    @Max(100)
    private Double skillsWeight;

    @NotNull
    @Min(0)
    @Max(100)
    private Double experienceWeight;

    @NotNull
    @Min(0)
    @Max(100)
    private Double educationWeight;

    @Min(0)
    @Max(100)
    private Double personalityWeight;

    @Min(0)
    @Max(100)
    private Double culturalFitWeight;

    private String configName;
    private String description;

    // Constructor with default values
    public RankingWeightConfig() {
        this.skillsWeight = 50.0;
        this.experienceWeight = 30.0;
        this.educationWeight = 20.0;
        this.personalityWeight = 0.0;
        this.culturalFitWeight = 0.0;
    }

    // Validate that total weight is 100
    public boolean isValid() {
        double total = skillsWeight + experienceWeight + educationWeight +
                (personalityWeight != null ? personalityWeight : 0) +
                (culturalFitWeight != null ? culturalFitWeight : 0);
        return Math.abs(total - 100.0) < 0.01; // Allow small floating point error
    }

    public String getValidationMessage() {
        double total = skillsWeight + experienceWeight + educationWeight +
                (personalityWeight != null ? personalityWeight : 0) +
                (culturalFitWeight != null ? culturalFitWeight : 0);
        return "Total weight must be 100%. Current total: " + total + "%";
    }

    // Getters and Setters
    public Double getSkillsWeight() { return skillsWeight; }
    public void setSkillsWeight(Double skillsWeight) { this.skillsWeight = skillsWeight; }

    public Double getExperienceWeight() { return experienceWeight; }
    public void setExperienceWeight(Double experienceWeight) { this.experienceWeight = experienceWeight; }

    public Double getEducationWeight() { return educationWeight; }
    public void setEducationWeight(Double educationWeight) { this.educationWeight = educationWeight; }

    public Double getPersonalityWeight() { return personalityWeight; }
    public void setPersonalityWeight(Double personalityWeight) { this.personalityWeight = personalityWeight; }

    public Double getCulturalFitWeight() { return culturalFitWeight; }
    public void setCulturalFitWeight(Double culturalFitWeight) { this.culturalFitWeight = culturalFitWeight; }

    public String getConfigName() { return configName; }
    public void setConfigName(String configName) { this.configName = configName; }

    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
}
