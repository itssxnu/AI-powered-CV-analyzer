package com.cv.aiml_project.entity;

public enum HiringStatus {
    NOT_REVIEWED("Not Reviewed"),
    UNDER_REVIEW("Under Review"),
    SHORTLISTED("Shortlisted"),
    INTERVIEWED("Interviewed"),
    OFFER_EXTENDED("Offer Extended"),
    OFFER_ACCEPTED("Offer Accepted"),
    OFFER_DECLINED("Offer Declined"),
    REJECTED("Rejected"),
    HIRED("Hired");

    private final String displayName;

    HiringStatus(String displayName) {
        this.displayName = displayName;
    }

    public String getDisplayName() {
        return displayName;
    }
}
