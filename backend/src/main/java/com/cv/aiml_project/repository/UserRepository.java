package com.cv.aiml_project.repository;

import com.cv.aiml_project.entity.Role;
import com.cv.aiml_project.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {

    Optional<User> findByUsername(String username);
    Optional<User> findByEmail(String email);

    boolean existsByUsername(String username);
    boolean existsByEmail(String email);

    // Find users by role
    List<User> findByRole(Role role);
    List<User> findByRoleAndIsActive(Role role, boolean isActive);

    // Count users by role
    long countByRole(Role role);
    long countByRoleAndIsActive(Role role, boolean isActive);

    // Find active users
    List<User> findByIsActive(boolean isActive);

    // Search users
    @Query("SELECT u FROM User u WHERE " +
            "LOWER(u.firstName) LIKE LOWER(CONCAT('%', :keyword, '%')) OR " +
            "LOWER(u.lastName) LIKE LOWER(CONCAT('%', :keyword, '%')) OR " +
            "LOWER(u.email) LIKE LOWER(CONCAT('%', :keyword, '%')) OR " +
            "LOWER(u.skills) LIKE LOWER(CONCAT('%', :keyword, '%'))")
    List<User> searchUsers(String keyword);

    // AI/ML specific queries
    //List<User> findByMlProcessed(boolean mlProcessed);
   // List<User> findByRoleAndMlProcessed(Role role, boolean mlProcessed);
}
