package com.cv.aiml_project.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.web.SecurityFilterChain;

@Configuration
@EnableWebSecurity
public class SecurityConfig {

    private final UserDetailsService userDetailsService;

    public SecurityConfig(UserDetailsService userDetailsService) {
        this.userDetailsService = userDetailsService;
    }

    @Bean
    public AuthenticationManager authenticationManager(AuthenticationConfiguration authenticationConfiguration) throws Exception {
        return authenticationConfiguration.getAuthenticationManager();
    }

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
                .authorizeHttpRequests(auth -> auth
                        .requestMatchers("/", "/auth/**", "/css/**", "/js/**", "/images/**").permitAll()
                        .requestMatchers("/candidate/**").hasRole("CANDIDATE")
                        .requestMatchers("/hr/**").hasRole("HR")
                        .requestMatchers("/admin/**").hasRole("ADMIN")
                        .anyRequest().authenticated()
                )
                .formLogin(form -> form
                        .loginPage("/auth/login")
                        .loginProcessingUrl("/auth/login")
                        .successHandler((request, response, authentication) -> {
                            // Redirect based on role
                            String role = authentication.getAuthorities().iterator().next().getAuthority();

                            if (role.equals("ROLE_ADMIN")) {
                                response.sendRedirect("/admin/dashboard");
                            } else if (role.equals("ROLE_HR")) {
                                response.sendRedirect("/hr/dashboard");
                            } else if (role.equals("ROLE_CANDIDATE")) {
                                response.sendRedirect("/candidate/dashboard");
                            } else {
                                response.sendRedirect("/");
                            }
                        })
                        .failureUrl("/auth/login?error=true")
                        .permitAll()
                )
                .logout(logout -> logout
                        .logoutUrl("/auth/logout")
                        .logoutSuccessUrl("/auth/login?logout")
                        .invalidateHttpSession(true)
                        .deleteCookies("JSESSIONID")
                        .permitAll()
                )
                .exceptionHandling(exception -> exception
                        .accessDeniedPage("/auth/access-denied")
                )
                .userDetailsService(userDetailsService)
                .csrf(csrf -> csrf.disable());


        return http.build();
    }
}
