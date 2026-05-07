import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import sys

# Set CRAN mirror to prevent interactive prompts
robjects.r('options(repos = c(CRAN = "https://cloud.r-project.org"))')

# Install required packages
robjects.r('if (!require(pwr)) install.packages("pwr")')
robjects.r('if (!require(MASS)) install.packages("MASS")')
robjects.r('if (!require(ordinal)) install.packages("ordinal")')

# Load libraries
robjects.r('library(pwr)')
robjects.r('library(MASS)')
robjects.r('library(ordinal)')

# Power analysis for ordinal logistic regression
robjects.r('''
# Print justifications and parameters
cat("POWER ANALYSIS FOR EDUCATIONAL ATTAINMENT AND COGNITIVE PROFILE HETEROGENEITY\n")
cat("============================================================================\n\n")

cat("STUDY PARAMETERS:\n")
cat("- Primary Hypothesis: 8-level ordinal educational attainment predictor\n")
cat("- Secondary Hypothesis: Education x Age group interaction\n")
cat("- Statistical Power: 0.80\n")
cat("- Alpha level: 0.025 (Bonferroni corrected for two heterogeneity metrics)\n")
cat("- Effect sizes based on conservative estimates for educational psychology research\n\n")

# Parameters for primary hypothesis
power_target <- 0.80
alpha_corrected <- 0.025
num_education_levels <- 8
num_age_groups <- 3

# Conservative effect size estimates for ordinal logistic regression
# Based on Cohen's conventions adapted for ordinal regression
# Small effect: OR = 1.2-1.5, Medium effect: OR = 1.5-2.0
# Using conservative small-to-medium effect size OR = 1.3

primary_effect_or <- 1.3
secondary_effect_or <- 1.15  # Smaller effect for interaction

cat("EFFECT SIZE JUSTIFICATIONS:\n")
cat("- Primary effect (education main effect): OR = 1.3\n")
cat("- This represents a conservative small-to-medium effect size\n")
cat("- Based on educational psychology literature showing modest but consistent\n")
cat("  relationships between educational attainment and cognitive differentiation\n")
cat("- Secondary effect (interaction): OR = 1.15\n")
cat("- Interaction effects are typically smaller than main effects\n\n")

# Convert OR to Cohen's d for power calculation
# Using approximation: d = ln(OR) * sqrt(3)/pi
primary_cohens_d <- log(primary_effect_or) * sqrt(3)/pi
secondary_cohens_d <- log(secondary_effect_or) * sqrt(3)/pi

cat("CONVERTED EFFECT SIZES:\n")
cat("- Primary effect Cohen's d:", round(primary_cohens_d, 3), "\n")
cat("- Secondary effect Cohen's d:", round(secondary_cohens_d, 3), "\n\n")

# Power analysis for primary hypothesis using pwr.f2.test
# For ordinal regression with 8 education levels
# df1 = k-1 = 7 (education levels - 1)
# Including covariates: age group (2 df), other covariates (~3 df)
df1_primary <- 7
df2_covariates <- 5  # age groups + other covariates

# Calculate f2 effect size from Cohen's d
f2_primary <- primary_cohens_d^2 / (1 - primary_cohens_d^2)

cat("PRIMARY HYPOTHESIS POWER ANALYSIS:\n")
cat("- Degrees of freedom (numerator):", df1_primary, "\n")
cat("- f2 effect size:", round(f2_primary, 4), "\n")

# Power analysis for primary effect
primary_power_result <- pwr.f2.test(
  u = df1_primary,
  v = NULL,
  f2 = f2_primary,
  sig.level = alpha_corrected,
  power = power_target
)

initial_n_primary <- primary_power_result$v + df1_primary + df2_covariates + 1
cat("- Initial sample size (primary):", ceiling(initial_n_primary), "\n\n")

# Power analysis for secondary hypothesis (interaction)
# Additional df for interaction terms
df1_interaction <- 2  # education x age interaction
f2_secondary <- secondary_cohens_d^2 / (1 - secondary_cohens_d^2)

cat("SECONDARY HYPOTHESIS POWER ANALYSIS:\n")
cat("- Degrees of freedom for interaction:", df1_interaction, "\n")
cat("- f2 effect size:", round(f2_secondary, 4), "\n")

secondary_power_result <- pwr.f2.test(
  u = df1_interaction,
  v = NULL,
  f2 = f2_secondary,
  sig.level = alpha_corrected,
  power = power_target
)

initial_n_secondary <- secondary_power_result$v + df1_primary + df1_interaction + df2_covariates + 1
cat("- Initial sample size (secondary):", ceiling(initial_n_secondary), "\n\n")

# Take the larger of the two sample sizes
initial_sample_size <- max(ceiling(initial_n_primary), ceiling(initial_n_secondary))

cat("SAMPLE SIZE CALCULATION:\n")
cat("- Required sample size for primary hypothesis:", ceiling(initial_n_primary), "\n")
cat("- Required sample size for secondary hypothesis:", ceiling(initial_n_secondary), "\n")
cat("- Initial sample size (maximum of both):", initial_sample_size, "\n\n")

# Add buffer for attrition and missing data (20% typical for cognitive studies)
attrition_rate <- 0.20
final_sample_size <- ceiling(initial_sample_size / (1 - attrition_rate))

cat("FINAL SAMPLE SIZE WITH ATTRITION BUFFER:\n")
cat("- Attrition rate assumed:", attrition_rate * 100, "%\n")
cat("- Final sample size:", final_sample_size, "\n\n")

# Sample size per age group for stratified analysis
sample_per_age_group <- ceiling(final_sample_size / num_age_groups)

cat("STRATIFIED SAMPLE SIZES:\n")
cat("- Sample size per age group:", sample_per_age_group, "\n")
cat("- Age groups: 18-39, 40-49, 50+\n")
cat("- Total across all age groups:", sample_per_age_group * num_age_groups, "\n\n")

# Print the required output format
cat("SUMMARY OUTPUT:\n")
''')

# Print initial and final sample sizes in the required format
robjects.r('''
print(paste("initial sample size:", initial_sample_size))
print(paste("final sample size:", final_sample_size))
''')

robjects.r('''
cat("\nMETHODOLOGICAL NOTES:\n")
cat("- Power analysis conducted using pwr.f2.test for multiple regression framework\n")
cat("- Effect sizes based on conservative estimates for educational psychology\n")
cat("- Bonferroni correction applied for multiple heterogeneity metrics\n")
cat("- Sample size accounts for ordinal nature of educational attainment predictor\n")
cat("- Stratified sampling recommended to ensure adequate representation across age groups\n")
''')

print("Power analysis completed successfully")