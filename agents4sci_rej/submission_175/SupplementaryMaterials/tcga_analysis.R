# TCGA Dataset Comprehensive Analysis
# Author: Claude Code
# Purpose: Analyze TCGA dataset structure for AI modeling feasibility

library(data.table)
library(dplyr)
library(ggplot2)
library(VennDiagram)
library(corrplot)

# Set working directory
setwd("agent4science")

# 1. EXAMINE RNA-SEQ DATA STRUCTURE
cat("=== TCGA RNA-seq Data Analysis ===\n")

# Get all RDS files
rna_files <- list.files("data/RNAseq_data", pattern = "\\.rds$", full.names = TRUE)
cancer_types <- gsub("_data\\.rds", "", basename(rna_files))

cat("Available cancer types:", length(cancer_types), "\n")
print(cancer_types)

# Function to analyze individual cancer type data
analyze_cancer_data <- function(file_path, cancer_type) {
  data <- readRDS(file_path)
  
  if (is.list(data)) {
    # Check if it's a list with expression and clinical data
    if ("expression" %in% names(data) || "clinical" %in% names(data)) {
      exp_data <- data$expression
      clin_data <- data$clinical
    } else {
      # Assume first element is expression data
      exp_data <- data[[1]]
      clin_data <- if(length(data) > 1) data[[2]] else NULL
    }
  } else {
    # Assume it's a matrix/data.frame of expression data
    exp_data <- data
    clin_data <- NULL
  }
  
  result <- list(
    cancer_type = cancer_type,
    n_samples = ifelse(is.matrix(exp_data) || is.data.frame(exp_data), 
                      ncol(exp_data), length(exp_data)),
    n_genes = ifelse(is.matrix(exp_data) || is.data.frame(exp_data), 
                    nrow(exp_data), NA),
    has_clinical = !is.null(clin_data),
    data_class = class(exp_data)[1],
    file_size_mb = file.size(file_path) / (1024^2)
  )
  
  return(result)
}

# Analyze all cancer types
cancer_summary <- map_dfr(seq_along(rna_files), function(i) {
  cat("Analyzing", cancer_types[i], "...\n")
  tryCatch({
    analyze_cancer_data(rna_files[i], cancer_types[i])
  }, error = function(e) {
    cat("Error analyzing", cancer_types[i], ":", e$message, "\n")
    return(data.frame(cancer_type = cancer_types[i], 
                     n_samples = NA, n_genes = NA, 
                     has_clinical = NA, data_class = "ERROR",
                     file_size_mb = file.size(rna_files[i]) / (1024^2)))
  })
})

print(cancer_summary)

# Detailed analysis of a representative large dataset (BRCA)
cat("\n=== Detailed Analysis of BRCA Dataset ===\n")
brca_data <- readRDS("data/RNAseq_data/BRCA_data.rds")
str(brca_data, max.level = 2)

# 2. ANALYZE CLINICAL DATA
cat("\n=== Clinical Data Analysis ===\n")
clinical_data <- readRDS("data/clinical_data/ALL_Cancer_clinical.rds")

cat("Clinical data structure:\n")
str(clinical_data, max.level = 1)

if (is.data.frame(clinical_data)) {
  cat("Number of patients:", nrow(clinical_data), "\n")
  cat("Number of clinical variables:", ncol(clinical_data), "\n")
  cat("Column names:\n")
  print(colnames(clinical_data))
  
  # Check for survival data
  survival_cols <- grep("time|event|status|death|survival|follow", 
                       colnames(clinical_data), ignore.case = TRUE)
  cat("\nPotential survival-related columns:\n")
  print(colnames(clinical_data)[survival_cols])
  
  # Check for staging/progression data
  stage_cols <- grep("stage|grade|tumor|progression|metast", 
                    colnames(clinical_data), ignore.case = TRUE)
  cat("\nPotential staging/progression columns:\n")
  print(colnames(clinical_data)[stage_cols])
  
  # Missing data analysis
  missing_data <- clinical_data %>%
    summarise_all(~sum(is.na(.) | . == "" | . == "Not Available")) %>%
    gather(variable, missing_count) %>%
    mutate(missing_percent = missing_count / nrow(clinical_data) * 100) %>%
    arrange(desc(missing_percent))
  
  cat("\nTop 10 variables with most missing data:\n")
  print(head(missing_data, 10))
}

# Save summary results
write.csv(cancer_summary, "tcga_cancer_summary.csv", row.names = FALSE)
cat("\nSummary saved to tcga_cancer_summary.csv\n")

cat("\n=== Analysis Complete ===\n")