#!/usr/bin/env Rscript

# Cross-tissue analysis of endocrine cells from CoVarNet results

.libPaths("/wanglab/rli/miniforge3/envs/r412/lib/R/library")

suppressPackageStartupMessages({
  library(CoVarNet)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(pheatmap)
  library(RColorBrewer)
})

cat("================================================================================\n")
cat("Cross-Tissue Analysis of Endocrine Cells\n")
cat("================================================================================\n\n")

# Load metadata and results
meta <- read.csv("/scratch/rli/project/agent/covarnet/covarnet_metadata_filtered_v2.csv", row.names = 1)
network <- readRDS("/scratch/rli/project/agent/covarnet/network_K12.rds")
cor_pair <- readRDS("/scratch/rli/project/agent/covarnet/cor_pair_K12.rds")
endocrine_enrichment <- readRDS("/scratch/rli/project/agent/covarnet/endocrine_enrichment_K12.rds")

# Apply tissue mapping
tissue_mapping <- list(
  "Stomach" = c("body of stomach", "cardia of stomach", "corpus", "pyloric antrum", "stomach"),
  "Small Intestine" = c("ileum", "small intestine", "ileal epithelium", "duodenum", "intestine", 
                        "epithelium of small intestine", "jejunum", "hindgut", "lamina propria of small intestine"),
  "Large Intestine" = c("colon", "rectum", "large intestine", "sigmoid colon", "transverse colon", 
                        "ascending colon", "caecum", "vermiform appendix", "intestinal mucosa", "descending colon"),
  "Esophagus" = c("lower esophagus", "esophagogastric junction", "submucosal esophageal gland"),
  "Liver/Biliary" = c("liver", "intrahepatic bile duct", "common bile duct", "gallbladder", "biliary system"),
  "Pancreas" = c("pancreas", "islet of Langerhans"),
  "Lung/Respiratory" = c("lung", "alveolar sac", "bronchus", "pleural effusion"),
  "Other" = c("mesenteric lymph node", "lymph node", "axilla", "thyroid gland", "adrenal gland",
              "prostate gland", "brain", "salivary gland epithelium", "bone spine", "nasopharynx")
)

meta$tissue_category <- "Other"
for(category in names(tissue_mapping)) {
  tissues <- tissue_mapping[[category]]
  meta$tissue_category[meta$tissue %in% tissues] <- category
}

# ============================================================================
# PART 1: Endocrine Cell Distribution Analysis
# ============================================================================
cat("PART 1: Endocrine Cell Distribution Across Tissues\n")
cat("="*60, "\n\n")

# Get endocrine cells
endocrine_cells <- meta[meta$majorCluster == "Endocrine", ]
cat("Total endocrine cells:", nrow(endocrine_cells), "\n\n")

# Endocrine distribution by tissue category
endocrine_by_tissue <- endocrine_cells %>%
  group_by(tissue_category) %>%
  summarise(
    n_cells = n(),
    n_samples = n_distinct(sampleID),
    n_subtypes = n_distinct(subCluster),
    pct_of_endocrine = 100 * n() / nrow(endocrine_cells)
  ) %>%
  arrange(desc(n_cells))

cat("Endocrine cells by tissue category:\n")
print(as.data.frame(endocrine_by_tissue))

# Endocrine subtypes distribution
endocrine_subtypes <- endocrine_cells %>%
  group_by(subCluster) %>%
  summarise(
    n_cells = n(),
    n_tissues = n_distinct(tissue),
    n_categories = n_distinct(tissue_category),
    tissues = paste(unique(head(tissue, 3)), collapse=", ")
  ) %>%
  arrange(desc(n_cells))

cat("\n\nTop endocrine subtypes:\n")
print(head(as.data.frame(endocrine_subtypes), 10))

# ============================================================================
# PART 2: Cross-Tissue Endocrine Patterns
# ============================================================================
cat("\n\nPART 2: Cross-Tissue Endocrine Patterns\n")
cat("="*60, "\n\n")

# Create endocrine subtype by tissue matrix
endocrine_tissue_matrix <- endocrine_cells %>%
  count(subCluster, tissue_category) %>%
  pivot_wider(names_from = tissue_category, values_from = n, values_fill = 0)

# Convert to matrix for heatmap
mat_endocrine <- as.matrix(endocrine_tissue_matrix[,-1])
rownames(mat_endocrine) <- endocrine_tissue_matrix$subCluster

# Identify pan-tissue vs tissue-specific endocrine types
tissue_specificity <- apply(mat_endocrine > 0, 1, sum)
pan_tissue <- names(tissue_specificity[tissue_specificity >= 3])
tissue_specific <- names(tissue_specificity[tissue_specificity == 1])

cat("Pan-tissue endocrine types (present in ≥3 tissue categories):\n")
for(cell in pan_tissue) {
  tissues_present <- names(mat_endocrine[cell,][mat_endocrine[cell,] > 0])
  cat(sprintf("  • %s: %s\n", cell, paste(tissues_present, collapse=", ")))
}

cat("\n\nTissue-specific endocrine types:\n")
for(cell in tissue_specific) {
  tissue_present <- names(mat_endocrine[cell,][mat_endocrine[cell,] > 0])
  count <- mat_endocrine[cell, tissue_present]
  cat(sprintf("  • %s: %s only (%d cells)\n", cell, tissue_present, count))
}

# Create heatmap of endocrine distribution
pdf("/scratch/rli/project/agent/covarnet/endocrine_tissue_distribution_heatmap.pdf", width=10, height=8)
pheatmap(log10(mat_endocrine + 1),
         main = "Endocrine Subtypes Across Tissue Categories",
         cluster_rows = TRUE,
         cluster_cols = TRUE,
         color = colorRampPalette(c("white", "#FFE4E1", "#FF69B4", "#FF1493", "#8B008B"))(100),
         fontsize = 8,
         labels_row = substr(rownames(mat_endocrine), 1, 30),
         angle_col = 45,
         legend_title = "log10(cells+1)")
dev.off()
cat("\n✓ Created: endocrine_tissue_distribution_heatmap.pdf\n")

# ============================================================================
# PART 3: Module-Tissue Relationships
# ============================================================================
cat("\n\nPART 3: Module-Tissue Relationships\n")
cat("="*60, "\n\n")

# Analyze which modules are enriched in which tissues
if(!is.null(network$filter)) {
  module_tissue <- network$filter %>%
    left_join(meta[,c("subCluster", "tissue_category", "majorCluster")], 
              by = "subCluster", 
              relationship = "many-to-many") %>%
    filter(majorCluster == "Endocrine") %>%
    group_by(cm, tissue_category) %>%
    summarise(
      n_connections = n(),
      mean_weight = mean(weight, na.rm = TRUE),
      .groups = 'drop'
    )
  
  if(nrow(module_tissue) > 0) {
    # Create module-tissue matrix
    module_tissue_mat <- module_tissue %>%
      select(cm, tissue_category, n_connections) %>%
      pivot_wider(names_from = tissue_category, values_from = n_connections, values_fill = 0)
    
    cat("Module enrichment by tissue:\n")
    print(as.data.frame(module_tissue_mat))
  }
}

# Top endocrine-enriched modules
cat("\n\nEndocrine-enriched modules (from previous analysis):\n")
endocrine_sorted <- sort(endocrine_enrichment, decreasing = TRUE)
for(i in 1:min(5, length(endocrine_sorted))) {
  cat(sprintf("  %s: %.1f%% enrichment\n", 
              names(endocrine_sorted)[i], 
              100 * endocrine_sorted[i] / sum(endocrine_sorted)))
}

# ============================================================================
# PART 4: Endocrine Network Connectivity Analysis
# ============================================================================
cat("\n\nPART 4: Endocrine Network Connectivity Patterns\n")
cat("="*60, "\n\n")

# Analyze endocrine correlations
endocrine_cors <- cor_pair[
  cor_pair$majorCluster1 == "Endocrine" | cor_pair$majorCluster2 == "Endocrine",
]

# Get significant correlations
sig_endocrine <- endocrine_cors[endocrine_cors$pval_fdr < 0.05, ]
cat("Significant endocrine correlations (FDR < 0.05):", nrow(sig_endocrine), "\n\n")

# Analyze what cell types endocrine cells connect with
endocrine_partners <- data.frame()
for(i in 1:nrow(sig_endocrine)) {
  if(sig_endocrine$majorCluster1[i] == "Endocrine") {
    partner_cluster <- sig_endocrine$majorCluster2[i]
    partner_cell <- sig_endocrine$subCluster2[i]
  } else {
    partner_cluster <- sig_endocrine$majorCluster1[i]
    partner_cell <- sig_endocrine$subCluster1[i]
  }
  endocrine_partners <- rbind(endocrine_partners, 
                              data.frame(partner_cluster = partner_cluster,
                                       partner_cell = partner_cell,
                                       correlation = abs(sig_endocrine$correlation[i])))
}

# Summarize partner cell types
partner_summary <- endocrine_partners %>%
  group_by(partner_cluster) %>%
  summarise(
    n_connections = n(),
    mean_correlation = mean(correlation),
    n_unique_cells = n_distinct(partner_cell)
  ) %>%
  arrange(desc(n_connections))

cat("Endocrine cells primarily connect with:\n")
print(as.data.frame(partner_summary))

# ============================================================================
# PART 5: Conserved vs Variable Endocrine Patterns
# ============================================================================
cat("\n\nPART 5: Conserved vs Variable Endocrine Patterns\n")
cat("="*60, "\n\n")

# Identify conserved endocrine interactions across tissues
if(nrow(sig_endocrine) > 0) {
  # Get endocrine-endocrine interactions
  endo_endo <- sig_endocrine[
    sig_endocrine$majorCluster1 == "Endocrine" & 
    sig_endocrine$majorCluster2 == "Endocrine",
  ]
  
  if(nrow(endo_endo) > 0) {
    cat("Endocrine-Endocrine interactions:", nrow(endo_endo), "\n")
    cat("Top interactions:\n")
    top_endo <- head(endo_endo[order(abs(endo_endo$correlation), decreasing = TRUE), ], 5)
    for(i in 1:nrow(top_endo)) {
      cat(sprintf("  • %s <-> %s: r=%.3f\n", 
                  top_endo$subCluster1[i], 
                  top_endo$subCluster2[i],
                  top_endo$correlation[i]))
    }
  }
  
  # Endocrine-Epithelial interactions
  endo_epi <- sig_endocrine[
    (sig_endocrine$majorCluster1 == "Endocrine" & sig_endocrine$majorCluster2 == "Epithelial") |
    (sig_endocrine$majorCluster1 == "Epithelial" & sig_endocrine$majorCluster2 == "Endocrine"),
  ]
  
  if(nrow(endo_epi) > 0) {
    cat("\n\nEndocrine-Epithelial interactions:", nrow(endo_epi), "\n")
  }
  
  # Endocrine-Immune interactions
  endo_immune <- sig_endocrine[
    (sig_endocrine$majorCluster1 == "Endocrine" & sig_endocrine$majorCluster2 == "Immune") |
    (sig_endocrine$majorCluster1 == "Immune" & sig_endocrine$majorCluster2 == "Endocrine"),
  ]
  
  if(nrow(endo_immune) > 0) {
    cat("Endocrine-Immune interactions:", nrow(endo_immune), "\n")
  }
}

# ============================================================================
# PART 6: Generate Summary Visualization
# ============================================================================
cat("\n\nGenerating summary visualizations...\n")

# Create a summary plot showing endocrine distribution
pdf("/scratch/rli/project/agent/covarnet/endocrine_cross_tissue_summary.pdf", width=14, height=10)
par(mfrow=c(2,2), mar=c(5,4,4,2))

# Plot 1: Endocrine cells by tissue category
barplot(endocrine_by_tissue$n_cells,
        names.arg = endocrine_by_tissue$tissue_category,
        col = colorRampPalette(c("#FFE4E1", "#FF69B4"))(nrow(endocrine_by_tissue)),
        las = 2,
        main = "Endocrine Cells by Tissue Category",
        ylab = "Number of cells",
        cex.names = 0.8)

# Plot 2: Endocrine subtypes diversity
barplot(table(tissue_specificity),
        col = c("#B8E6D3", "#FFB6C1", "#B8D4E6", "#FFFFB3"),
        main = "Tissue Specificity of Endocrine Subtypes",
        xlab = "Number of tissue categories",
        ylab = "Number of endocrine subtypes")

# Plot 3: Module enrichment
if(length(endocrine_enrichment) > 0) {
  barplot(sort(endocrine_enrichment, decreasing = TRUE)[1:min(8, length(endocrine_enrichment))],
          col = heat.colors(8),
          las = 2,
          main = "Top Endocrine-Enriched Modules",
          ylab = "Enrichment score",
          cex.names = 0.8)
}

# Plot 4: Partner cell types
if(nrow(partner_summary) > 0) {
  pie(partner_summary$n_connections,
      labels = paste0(partner_summary$partner_cluster, "\n(", partner_summary$n_connections, ")"),
      col = rainbow(nrow(partner_summary)),
      main = "Endocrine Network Partners")
}

dev.off()
cat("✓ Created: endocrine_cross_tissue_summary.pdf\n")

# ============================================================================
# Key Insights Summary
# ============================================================================
cat("\n\n")
cat("="*80, "\n")
cat("KEY INSIGHTS FROM CROSS-TISSUE ENDOCRINE ANALYSIS\n")
cat("="*80, "\n\n")

cat("1. TISSUE DISTRIBUTION:\n")
cat(sprintf("   • Endocrine cells found in %d tissue categories\n", n_distinct(endocrine_cells$tissue_category)))
cat(sprintf("   • Highest concentration in: %s (%.1f%%)\n", 
            endocrine_by_tissue$tissue_category[1], 
            endocrine_by_tissue$pct_of_endocrine[1]))
cat(sprintf("   • Most diverse tissue: %d endocrine subtypes\n", max(endocrine_by_tissue$n_subtypes)))

cat("\n2. CELLULAR DIVERSITY:\n")
cat(sprintf("   • %d distinct endocrine subtypes identified\n", n_distinct(endocrine_cells$subCluster)))
cat(sprintf("   • %d pan-tissue types (≥3 tissues)\n", length(pan_tissue)))
cat(sprintf("   • %d tissue-specific types\n", length(tissue_specific)))

cat("\n3. NETWORK PATTERNS:\n")
cat(sprintf("   • %d significant endocrine connections\n", nrow(sig_endocrine)))
cat(sprintf("   • Primary partner: %s cells (%d connections)\n", 
            partner_summary$partner_cluster[1], 
            partner_summary$n_connections[1]))
cat(sprintf("   • Mean correlation strength: %.3f\n", mean(abs(sig_endocrine$correlation))))

cat("\n4. MODULE ORGANIZATION:\n")
top_module <- names(endocrine_sorted)[1]
cat(sprintf("   • Top endocrine module: %s (%.1f%% enrichment)\n", 
            top_module, 100 * endocrine_sorted[1] / sum(endocrine_sorted)))
cat(sprintf("   • %d modules with >10%% endocrine enrichment\n", 
            sum((endocrine_enrichment / sum(endocrine_enrichment)) > 0.1)))

cat("\n5. CONSERVATION PATTERNS:\n")
if(length(pan_tissue) > 0) {
  cat(sprintf("   • Most conserved: %s\n", pan_tissue[1]))
}
cat(sprintf("   • Tissue categories with unique endocrine types: %d\n", 
            sum(apply(mat_endocrine, 2, function(x) sum(x > 0 & rowSums(mat_endocrine > 0) == 1)) > 0)))

cat("\n✅ Cross-tissue endocrine analysis complete!\n")

# Save analysis results
save(endocrine_by_tissue, endocrine_subtypes, partner_summary, tissue_specificity,
     file = "/scratch/rli/project/agent/covarnet/endocrine_cross_tissue_results.RData")
cat("\n✓ Results saved to: endocrine_cross_tissue_results.RData\n")