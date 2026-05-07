#!/usr/bin/env Rscript

.libPaths("/wanglab/rli/miniforge3/envs/r412/lib/R/library")
suppressPackageStartupMessages({
  library(dplyr)
  library(igraph)
})

# Load metadata and apply tissue mapping
meta <- read.csv("/scratch/rli/project/agent/covarnet/covarnet_metadata_filtered_v2.csv", row.names = 1)

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
  "Lymphatic/Immune" = c("mesenteric lymph node", "lymph node", "axilla"),
  "Endocrine_Tissue" = c("thyroid gland", "adrenal gland"),
  "Reproductive" = c("prostate gland"),
  "Nervous System" = c("brain"),
  "Salivary" = c("salivary gland epithelium"),
  "Other" = c("bone spine", "nasopharynx")
)

meta$tissue_category <- "Other"
for(category in names(tissue_mapping)) {
  tissues <- tissue_mapping[[category]]
  meta$tissue_category[meta$tissue %in% tissues] <- category
}

# Get tissue info for each cell type
cell_tissue_map <- meta %>%
  group_by(subCluster) %>%
  summarise(
    primary_tissue = names(sort(table(tissue), decreasing = TRUE))[1],
    primary_category = names(sort(table(tissue_category), decreasing = TRUE))[1],
    n_tissues = n_distinct(tissue),
    n_categories = n_distinct(tissue_category),
    .groups = "drop"
  )

# Load network data
network <- readRDS("/scratch/rli/project/agent/covarnet/network_K12.rds")
sig_edges <- network$global$edge[network$global$edge$pval_fdr < 0.05, ]

# Get endocrine edges and nodes
endocrine_edges <- sig_edges[
  sig_edges$majorCluster1 == "Endocrine" | sig_edges$majorCluster2 == "Endocrine",
]

endocrine_nodes <- unique(c(
  endocrine_edges$subCluster1[endocrine_edges$majorCluster1 == "Endocrine"],
  endocrine_edges$subCluster2[endocrine_edges$majorCluster2 == "Endocrine"]
))

cat("Found", length(endocrine_nodes), "endocrine cell types\n")

# Create subgraph
endo_subnet_edges <- endocrine_edges[
  endocrine_edges$subCluster1 %in% endocrine_nodes | 
  endocrine_edges$subCluster2 %in% endocrine_nodes,
]

g_endocrine <- graph_from_data_frame(
  d = endo_subnet_edges[, c("subCluster1", "subCluster2")],
  directed = FALSE
)

# Add edge weights for better layout
E(g_endocrine)$weight <- abs(endo_subnet_edges$correlation[
  match(paste(get.edgelist(g_endocrine)[,1], get.edgelist(g_endocrine)[,2]),
        paste(endo_subnet_edges$subCluster1, endo_subnet_edges$subCluster2))
])

# Prepare colors and labels
node_names <- V(g_endocrine)$name
node_colors <- character(length(node_names))
node_labels <- character(length(node_names))

# PASTEL COLOR PALETTE
pastel_colors <- c(
  "Small Intestine" = "#B8E6D3",     # Soft mint green
  "Large Intestine" = "#B8D4E6",     # Soft sky blue  
  "Stomach" = "#FFFACD",             # Lemon chiffon
  "Lung/Respiratory" = "#E6D3E6",    # Soft lavender
  "Liver/Biliary" = "#FFE4E1",       # Misty rose
  "Pancreas" = "#FFDAB9",            # Peach puff
  "Lymphatic/Immune" = "#D4E6B8",    # Soft lime green
  "Esophagus" = "#FFE0EC",           # Light pink
  "Endocrine_Tissue" = "#E0D3F0",    # Pale purple
  "Nervous System" = "#FFF8DC",      # Cornsilk
  "Salivary" = "#FFEBF0",            # Lavender blush
  "Other" = "#F0F0F0"                # Light gray
)

# Pastel red for endocrine cells
endocrine_color <- "#FFB6C1"  # Light pink-red

# Assign colors and create shorter labels for less overlap
for(i in seq_along(node_names)) {
  node_name <- node_names[i]
  
  if(node_name %in% endocrine_nodes) {
    # Endocrine nodes
    node_colors[i] <- endocrine_color
    # Shorten long endocrine names
    if(grepl("enteroendocrine", node_name)) {
      node_labels[i] <- gsub("enteroendocrine cell", "EEC", node_name)
      node_labels[i] <- gsub("type ", "", node_labels[i])
    } else if(grepl("neuroendocrine", node_name)) {
      node_labels[i] <- gsub("neuroendocrine cell", "NEC", node_labels[i])
    } else {
      node_labels[i] <- node_name
    }
  } else {
    # Non-endocrine nodes - add tissue annotation
    tissue_info <- cell_tissue_map[cell_tissue_map$subCluster == node_name,]
    if(nrow(tissue_info) > 0) {
      category <- tissue_info$primary_category[1]
      
      # Set pastel color based on tissue category
      if(category %in% names(pastel_colors)) {
        node_colors[i] <- pastel_colors[category]
      } else {
        node_colors[i] <- "#F0F0F0"
      }
      
      # Create shorter labels for common cell types
      short_name <- node_name
      short_name <- gsub("activated ", "act. ", short_name)
      short_name <- gsub("positive", "+", short_name)
      short_name <- gsub("negative", "-", short_name)
      short_name <- gsub("alpha-beta", "αβ", short_name)
      short_name <- gsub("memory ", "mem. ", short_name)
      short_name <- gsub("regulatory ", "reg. ", short_name)
      short_name <- gsub("intestinal ", "int. ", short_name)
      short_name <- gsub("epithelial cell", "EC", short_name)
      
      # Create abbreviated tissue category labels
      cat_short <- category
      cat_short <- gsub("Small Intestine", "SI", cat_short)
      cat_short <- gsub("Large Intestine", "LI", cat_short)
      cat_short <- gsub("Lymphatic/Immune", "Immune", cat_short)
      cat_short <- gsub("Lung/Respiratory", "Lung", cat_short)
      cat_short <- gsub("Liver/Biliary", "Liver", cat_short)
      
      # Create compact label
      node_labels[i] <- paste0(short_name, "\n[", cat_short, "]")
      
      # Only add tissue count if > 2
      if(tissue_info$n_tissues[1] > 2) {
        node_labels[i] <- paste0(node_labels[i], " (", tissue_info$n_tissues[1], "t)")
      }
    } else {
      node_colors[i] <- "#F0F0F0"
      node_labels[i] <- paste0(node_name, "\n[?]")
    }
  }
}

V(g_endocrine)$color <- node_colors
V(g_endocrine)$label <- node_labels

# Size nodes by degree, with endocrine nodes larger
V(g_endocrine)$size <- 10 + 15 * (degree(g_endocrine) / max(degree(g_endocrine)))
is_endocrine <- V(g_endocrine)$name %in% endocrine_nodes
V(g_endocrine)$size[is_endocrine] <- V(g_endocrine)$size[is_endocrine] * 1.4

# Create plot with improved layout
pdf("/scratch/rli/project/agent/covarnet/network_endocrine_tissue_pastel.pdf", width=18, height=16)
par(mar=c(2,2,4,2))
set.seed(123)  # Different seed for better layout

# Use Kamada-Kawai layout for better spacing
layout_endo <- layout_with_kk(g_endocrine, weights = E(g_endocrine)$weight)

# Scale layout to spread nodes more
layout_endo <- layout_endo * 1.2

# Plot with adjusted parameters
plot(g_endocrine,
     layout = layout_endo,
     vertex.label = V(g_endocrine)$label,
     vertex.label.cex = 0.45,  # Smaller font size
     vertex.label.color = "black",
     vertex.label.dist = 0.4,   # Increased distance from node
     vertex.label.font = 1,
     vertex.label.degree = -pi/2,  # Labels below nodes
     edge.width = 0.5,          # Thinner edges
     edge.color = adjustcolor("gray60", alpha=0.3),  # Lighter edges
     edge.curved = 0.1,         # Slight curve to edges
     main = "Endocrine Cell Subnetwork with Tissue Annotations (Pastel Palette)",
     sub = paste("Soft pink: Endocrine cells | Pastel colors: Non-endocrine by tissue |", 
                vcount(g_endocrine), "nodes,", ecount(g_endocrine), "edges"))

# Get non-endocrine nodes for legend
is_not_endocrine <- !(V(g_endocrine)$name %in% endocrine_nodes)
non_endocrine_nodes <- V(g_endocrine)$name[is_not_endocrine]

# Get used categories
used_categories <- unique(cell_tissue_map$primary_category[
  cell_tissue_map$subCluster %in% non_endocrine_nodes
])

# Create legend with pastel colors
legend_labels <- c("Endocrine cells", used_categories)
legend_colors <- c(endocrine_color)
for(cat in used_categories) {
  if(cat %in% names(pastel_colors)) {
    legend_colors <- c(legend_colors, pastel_colors[cat])
  } else {
    legend_colors <- c(legend_colors, "#F0F0F0")
  }
}

legend("topright", 
       legend = legend_labels, 
       fill = legend_colors,
       title = "Cell Type Categories", 
       cex = 0.65, 
       bty = "n",
       bg = "white")

dev.off()

# Create a second version with different layout algorithm for comparison
cat("\nCreating alternative layout version...\n")

pdf("/scratch/rli/project/agent/covarnet/network_endocrine_tissue_pastel_v2.pdf", width=20, height=18)
par(mar=c(2,2,4,2))
set.seed(456)

# Use Fruchterman-Reingold with repulsion tuning
layout_endo2 <- layout_with_fr(g_endocrine, 
                               weights = E(g_endocrine)$weight,
                               niter = 1000,  # More iterations
                               area = vcount(g_endocrine)^2.5,  # More area
                               repulserad = vcount(g_endocrine)^2.8)  # More repulsion

# Scale layout
layout_endo2 <- layout_endo2 * 1.3

plot(g_endocrine,
     layout = layout_endo2,
     vertex.label = V(g_endocrine)$label,
     vertex.label.cex = 0.4,    # Even smaller font
     vertex.label.color = "black",
     vertex.label.dist = 0.5,   # More distance
     vertex.label.font = 1,
     vertex.label.degree = seq(-pi, pi, length.out = vcount(g_endocrine)),  # Spread labels around
     edge.width = 0.5,
     edge.color = adjustcolor("gray60", alpha=0.3),
     edge.curved = 0.15,        # More curve to avoid label overlap
     vertex.frame.color = adjustcolor("gray40", alpha=0.5),  # Add subtle border
     main = "Endocrine Cell Subnetwork (Alternative Layout - Pastel Palette)",
     sub = paste("Soft pink: Endocrine | Pastel colors: Tissue categories |", 
                "Font size reduced, spacing increased"))

# Add legend
legend("topright", 
       legend = legend_labels, 
       fill = legend_colors,
       title = "Cell Type Categories", 
       cex = 0.6, 
       bty = "n",
       bg = adjustcolor("white", alpha=0.9))

dev.off()

# Print summary
cat("\n✓ Created: network_endocrine_tissue_pastel.pdf (Kamada-Kawai layout)\n")
cat("✓ Created: network_endocrine_tissue_pastel_v2.pdf (Fruchterman-Reingold layout)\n\n")

cat("Improvements implemented:\n")
cat("  • Pastel color palette for all nodes\n")
cat("  • Reduced font size (0.45 and 0.4)\n")
cat("  • Abbreviated long cell type names\n")
cat("  • Increased label distance from nodes\n")
cat("  • Two layout algorithms for optimal spacing\n")
cat("  • Larger canvas size (18x16 and 20x18)\n")
cat("  • Lighter edge colors for clarity\n")

cat("\nNetwork Summary:\n")
cat("  Endocrine nodes:", sum(is_endocrine), "\n")
cat("  Non-endocrine nodes:", sum(is_not_endocrine), "\n")
cat("  Total edges:", ecount(g_endocrine), "\n")

# Show tissue category distribution
tissue_dist <- table(cell_tissue_map$primary_category[
  cell_tissue_map$subCluster %in% non_endocrine_nodes
])
cat("\nNon-endocrine cells by tissue category:\n")
for(i in 1:length(tissue_dist)) {
  cat(sprintf("  • %s: %d cell types\n", names(tissue_dist)[i], tissue_dist[i]))
}

cat("\n✅ Pastel palette tissue-annotated networks complete!\n")