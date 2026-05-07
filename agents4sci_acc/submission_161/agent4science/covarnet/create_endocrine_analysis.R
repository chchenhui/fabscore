#!/usr/bin/env Rscript

# Create enhanced network visualizations focusing on endocrine relationships
# and annotated community detection

.libPaths("/wanglab/rli/miniforge3/envs/r412/lib/R/library")

suppressPackageStartupMessages({
  library(CoVarNet)
  library(igraph)
  library(dplyr)
  library(RColorBrewer)
})

cat("================================================================================\n")
cat("Creating Endocrine-Focused Network Visualizations\n")
cat("================================================================================\n\n")

# Load the saved network data
cat("Loading network data...\n")
network <- readRDS("/scratch/rli/project/agent/covarnet/network_K12.rds")
cor_pair <- readRDS("/scratch/rli/project/agent/covarnet/cor_pair_K12.rds")

# Check network structure
if(is.null(network$global) || !("edge" %in% names(network$global))) {
  stop("network$global does not have the expected structure")
}

cat("  Found", nrow(network$global$edge), "total edges\n")
cat("  Found", nrow(network$global$node), "nodes\n\n")

# Filter for significant edges
sig_edges <- network$global$edge[network$global$edge$pval_fdr < 0.05, ]
cat("  Using", nrow(sig_edges), "significant edges (FDR < 0.05)\n\n")

# ============================================================================
# PART 1: Create Endocrine-Focused Global Network with Relationship Strength
# ============================================================================
cat("Creating endocrine-focused global network with relationship strength...\n")

# Identify endocrine-related edges
endocrine_edges <- sig_edges[
  sig_edges$majorCluster1 == "Endocrine" | sig_edges$majorCluster2 == "Endocrine",
]
cat("  Found", nrow(endocrine_edges), "edges involving endocrine cells\n")

# Create full graph for context
g_full <- graph_from_data_frame(
  d = sig_edges[, c("subCluster1", "subCluster2")],
  directed = FALSE
)

# Add edge attributes
E(g_full)$correlation <- abs(sig_edges$correlation)
E(g_full)$pval <- sig_edges$pval_fdr

# Identify which edges are endocrine-related
edge_list <- as_edgelist(g_full)
is_endocrine_edge <- rep(FALSE, ecount(g_full))
for(i in 1:ecount(g_full)) {
  edge_nodes <- edge_list[i,]
  # Check if this edge involves endocrine cells
  is_endocrine <- any(
    (sig_edges$subCluster1 %in% edge_nodes & sig_edges$majorCluster1 == "Endocrine") |
    (sig_edges$subCluster2 %in% edge_nodes & sig_edges$majorCluster2 == "Endocrine")
  )
  is_endocrine_edge[i] <- is_endocrine
}

# Color edges based on endocrine involvement and correlation strength
edge_colors <- rep(adjustcolor("gray70", alpha=0.3), ecount(g_full))
endocrine_correlations <- E(g_full)$correlation[is_endocrine_edge]

# Create color gradient for endocrine edges based on correlation strength
if(sum(is_endocrine_edge) > 0) {
  # Use red gradient for endocrine edges
  correlation_range <- range(endocrine_correlations, na.rm = TRUE)
  edge_colors[is_endocrine_edge] <- adjustcolor(
    colorRampPalette(c("#FFB6C1", "#FF1493", "#8B0000"))(100)[
      cut(endocrine_correlations, breaks = 100, labels = FALSE)
    ],
    alpha = 0.7
  )
}

# Set edge widths based on correlation strength
edge_widths <- 0.3 + 3 * (E(g_full)$correlation / max(E(g_full)$correlation, na.rm = TRUE))
edge_widths[is_endocrine_edge] <- edge_widths[is_endocrine_edge] * 1.5  # Make endocrine edges thicker

# Color nodes based on major cluster with endocrine highlighted
node_names <- V(g_full)$name
node_colors <- rep("#BEBADA", length(node_names))  # Default gray

# Get major cluster for each node
for(i in seq_along(node_names)) {
  node_name <- node_names[i]
  # Find major cluster
  cluster1 <- sig_edges$majorCluster1[sig_edges$subCluster1 == node_name][1]
  cluster2 <- sig_edges$majorCluster2[sig_edges$subCluster2 == node_name][1]
  major_cluster <- ifelse(!is.na(cluster1), cluster1, cluster2)
  
  if(!is.na(major_cluster)) {
    if(major_cluster == "Endocrine") {
      node_colors[i] <- "#FF6B6B"  # Bright red for endocrine
    } else if(major_cluster == "Epithelial") {
      node_colors[i] <- "#FFFFB3"
    } else if(major_cluster == "Immune") {
      node_colors[i] <- "#8DD3C7"
    } else if(major_cluster == "Stromal") {
      node_colors[i] <- "#80B1D3"
    } else if(major_cluster == "Endothelial") {
      node_colors[i] <- "#B3DE69"
    }
  }
}

V(g_full)$color <- node_colors

# Size nodes by degree, with endocrine nodes larger
node_sizes <- 3 + 8 * (degree(g_full) / max(degree(g_full)))
is_endocrine_node <- node_colors == "#FF6B6B"
node_sizes[is_endocrine_node] <- node_sizes[is_endocrine_node] * 1.5

V(g_full)$size <- node_sizes

# Create the enhanced plot
pdf("/scratch/rli/project/agent/covarnet/network_global_endocrine_focus.pdf", width=16, height=14)
par(mar=c(2,2,4,2))

# Use force-directed layout
set.seed(42)
layout_endocrine <- layout_with_fr(g_full)

# Plot
plot(g_full,
     layout = layout_endocrine,
     vertex.label = ifelse(is_endocrine_node | degree(g_full) > 20, 
                           V(g_full)$name, ""),  # Label only endocrine and highly connected
     vertex.label.cex = 0.6,
     vertex.label.color = "black",
     vertex.label.dist = 0.3,
     edge.color = edge_colors,
     edge.width = edge_widths,
     main = "Global Network: Endocrine Cell Relationships",
     sub = "Red edges: endocrine connections (darker = stronger correlation) | Node size: degree centrality"
)

# Add legends
# Correlation strength legend for endocrine edges
if(sum(is_endocrine_edge) > 0) {
  correlation_breaks <- seq(min(endocrine_correlations), max(endocrine_correlations), length.out = 5)
  legend_colors <- colorRampPalette(c("#FFB6C1", "#FF1493", "#8B0000"))(4)
  legend("topright",
         legend = sprintf("%.2f - %.2f", correlation_breaks[1:4], correlation_breaks[2:5]),
         col = legend_colors,
         lwd = 3,
         title = "Endocrine Edge\nCorrelation Strength",
         cex = 0.8,
         bty = "n")
}

# Cell type legend
cluster_colors <- c(
  "Endocrine" = "#FF6B6B",
  "Epithelial" = "#FFFFB3",
  "Immune" = "#8DD3C7",
  "Stromal" = "#80B1D3",
  "Endothelial" = "#B3DE69",
  "Other" = "#BEBADA"
)

legend("topleft",
       legend = names(cluster_colors),
       fill = cluster_colors,
       title = "Cell Type Groups",
       cex = 0.8,
       bty = "n")

# Add summary statistics
endocrine_stats <- paste(
  "Endocrine connections:", sum(is_endocrine_edge),
  "\nMean correlation:", round(mean(endocrine_correlations), 3),
  "\nEndocrine nodes:", sum(is_endocrine_node)
)
text(x = -1.3, y = -1.3, labels = endocrine_stats, cex = 0.7, adj = 0)

dev.off()
cat("  ✓ Saved: network_global_endocrine_focus.pdf\n\n")

# ============================================================================
# PART 2: Create Annotated Community Detection Plot
# ============================================================================
cat("Creating annotated community detection visualization...\n")

# Apply community detection
comm_louvain <- cluster_louvain(g_full)
modularity_score <- modularity(comm_louvain)

cat("  Communities found:", length(comm_louvain), "\n")
cat("  Modularity score:", round(modularity_score, 3), "\n")

# Analyze community composition
community_composition <- list()
for(i in 1:length(comm_louvain)) {
  comm_nodes <- V(g_full)$name[membership(comm_louvain) == i]
  
  # Get major clusters for this community
  major_clusters <- character()
  for(node in comm_nodes) {
    cluster1 <- sig_edges$majorCluster1[sig_edges$subCluster1 == node][1]
    cluster2 <- sig_edges$majorCluster2[sig_edges$subCluster2 == node][1]
    major_cluster <- ifelse(!is.na(cluster1), cluster1, cluster2)
    if(!is.na(major_cluster)) major_clusters <- c(major_clusters, major_cluster)
  }
  
  # Count major cluster frequencies
  cluster_table <- table(major_clusters)
  dominant_cluster <- names(cluster_table)[which.max(cluster_table)]
  
  # Check for endocrine cells
  has_endocrine <- "Endocrine" %in% major_clusters
  endocrine_count <- sum(major_clusters == "Endocrine", na.rm = TRUE)
  
  community_composition[[i]] <- list(
    size = length(comm_nodes),
    dominant = dominant_cluster,
    has_endocrine = has_endocrine,
    endocrine_count = endocrine_count,
    diversity = length(unique(major_clusters)),
    nodes = comm_nodes[1:min(3, length(comm_nodes))]  # Sample nodes for annotation
  )
}

# Create annotated community plot
pdf("/scratch/rli/project/agent/covarnet/network_communities_annotated.pdf", width=16, height=14)
par(mar=c(3,2,4,2))

# Use force-directed layout
set.seed(42)
layout_comm <- layout_with_fr(g_full)

# Color communities, highlighting those with endocrine cells
n_communities <- length(comm_louvain)
community_colors <- rep("gray90", n_communities)

for(i in 1:n_communities) {
  if(community_composition[[i]]$has_endocrine) {
    # Red gradient based on endocrine proportion
    endocrine_prop <- community_composition[[i]]$endocrine_count / community_composition[[i]]$size
    community_colors[i] <- adjustcolor(
      rgb(1, 1-endocrine_prop*0.7, 1-endocrine_prop*0.7),
      alpha = 0.3
    )
  } else {
    # Use default colors for non-endocrine communities
    community_colors[i] <- adjustcolor(
      brewer.pal(12, "Set3")[i %% 12 + 1],
      alpha = 0.2
    )
  }
}

# Plot with communities
plot(comm_louvain, g_full,
     layout = layout_comm,
     vertex.label = NA,  # Will add selective labels
     vertex.size = 2 + 6 * (degree(g_full) / max(degree(g_full))),
     vertex.color = node_colors,
     edge.width = 0.2,
     edge.color = adjustcolor("gray50", alpha=0.2),
     mark.border = adjustcolor("gray30", alpha=0.5),
     mark.col = community_colors,
     main = "Annotated Network Communities with Endocrine Highlights",
     sub = paste("Modularity:", round(modularity_score, 3), 
                "| Communities:", n_communities,
                "| Red shading: communities with endocrine cells")
)

# Add community annotations
for(i in 1:min(10, n_communities)) {  # Annotate top 10 communities by size
  if(community_composition[[i]]$size > 3) {  # Only annotate larger communities
    # Find center of community
    comm_nodes_idx <- which(membership(comm_louvain) == i)
    comm_layout <- layout_comm[comm_nodes_idx, , drop = FALSE]
    center_x <- mean(comm_layout[,1])
    center_y <- mean(comm_layout[,2])
    
    # Create annotation text
    annotation <- paste0(
      "C", i, "\n",
      "n=", community_composition[[i]]$size, "\n",
      community_composition[[i]]$dominant,
      ifelse(community_composition[[i]]$has_endocrine,
             paste0("\nEndo:", community_composition[[i]]$endocrine_count),
             "")
    )
    
    # Add background box for better readability
    text(center_x, center_y, annotation,
         cex = 0.6,
         col = ifelse(community_composition[[i]]$has_endocrine, "darkred", "darkblue"),
         font = 2)
  }
}

# Add detailed legend
legend_text <- character()
legend_colors <- character()
for(i in 1:min(8, n_communities)) {
  comp <- community_composition[[i]]
  legend_text[i] <- paste0(
    "C", i, " (n=", comp$size, "): ",
    comp$dominant,
    ifelse(comp$has_endocrine, paste0(" [E:", comp$endocrine_count, "]"), "")
  )
  legend_colors[i] <- ifelse(comp$has_endocrine, 
                             adjustcolor("red", alpha=0.3),
                             adjustcolor("blue", alpha=0.3))
}

legend("topright",
       legend = legend_text,
       fill = legend_colors,
       title = "Communities (E: Endocrine count)",
       cex = 0.7,
       bty = "n",
       ncol = 1)

# Add statistics box
stats_text <- paste(
  "Total communities:", n_communities,
  "\nCommunities with endocrine:", sum(sapply(community_composition, function(x) x$has_endocrine)),
  "\nTotal endocrine nodes:", sum(is_endocrine_node),
  "\nLargest community:", max(sapply(community_composition, function(x) x$size)), "nodes",
  "\nMost diverse:", max(sapply(community_composition, function(x) x$diversity)), "cell types"
)
text(x = min(layout_comm[,1]), y = min(layout_comm[,2]), 
     labels = stats_text, cex = 0.6, adj = c(0, 0),
     bg = "white", col = "black")

dev.off()
cat("  ✓ Saved: network_communities_annotated.pdf\n\n")

# ============================================================================
# PART 3: Create Endocrine-Specific Sub-network
# ============================================================================
cat("Creating endocrine-specific sub-network...\n")

# Get all nodes connected to endocrine cells
endocrine_nodes <- unique(c(
  endocrine_edges$subCluster1[endocrine_edges$majorCluster1 == "Endocrine"],
  endocrine_edges$subCluster2[endocrine_edges$majorCluster2 == "Endocrine"]
))

# Get their neighbors
endocrine_neighbors <- unique(c(
  endocrine_edges$subCluster1[endocrine_edges$subCluster2 %in% endocrine_nodes],
  endocrine_edges$subCluster2[endocrine_edges$subCluster1 %in% endocrine_nodes]
))

all_relevant_nodes <- unique(c(endocrine_nodes, endocrine_neighbors))

# Create subgraph
g_endocrine <- induced_subgraph(g_full, all_relevant_nodes)

# Update node colors
endo_node_colors <- rep("#BEBADA", length(V(g_endocrine)))
for(i in seq_along(V(g_endocrine)$name)) {
  if(V(g_endocrine)$name[i] %in% endocrine_nodes) {
    endo_node_colors[i] <- "#FF6B6B"  # Red for endocrine
  }
}
V(g_endocrine)$color <- endo_node_colors

# Create plot
pdf("/scratch/rli/project/agent/covarnet/network_endocrine_subnetwork.pdf", width=14, height=12)
par(mar=c(2,2,4,2))

set.seed(42)
layout_endo <- layout_with_kk(g_endocrine)  # Use Kamada-Kawai for better separation

plot(g_endocrine,
     layout = layout_endo,
     vertex.label = V(g_endocrine)$name,
     vertex.label.cex = 0.7,
     vertex.label.color = "black",
     vertex.size = 8 + 10 * (degree(g_endocrine) / max(degree(g_endocrine))),
     edge.width = 1,
     edge.color = adjustcolor("gray50", alpha=0.5),
     main = "Endocrine Cell Sub-network",
     sub = paste("Endocrine cells (red) and their direct neighbors | ",
                vcount(g_endocrine), "nodes,", ecount(g_endocrine), "edges")
)

# Add legend
legend("topright",
       legend = c("Endocrine cells", "Connected cells"),
       fill = c("#FF6B6B", "#BEBADA"),
       cex = 0.9,
       bty = "n")

dev.off()
cat("  ✓ Saved: network_endocrine_subnetwork.pdf\n\n")

# ============================================================================
# Summary
# ============================================================================
cat("================================================================================\n")
cat("Endocrine-Focused Network Analysis Summary:\n")
cat("================================================================================\n")
cat("Generated visualizations:\n")
cat("  • network_global_endocrine_focus.pdf - Global network with endocrine relationship strength\n")
cat("  • network_communities_annotated.pdf - Annotated communities with endocrine highlights\n")
cat("  • network_endocrine_subnetwork.pdf - Endocrine cells and direct neighbors\n\n")

cat("Endocrine network statistics:\n")
cat("  • Endocrine nodes:", sum(is_endocrine_node), "\n")
cat("  • Edges involving endocrine cells:", sum(is_endocrine_edge), "\n")
cat("  • Communities with endocrine cells:", sum(sapply(community_composition, function(x) x$has_endocrine)), "\n")
cat("  • Mean correlation strength (endocrine edges):", round(mean(endocrine_correlations), 3), "\n")

cat("\n✅ All endocrine-focused visualizations created successfully!\n")