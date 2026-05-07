#!/usr/bin/env Rscript

# Create improved network visualizations with grouping and sub-networks
# This supplements the existing global_enhanced_v1 and v2 plots

.libPaths("/wanglab/rli/miniforge3/envs/r412/lib/R/library")

suppressPackageStartupMessages({
  library(CoVarNet)
  library(igraph)
  library(dplyr)
  library(RColorBrewer)
})

cat("================================================================================\n")
cat("Creating Improved Network Visualizations\n")
cat("================================================================================\n\n")

# Load the saved network data
cat("Loading network data...\n")
network <- readRDS("/scratch/rli/project/agent/covarnet/network_K12.rds")
cor_pair <- readRDS("/scratch/rli/project/agent/covarnet/cor_pair_K12.rds")

# Check if network$global has the expected structure
if(is.null(network$global) || !("edge" %in% names(network$global))) {
  stop("network$global does not have the expected structure")
}

cat("  Found", nrow(network$global$edge), "total edges\n")
cat("  Found", nrow(network$global$node), "nodes\n\n")

# Filter for significant edges
sig_edges <- network$global$edge[network$global$edge$pval_fdr < 0.05, ]
cat("  Using", nrow(sig_edges), "significant edges (FDR < 0.05)\n\n")

# ============================================================================
# PART 1: Create Simplified Global Network (Major Groups Only)
# ============================================================================
cat("Creating simplified global network with major cell type groups...\n")

# Get unique major clusters from the edges
major_clusters <- unique(c(sig_edges$majorCluster1, sig_edges$majorCluster2))
cat("  Major clusters found:", paste(major_clusters, collapse=", "), "\n")

# Aggregate edges by major cluster pairs
major_edges <- sig_edges %>%
  group_by(majorCluster1, majorCluster2) %>%
  summarise(
    n_connections = n(),
    mean_correlation = mean(abs(correlation), na.rm = TRUE),
    min_pval = min(pval_fdr, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  filter(majorCluster1 != majorCluster2)  # Remove self-loops for clarity

cat("  Aggregated to", nrow(major_edges), "major cluster connections\n")

# Create igraph object for major clusters
g_major <- graph_from_data_frame(
  d = major_edges[, c("majorCluster1", "majorCluster2")],
  directed = FALSE
)

# Add edge attributes
E(g_major)$weight <- major_edges$n_connections
E(g_major)$correlation <- major_edges$mean_correlation

# Add node colors
cluster_colors <- c(
  "Immune" = "#8DD3C7",
  "Epithelial" = "#FFFFB3",
  "Endocrine" = "#FB8072",
  "Stromal" = "#80B1D3",
  "Other" = "#BEBADA"
)

V(g_major)$color <- cluster_colors[V(g_major)$name]
V(g_major)$color[is.na(V(g_major)$color)] <- "#BEBADA"  # Default color

# Calculate node sizes based on degree
V(g_major)$size <- 20 + 5 * degree(g_major)

# Create simplified global plot
pdf("/scratch/rli/project/agent/covarnet/network_global_simplified.pdf", width=12, height=10)
par(mar=c(1,1,3,1))

# Use force-directed layout
set.seed(42)
layout_major <- layout_with_fr(g_major)

# Scale edge widths
edge_widths <- 0.5 + 4 * (E(g_major)$weight / max(E(g_major)$weight))

plot(g_major,
     layout = layout_major,
     vertex.label = V(g_major)$name,
     vertex.label.cex = 1.2,
     vertex.label.color = "black",
     vertex.label.font = 2,
     edge.width = edge_widths,
     edge.color = adjustcolor("gray40", alpha=0.6),
     main = "Simplified Global Network - Major Cell Type Groups",
     sub = "Edge width represents number of connections between groups"
)

# Add legend
legend("topright",
       legend = names(cluster_colors),
       fill = cluster_colors,
       title = "Cell Type Groups",
       cex = 0.9,
       bty = "n")

dev.off()
cat("  ✓ Saved: network_global_simplified.pdf\n\n")

# ============================================================================
# PART 2: Create Top N Most Connected Nodes Network
# ============================================================================
cat("Creating network with top N most connected nodes...\n")

# Create full igraph from significant edges
g_full <- graph_from_data_frame(
  d = sig_edges[, c("subCluster1", "subCluster2")],
  directed = FALSE
)

# Calculate node metrics
node_degrees <- degree(g_full)
node_betweenness <- betweenness(g_full)

# Get top N nodes by degree
N <- 30  # Show top 30 most connected nodes
top_nodes <- names(sort(node_degrees, decreasing = TRUE)[1:min(N, length(node_degrees))])
cat("  Selected top", length(top_nodes), "most connected nodes\n")

# Create subgraph with only top nodes
g_top <- induced_subgraph(g_full, top_nodes)

# Add node attributes
# Get major cluster for each node
node_major_cluster <- character(length(V(g_top)))
for(i in seq_along(V(g_top))) {
  node_name <- V(g_top)$name[i]
  # Find major cluster from edges
  cluster1 <- sig_edges$majorCluster1[sig_edges$subCluster1 == node_name][1]
  cluster2 <- sig_edges$majorCluster2[sig_edges$subCluster2 == node_name][1]
  node_major_cluster[i] <- ifelse(!is.na(cluster1), cluster1, cluster2)
}

V(g_top)$color <- cluster_colors[node_major_cluster]
V(g_top)$color[is.na(V(g_top)$color)] <- "#BEBADA"

# Size by degree
V(g_top)$size <- 8 + 15 * (degree(g_top) / max(degree(g_top)))

# Create top nodes network plot
pdf("/scratch/rli/project/agent/covarnet/network_top_connected.pdf", width=14, height=12)
par(mar=c(1,1,3,1))

# Use force-directed layout with community detection
set.seed(42)
communities <- cluster_fast_greedy(g_top)
layout_top <- layout_with_fr(g_top)

# Plot with community coloring in background
plot(communities, g_top,
     layout = layout_top,
     vertex.label = V(g_top)$name,
     vertex.label.cex = 0.7,
     vertex.label.color = "black",
     vertex.label.dist = 0.5,
     edge.width = 1,
     edge.color = adjustcolor("gray50", alpha=0.5),
     mark.border = NA,
     mark.col = adjustcolor(rainbow(length(communities)), alpha=0.1),
     main = "Top 30 Most Connected Cell Types",
     sub = "Communities detected using fast greedy algorithm"
)

# Add degree information
text(x = -1.3, y = 1.3, 
     labels = paste("Node size ~ Degree\nColor ~ Major cluster"),
     cex = 0.8, adj = 0)

dev.off()
cat("  ✓ Saved: network_top_connected.pdf\n\n")

# ============================================================================
# PART 3: Create Sub-networks for Each Major Cell Type Category
# ============================================================================
cat("Creating sub-networks for each major cell type category...\n")

# Create a multi-panel plot with sub-networks
pdf("/scratch/rli/project/agent/covarnet/network_subgroups.pdf", width=16, height=12)

# Set up multi-panel layout (2x3 for up to 6 groups)
n_groups <- length(major_clusters)
n_cols <- min(3, n_groups)
n_rows <- ceiling(n_groups / n_cols)
par(mfrow=c(n_rows, n_cols), mar=c(2,2,3,1))

for(cluster in major_clusters) {
  # Get edges involving this cluster
  cluster_edges <- sig_edges[
    sig_edges$majorCluster1 == cluster | sig_edges$majorCluster2 == cluster,
  ]
  
  if(nrow(cluster_edges) > 0) {
    # Get unique cell types in this cluster
    cluster_cells <- unique(c(
      cluster_edges$subCluster1[cluster_edges$majorCluster1 == cluster],
      cluster_edges$subCluster2[cluster_edges$majorCluster2 == cluster]
    ))
    
    # Filter edges to only those within or connected to this cluster
    sub_edges <- cluster_edges[
      cluster_edges$subCluster1 %in% cluster_cells | 
      cluster_edges$subCluster2 %in% cluster_cells,
    ]
    
    if(nrow(sub_edges) > 2) {
      # Create subgraph
      g_sub <- graph_from_data_frame(
        d = sub_edges[, c("subCluster1", "subCluster2")],
        directed = FALSE
      )
      
      # Color nodes by whether they're in this cluster or connected to it
      node_colors <- rep("lightgray", length(V(g_sub)))
      node_colors[V(g_sub)$name %in% cluster_cells] <- cluster_colors[cluster]
      if(is.na(cluster_colors[cluster])) {
        node_colors[V(g_sub)$name %in% cluster_cells] <- "#BEBADA"
      }
      V(g_sub)$color <- node_colors
      
      # Size by degree
      V(g_sub)$size <- 5 + 10 * (degree(g_sub) / max(degree(g_sub), 1))
      
      # Plot
      set.seed(42)
      plot(g_sub,
           layout = layout_with_fr(g_sub),
           vertex.label = V(g_sub)$name,
           vertex.label.cex = 0.5,
           vertex.label.color = "black",
           edge.width = 0.5,
           edge.color = adjustcolor("gray50", alpha=0.5),
           main = paste(cluster, "Network"),
           sub = paste(length(cluster_cells), "cell types,", 
                      ecount(g_sub), "connections")
      )
    } else {
      # Empty plot for groups with too few edges
      plot.new()
      text(0.5, 0.5, paste(cluster, "\n(insufficient connections)"), cex=1.2)
    }
  } else {
    # Empty plot for groups with no edges
    plot.new()
    text(0.5, 0.5, paste(cluster, "\n(no connections)"), cex=1.2)
  }
}

dev.off()
cat("  ✓ Saved: network_subgroups.pdf\n\n")

# ============================================================================
# PART 4: Create Enhanced Community Detection Plot
# ============================================================================
cat("Creating enhanced community detection visualization...\n")

# Use the full graph for community detection
g_community <- g_full

# Apply multiple community detection algorithms
cat("  Running community detection algorithms...\n")
comm_louvain <- cluster_louvain(g_community)
comm_walktrap <- cluster_walktrap(g_community)
modularity_score <- modularity(comm_louvain)

cat("  Louvain communities found:", length(comm_louvain), "\n")
cat("  Modularity score:", round(modularity_score, 3), "\n")

# Create community plot
pdf("/scratch/rli/project/agent/covarnet/network_communities.pdf", width=14, height=12)
par(mar=c(2,2,3,2))

# Use force-directed layout
set.seed(42)
layout_comm <- layout_with_fr(g_community)

# Assign colors to communities
n_communities <- length(comm_louvain)
if(n_communities <= 12) {
  community_colors <- brewer.pal(max(3, n_communities), "Set3")
} else {
  community_colors <- rainbow(n_communities)
}

# Plot with communities
plot(comm_louvain, g_community,
     layout = layout_comm,
     vertex.label = NA,  # No labels for clarity
     vertex.size = 3 + 5 * (degree(g_community) / max(degree(g_community))),
     edge.width = 0.3,
     edge.color = adjustcolor("gray50", alpha=0.3),
     mark.border = adjustcolor("gray30", alpha=0.5),
     mark.col = adjustcolor(community_colors, alpha=0.2),
     main = "Network Communities (Louvain Algorithm)",
     sub = paste("Modularity:", round(modularity_score, 3), 
                "| Communities:", n_communities,
                "| Nodes:", vcount(g_community),
                "| Edges:", ecount(g_community))
)

# Add legend showing community sizes
top_communities <- sort(sizes(comm_louvain), decreasing = TRUE)[1:min(10, n_communities)]
legend("topright",
       legend = paste("Community", names(top_communities), 
                     "(n=", top_communities, ")", sep=" "),
       fill = community_colors[as.numeric(names(top_communities))],
       title = "Top Communities by Size",
       cex = 0.7,
       bty = "n")

dev.off()
cat("  ✓ Saved: network_communities.pdf\n\n")

# ============================================================================
# Summary
# ============================================================================
cat("================================================================================\n")
cat("Network Visualization Summary:\n")
cat("================================================================================\n")
cat("Original plots (preserved):\n")
cat("  • network_global_enhanced_v1.pdf - Original force-directed layout\n")
cat("  • network_global_enhanced_v2.pdf - Original circular layout\n\n")
cat("New improved plots:\n")
cat("  • network_global_simplified.pdf - Major cell type groups only\n")
cat("  • network_top_connected.pdf - Top 30 most connected nodes\n")
cat("  • network_subgroups.pdf - Sub-networks for each major category\n")
cat("  • network_communities.pdf - Community detection visualization\n\n")
cat("Network statistics:\n")
cat("  • Total nodes:", vcount(g_full), "\n")
cat("  • Total edges:", ecount(g_full), "\n")
cat("  • Major groups:", length(major_clusters), "\n")
cat("  • Communities detected:", n_communities, "\n")
cat("  • Modularity score:", round(modularity_score, 3), "\n")
cat("\n✅ All improved network visualizations created successfully!\n")