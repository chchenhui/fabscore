import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
from pathlib import Path

# Set style for academic papers
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Create figures directory
figures_dir = Path("/path/to/project/LLM-Surveying-LLMs-An-Agentic-Pipeline-for-Autonomous-Scientific-Paper-Surveying/paper/figures")
figures_dir.mkdir(exist_ok=True)

# Figure 1: Comparison of Topics Processed
topics_data = {
    'Topics': ['RLHF\nAlignment', 'Instruction\nTuning', 'In-context\nLearning', 
               'Synthetic\nData', 'Multimodal\nRL', 'LLM\nAgents'],
    'Papers': [443, 100, 100, 100, 75, 100],
    'Clusters': [13, 10, 8, 8, 8, 9]
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Papers bar chart
x_pos = np.arange(len(topics_data['Topics']))
colors = ['#FF6B6B' if p > 400 else '#4ECDC4' if p == 100 else '#95E77E' for p in topics_data['Papers']]
bars1 = ax1.bar(x_pos, topics_data['Papers'], color=colors, alpha=0.8)
ax1.set_xlabel('Research Topic', fontsize=12)
ax1.set_ylabel('Number of Papers Retrieved', fontsize=12)
ax1.set_title('(a) Papers Retrieved per Topic', fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(topics_data['Topics'], rotation=0)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom', fontsize=10)

# Clusters bar chart
bars2 = ax2.bar(x_pos, topics_data['Clusters'], color='#6C5CE7', alpha=0.8)
ax2.set_xlabel('Research Topic', fontsize=12)
ax2.set_ylabel('Number of Clusters Generated', fontsize=12)
ax2.set_title('(b) Clusters Generated per Topic', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(topics_data['Topics'], rotation=0)
ax2.set_ylim(0, 15)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(figures_dir / 'topic_statistics.pdf', dpi=300, bbox_inches='tight')
plt.savefig(figures_dir / 'topic_statistics.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Cluster Distribution for LLM Agents and Synthetic Data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# LLM Agents cluster distribution
llm_agents_clusters = {
    'Medical & Healthcare': 8,
    'Planning & Task': 15,
    'Evaluation': 17,
    'Frameworks': 15,
    'Vision & Multimodal': 8,
    'Domain-Specific': 6,
    'Reasoning': 16,
    'Safety': 14,
    'Accessibility': 1
}

# Pie chart for LLM Agents
colors1 = plt.cm.Set3(np.linspace(0, 1, len(llm_agents_clusters)))
wedges, texts, autotexts = ax1.pie(llm_agents_clusters.values(), 
                                    labels=None,  # We'll add custom labels
                                    colors=colors1,
                                    autopct='%1.0f%%',
                                    startangle=90)
ax1.set_title('(a) LLM Agents Paper Distribution\n(100 papers, 9 clusters)', 
              fontsize=13, fontweight='bold', pad=20)

# Add legend with full labels
ax1.legend(wedges, llm_agents_clusters.keys(), 
          title="Clusters", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
          fontsize=9)

# Synthetic Data cluster distribution
synthetic_clusters = {
    'Text Classification': 22,
    'Medical & Clinical': 20,
    'Tabular Data': 15,
    'Language Generation': 14,
    'Multimodal': 10,
    'Math Reasoning': 9,
    'Privacy-Preserving': 8,
    'Other': 2
}

# Pie chart for Synthetic Data
colors2 = plt.cm.Set2(np.linspace(0, 1, len(synthetic_clusters)))
wedges2, texts2, autotexts2 = ax2.pie(synthetic_clusters.values(),
                                       labels=None,
                                       colors=colors2,
                                       autopct='%1.0f%%',
                                       startangle=45)
ax2.set_title('(b) Synthetic Data Paper Distribution\n(100 papers, 8 clusters)', 
              fontsize=13, fontweight='bold', pad=20)

# Add legend
ax2.legend(wedges2, synthetic_clusters.keys(),
          title="Clusters", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
          fontsize=9)

plt.tight_layout()
plt.savefig(figures_dir / 'cluster_distribution.pdf', dpi=300, bbox_inches='tight')
plt.savefig(figures_dir / 'cluster_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Processing Pipeline Timeline
fig, ax = plt.subplots(figsize=(12, 6))

stages = ['Paper\nSearch', 'API\nRetrieval', 'Clustering', 'Survey\nWriting', 'Evaluation']
times = [2, 3, 3, 10, 2]  # Minutes for each stage
cumulative = np.cumsum([0] + times[:-1])

colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']

# Create horizontal bar chart
bars = ax.barh(range(len(stages)), times, left=cumulative, color=colors, alpha=0.8, height=0.6)

# Add stage labels on y-axis
ax.set_yticks(range(len(stages)))
ax.set_yticklabels(stages, fontsize=11)

# Clean up x-axis with better tick positions
ax.set_xlabel('Time (minutes)', fontsize=12)
ax.set_xticks(range(0, 22, 2))  # Set ticks every 2 minutes from 0 to 20
ax.set_xticklabels([str(i) for i in range(0, 22, 2)], fontsize=10)

ax.set_title('Processing Pipeline Timeline for Survey Generation', fontsize=14, fontweight='bold', pad=20)

# Add time labels on bars
for i, (bar, time) in enumerate(zip(bars, times)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2,
            f'{time} min', ha='center', va='center', fontweight='bold', color='white', fontsize=10)

# Add total time annotation at the bottom
total_time = sum(times)
ax.text(total_time/2, -1.2, f'Total Processing Time: {total_time} minutes', 
        ha='center', fontsize=12, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))

ax.set_xlim(-0.5, total_time + 1)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(figures_dir / 'pipeline_timeline.pdf', dpi=300, bbox_inches='tight')
plt.savefig(figures_dir / 'pipeline_timeline.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 4: System Architecture Components
fig, ax = plt.subplots(figsize=(10, 8))

# Component sizes (representing complexity/importance)
components = {
    'Paper Search\nSpecialist': 100,
    'Topic Mining\nAgent': 85,
    'Survey Writer\nAgent': 120,
    'Quality\nEvaluator': 90,
}

# Create bubble chart
np.random.seed(42)
x = np.array([1, 2, 1, 2])
y = np.array([2, 2, 1, 1])
sizes = list(components.values())
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

scatter = ax.scatter(x, y, s=[s*20 for s in sizes], c=colors, alpha=0.6, edgecolors='black', linewidth=2)

# Add labels
for i, (label, size) in enumerate(components.items()):
    ax.annotate(label, (x[i], y[i]), ha='center', va='center', fontsize=11, fontweight='bold')

ax.set_xlim(0.5, 2.5)
ax.set_ylim(0.5, 2.5)
ax.set_title('Agentic Auto-Survey System Components', fontsize=14, fontweight='bold', pad=20)
ax.axis('off')

# Add arrows to show flow
arrow_props = dict(arrowstyle='->', lw=2, color='gray', alpha=0.5)
ax.annotate('', xy=(2, 1.8), xytext=(1, 1.8), arrowprops=arrow_props)  # Search to Mining
ax.annotate('', xy=(1.2, 1), xytext=(1.8, 1.8), arrowprops=arrow_props)  # Mining to Writer
ax.annotate('', xy=(2, 1.2), xytext=(1.2, 1.2), arrowprops=arrow_props)  # Writer to Evaluator

# Add legend for component roles
legend_text = [
    'Search: Query expansion & API integration',
    'Mining: K-means clustering & embeddings',
    'Writer: Synthesis & citation management',
    'Evaluator: 12-dimensional assessment'
]
for i, text in enumerate(legend_text):
    ax.text(1.5, 0.3 - i*0.1, text, fontsize=9, ha='center', style='italic')

plt.tight_layout()
plt.savefig(figures_dir / 'system_architecture.pdf', dpi=300, bbox_inches='tight')
plt.savefig(figures_dir / 'system_architecture.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Figures generated successfully in {figures_dir}")
print("Generated files:")
print("- topic_statistics.pdf/png")
print("- cluster_distribution.pdf/png")
print("- pipeline_timeline.pdf/png")
print("- system_architecture.pdf/png")