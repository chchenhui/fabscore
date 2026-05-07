# This script makes graphs from the docking results file.
# Docking results are like a scoreboard of how well proteins fit.

import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('results', exist_ok=True)

# Read the docking results file
df = pd.read_csv('docking_results.csv')

# Make a bar chart of average docking scores
mean = df.groupby('variant')['docking_score'].mean().reindex(['Wild-Type','Electrostatic','Hydrophobic'])
plt.figure(figsize=(6,4))
mean.plot(kind='bar', color=['purple','blue','red'])
plt.ylabel('Docking Score (lower = better)')
plt.title('Average Docking Scores by Variant')
plt.tight_layout()
plt.savefig('results/avg_docking_scores.png', dpi=300)
plt.close()

# Make a boxplot of confidence scores
plt.figure(figsize=(6,4))
df.boxplot(column='confidence', by='variant')
plt.ylabel('Confidence (%)')
plt.title('Confidence by Variant')
plt.suptitle('')
plt.tight_layout()
plt.savefig('results/confidence_boxplot.png', dpi=300)
plt.close()

print('Done! Check the results/ folder for your graphs.')
