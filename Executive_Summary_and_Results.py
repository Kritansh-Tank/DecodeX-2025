import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('colorblind')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Load results if they exist, or use example data
results_file = 'part_mapping_results.csv'

if os.path.exists(results_file):
    mapping_results = pd.read_csv(results_file)
    print(f"Loaded {len(mapping_results)} mapping results from file.")
else:
    # Create sample results for the visualization (representative of actual results)
    print("Results file not found. Using sample data for visualization.")
    sample_data = {
        'match_type': ['Exact', 'Fuzzy', 'TF-IDF'] * 50,
        'confidence_score': np.concatenate([
            np.ones(50) * 100,  # Exact matches always 100%
            np.random.normal(85, 5, 50),  # Fuzzy matches around 85%
            np.random.normal(65, 10, 50)  # TF-IDF matches more varied
        ]),
        'surveyor_part': ['Front Bumper', 'Rear Door', 'Headlight Assembly'] * 50,
        'best_garage_match': ['FRONT BUMPER ASSEMBLY', 'DOOR REAR', 'HEADLAMP UNIT'] * 50
    }
    mapping_results = pd.DataFrame(sample_data)
    
    # Cap confidence scores at 100
    mapping_results['confidence_score'] = mapping_results['confidence_score'].clip(upper=100)

# Create output directory for figures
os.makedirs('figures', exist_ok=True)

# ===============================================================
# Executive Summary Visualizations
# ===============================================================

# 1. Match Type Distribution
plt.figure(figsize=(10, 6))
match_counts = mapping_results['match_type'].value_counts()
match_percentages = match_counts / match_counts.sum() * 100

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
ax = sns.barplot(x=match_percentages.index, y=match_percentages.values, palette=colors)
plt.title('Distribution of Matching Techniques', fontsize=18, fontweight='bold')
plt.xlabel('Matching Technique', fontsize=14)
plt.ylabel('Percentage (%)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add percentage labels on top of bars
for i, p in enumerate(ax.patches):
    ax.annotate(f'{p.get_height():.1f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/matching_technique_distribution.png', dpi=300, bbox_inches='tight')

# 2. Confidence Score Distribution by Match Type
plt.figure(figsize=(12, 7))
sns.boxplot(x='match_type', y='confidence_score', data=mapping_results, palette=colors, width=0.5)
sns.stripplot(x='match_type', y='confidence_score', data=mapping_results.sample(min(100, len(mapping_results))), 
              size=5, color='black', alpha=0.5, jitter=True)

plt.title('Confidence Score Distribution by Match Type', fontsize=18, fontweight='bold')
plt.xlabel('Match Type', fontsize=14)
plt.ylabel('Confidence Score (%)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('figures/confidence_distribution.png', dpi=300, bbox_inches='tight')

# 3. Match Success Rate Visualization (Pie Chart)
total_parts = len(mapping_results)
matched_parts = mapping_results['best_garage_match'].notna().sum()
unmatched_parts = total_parts - matched_parts
match_rate = matched_parts / total_parts * 100

plt.figure(figsize=(8, 8))
plt.pie([matched_parts, unmatched_parts], 
        labels=['Matched', 'Unmatched'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['#2ca02c', '#d62728'],
        explode=(0.1, 0),
        shadow=True,
        textprops={'fontsize': 14})
plt.title('Part Mapping Success Rate', fontsize=18, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.savefig('figures/matching_success_rate.png', dpi=300, bbox_inches='tight')

# 4. Example Mapping Visualization
plt.figure(figsize=(14, 8))

# Select some example mappings (best matches from each type)
if len(mapping_results) >= 6:
    examples = pd.concat([
        mapping_results[mapping_results['match_type'] == 'Exact'].head(2),
        mapping_results[mapping_results['match_type'] == 'Fuzzy'].sort_values('confidence_score', ascending=False).head(2),
        mapping_results[mapping_results['match_type'] == 'TF-IDF'].sort_values('confidence_score', ascending=False).head(2)
    ])
else:
    examples = mapping_results.head(min(6, len(mapping_results)))

# Prepare the plot
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 1)
ax.set_ylim(0, len(examples) + 1)
ax.set_axis_off()

# Title
plt.suptitle('Examples of Part Mapping Results', fontsize=20, fontweight='bold', y=0.95)

# Header
ax.text(0.01, len(examples) + 0.5, 'Surveyor Part', fontsize=14, fontweight='bold')
ax.text(0.5, len(examples) + 0.5, 'Garage Part', fontsize=14, fontweight='bold')
ax.text(0.85, len(examples) + 0.5, 'Match Type', fontsize=14, fontweight='bold')
ax.text(0.95, len(examples) + 0.5, 'Confidence', fontsize=14, fontweight='bold')

# Separator line
ax.axhline(y=len(examples) + 0.3, color='black', linestyle='-', linewidth=1)

# Content rows
for i, (_, row) in enumerate(examples.iterrows()):
    y_pos = len(examples) - i
    
    # Match type color
    if row['match_type'] == 'Exact':
        color = '#1f77b4'
    elif row['match_type'] == 'Fuzzy':
        color = '#ff7f0e'
    else:
        color = '#2ca02c'
    
    # Draw row with colored background based on match type
    rect = plt.Rectangle((0, y_pos - 0.4), 1, 0.8, facecolor=color, alpha=0.1)
    ax.add_patch(rect)
    
    # Text
    ax.text(0.01, y_pos, str(row['surveyor_part']), fontsize=12, va='center')
    ax.text(0.5, y_pos, str(row['best_garage_match']), fontsize=12, va='center')
    ax.text(0.85, y_pos, row['match_type'], fontsize=12, va='center')
    ax.text(0.95, y_pos, f"{row['confidence_score']:.1f}%", fontsize=12, va='center')
    
    # Separator line
    if i < len(examples) - 1:
        ax.axhline(y=y_pos - 0.4, color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('figures/example_mappings.png', dpi=300, bbox_inches='tight')

# 5. Create Executive Summary Visualization
plt.figure(figsize=(16, 10))

# Set up grid for the dashboard
gs = plt.GridSpec(2, 3, height_ratios=[1, 1])

# 1. Title and Summary
ax_title = plt.subplot(gs[0, 0:3])
ax_title.set_axis_off()
ax_title.text(0.5, 0.9, 'AutoSure Insurance', fontsize=30, fontweight='bold', ha='center')
ax_title.text(0.5, 0.7, 'Part Mapping Solution - Executive Summary', fontsize=24, ha='center')

# Key Metrics
ax_title.text(0.1, 0.4, f'Total Parts Analyzed:', fontsize=14)
ax_title.text(0.4, 0.4, f'{total_parts:,}', fontsize=14, fontweight='bold')

ax_title.text(0.6, 0.4, f'Match Success Rate:', fontsize=14)
ax_title.text(0.9, 0.4, f'{match_rate:.1f}%', fontsize=14, fontweight='bold', 
              color='green' if match_rate > 50 else 'orange' if match_rate > 25 else 'red')

# Avg confidence
avg_confidence = mapping_results['confidence_score'].mean()
ax_title.text(0.1, 0.3, f'Average Confidence Score:', fontsize=14)
ax_title.text(0.4, 0.3, f'{avg_confidence:.1f}%', fontsize=14, fontweight='bold')

# Match type breakdown
ax_title.text(0.6, 0.3, f'Top Match Type:', fontsize=14)
top_match = match_counts.index[0]
ax_title.text(0.9, 0.3, f'{top_match} ({match_percentages[top_match]:.1f}%)', fontsize=14, fontweight='bold')

# Recommendations
ax_title.text(0.1, 0.15, 'Key Recommendations:', fontsize=14, fontweight='bold')
ax_title.text(0.1, 0.05, '1. Standardize part nomenclature across surveyor and garage systems', fontsize=12)
ax_title.text(0.6, 0.05, '2. Implement this matching system as an automated verification tool', fontsize=12)

# 2. Success Rate Pie Chart
ax_pie = plt.subplot(gs[1, 0])
ax_pie.pie([matched_parts, unmatched_parts], 
       labels=['Matched', 'Unmatched'],
       autopct='%1.1f%%',
       startangle=90,
       colors=['#2ca02c', '#d62728'],
       explode=(0.1, 0),
       shadow=True,
       textprops={'fontsize': 10})
ax_pie.set_title('Part Mapping Success Rate', fontsize=14, fontweight='bold')
ax_pie.axis('equal')

# 3. Match Type Distribution
ax_bar = plt.subplot(gs[1, 1])
sns.barplot(x=match_percentages.index, y=match_percentages.values, palette=colors, ax=ax_bar)
ax_bar.set_title('Matching Techniques Used', fontsize=14, fontweight='bold')
ax_bar.set_xlabel('')
ax_bar.set_ylabel('Percentage (%)', fontsize=10)
for i, p in enumerate(ax_bar.patches):
    ax_bar.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'bottom', fontsize=10)

# 4. Confidence Score Distribution
ax_box = plt.subplot(gs[1, 2])
sns.boxplot(x='match_type', y='confidence_score', data=mapping_results, palette=colors, ax=ax_box)
ax_box.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
ax_box.set_xlabel('')
ax_box.set_ylabel('Confidence (%)', fontsize=10)

plt.tight_layout()
plt.savefig('figures/executive_summary_dashboard.png', dpi=300, bbox_inches='tight')

print("\nExecutive summary visualizations have been created in the 'figures' directory.")
print("The following files were generated:")
print("  - matching_technique_distribution.png")
print("  - confidence_distribution.png")
print("  - matching_success_rate.png")
print("  - example_mappings.png")
print("  - executive_summary_dashboard.png") 