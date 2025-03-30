import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('colorblind')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

print("AutoSure Insurance - Damaged Parts Analysis")
print("Loading data...")

# Load the datasets
surveyor_df = pd.read_csv('surveyor_data.csv', low_memory=False)
parts_code_df = pd.read_excel('Primary_Parts_Code.xlsx')

print(f"\nData loaded successfully!")
print(f"Surveyor dataset: {surveyor_df.shape[0]:,} records")
print(f"Parts code master: {parts_code_df.shape[0]} records")

# Create output directory for figures
os.makedirs('analysis_figures', exist_ok=True)

# ===============================================================
# 1. Most Commonly Damaged Parts Analysis
# ===============================================================
print("\n\n" + "="*80)
print(" "*20 + "MOST COMMONLY DAMAGED PARTS ANALYSIS")
print("="*80)

# Count occurrences of each part
part_counts = surveyor_df['TXT_PARTS_NAME'].value_counts()
total_parts = len(surveyor_df)

# Create a DataFrame with part counts and percentages
common_parts_df = pd.DataFrame({
    'Part_Name': part_counts.index,
    'Count': part_counts.values,
    'Percentage': (part_counts.values / total_parts * 100).round(2)
})

# Display the top 20 most common parts
print("\nTop 20 Most Commonly Damaged Parts:")
print(common_parts_df.head(20).to_string(index=False))

# Save to CSV
common_parts_df.to_csv('most_common_parts.csv', index=False)
print(f"\nComplete list saved to 'most_common_parts.csv'")

# Visualize top 10 parts
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Count', y='Part_Name', data=common_parts_df.head(10),
                palette='viridis', hue='Part_Name', legend=False)

# Add count and percentage labels
for i, (count, percentage) in enumerate(zip(common_parts_df.head(10)['Count'], 
                                           common_parts_df.head(10)['Percentage'])):
    ax.text(count + 5, i, f"{count:,} ({percentage}%)", va='center')

plt.title('Top 10 Most Commonly Damaged Parts', fontsize=16, fontweight='bold')
plt.xlabel('Number of Claims', fontsize=12)
plt.ylabel('Part Name', fontsize=12)
plt.tight_layout()
plt.savefig('analysis_figures/top10_damaged_parts.png', dpi=300, bbox_inches='tight')

# ===============================================================
# 2. Distribution of Claims by Primary Parts
# ===============================================================
print("\n\n" + "="*80)
print(" "*20 + "DISTRIBUTION OF CLAIMS BY PRIMARY PARTS")
print("="*80)

# Map part codes to primary parts
print("\nMapping part codes to primary parts categories...")

# Create a dictionary mapping part codes to part names
part_code_map = dict(zip(parts_code_df['Surveyor Part Code'], 
                        parts_code_df['Surveyor Part Name']))

# Map the codes to the surveyor data
surveyor_df['Primary_Part_Name'] = surveyor_df['NUM_PART_CODE'].map(part_code_map)

# Count claims by primary part
primary_part_counts = surveyor_df['Primary_Part_Name'].value_counts()
total_primary_counts = primary_part_counts.sum()

# Create DataFrame with counts and percentages
primary_parts_df = pd.DataFrame({
    'Primary_Part': primary_part_counts.index,
    'Claim_Count': primary_part_counts.values,
    'Percentage': (primary_part_counts.values / total_primary_counts * 100).round(2)
})

# Fill NaN values (parts without mapping) as "Other/Unknown"
primary_parts_df = primary_parts_df.fillna('Other/Unknown')

# Display results
print("\nDistribution of Claims by Primary Parts:")
print(primary_parts_df.to_string(index=False))

# Save to CSV
primary_parts_df.to_csv('primary_parts_distribution.csv', index=False)
print(f"\nComplete distribution saved to 'primary_parts_distribution.csv'")

# Visualize primary parts distribution (top 10)
plt.figure(figsize=(12, 8))
primary_parts_df_sorted = primary_parts_df.sort_values('Claim_Count', ascending=False)
primary_top10 = primary_parts_df_sorted.head(10)

ax = sns.barplot(x='Claim_Count', y='Primary_Part', data=primary_top10,
                palette='mako', hue='Primary_Part', legend=False)

# Add count and percentage labels
for i, (count, percentage) in enumerate(zip(primary_top10['Claim_Count'], 
                                          primary_top10['Percentage'])):
    ax.text(count + 5, i, f"{count:,} ({percentage}%)", va='center')

plt.title('Top 10 Primary Parts by Claim Count', fontsize=16, fontweight='bold')
plt.xlabel('Number of Claims', fontsize=12)
plt.ylabel('Primary Part', fontsize=12)
plt.tight_layout()
plt.savefig('analysis_figures/primary_parts_distribution.png', dpi=300, bbox_inches='tight')

# ===============================================================
# 3. Top 10 Secondary Parts Associated with Each Primary Part
# ===============================================================
print("\n\n" + "="*80)
print(" "*15 + "TOP 10 SECONDARY PARTS ASSOCIATED WITH EACH PRIMARY PART")
print("="*80)

# Function to find secondary parts for a given primary part
def get_secondary_parts(primary_part):
    # Get all claims that involved this primary part
    primary_claims = surveyor_df[surveyor_df['Primary_Part_Name'] == primary_part]['REFERENCE_NUM'].unique()
    
    # Find all parts in these claims
    parts_in_claims = surveyor_df[surveyor_df['REFERENCE_NUM'].isin(primary_claims)]
    
    # Exclude the primary part itself
    secondary_parts = parts_in_claims[
        (parts_in_claims['Primary_Part_Name'] != primary_part) | 
        (parts_in_claims['Primary_Part_Name'].isna())
    ]
    
    # Count occurrences of each secondary part
    sec_part_counts = secondary_parts['TXT_PARTS_NAME'].value_counts()
    total_claims = len(primary_claims)
    
    # Create DataFrame
    if len(sec_part_counts) > 0:
        sec_parts_df = pd.DataFrame({
            'Secondary_Part': sec_part_counts.index,
            'Claim_Count': sec_part_counts.values,
            'Percentage': (sec_part_counts.values / total_claims * 100).round(2)
        })
        return sec_parts_df.head(10)  # Return top 10
    else:
        return pd.DataFrame(columns=['Secondary_Part', 'Claim_Count', 'Percentage'])

# Function to clean filename
def clean_filename(name):
    # Replace invalid filename characters with underscores
    return re.sub(r'[\\/*?:"<>|]', '_', name)

# Get the top 10 primary parts by claim count
top_primary_parts = primary_parts_df_sorted.head(10)['Primary_Part'].tolist()

# Find secondary parts for each primary part
secondary_parts_results = {}

print("\nAnalyzing top secondary parts for each primary part category...")
for primary_part in top_primary_parts:
    if pd.isna(primary_part):
        continue
    
    print(f"Processing: {primary_part}")
    secondary_parts = get_secondary_parts(primary_part)
    secondary_parts_results[primary_part] = secondary_parts
    
    # Save individual result to CSV
    if not secondary_parts.empty:
        safe_name = clean_filename(primary_part)
        file_name = f"secondary_parts_{safe_name}.csv"
        secondary_parts.to_csv(file_name, index=False)
        print(f"  - Saved {len(secondary_parts)} secondary parts to {file_name}")

# Print results for each primary part
print("\nTop 10 Secondary Parts for Each Primary Part Category:")
for primary_part, sec_parts_df in secondary_parts_results.items():
    if not sec_parts_df.empty:
        print(f"\n--- Primary Part: {primary_part} ---")
        print(sec_parts_df.to_string(index=False))
        print("-" * 50)

# Create visualization for a sample of primary parts (top 3)
sample_primaries = top_primary_parts[:3]
for primary_part in sample_primaries:
    if pd.isna(primary_part) or primary_part not in secondary_parts_results:
        continue
        
    sec_parts = secondary_parts_results[primary_part]
    if not sec_parts.empty:
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='Claim_Count', y='Secondary_Part', data=sec_parts,
                        palette='cool', hue='Secondary_Part', legend=False)
        
        # Add count and percentage labels
        for i, (count, percentage) in enumerate(zip(sec_parts['Claim_Count'], 
                                                  sec_parts['Percentage'])):
            ax.text(count + 5, i, f"{count:,} ({percentage}%)", va='center')
        
        safe_name = clean_filename(primary_part)
        plt.title(f'Top 10 Secondary Parts Associated with {primary_part}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Number of Claims', fontsize=12)
        plt.ylabel('Secondary Part', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'analysis_figures/secondary_parts_{safe_name}.png', 
                   dpi=300, bbox_inches='tight')

print("\nAnalysis completed successfully.")
print("Results and visualizations have been saved to CSV files and the 'analysis_figures' directory.") 