import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process
import re
import os
from datetime import datetime
import time

# Record start time
start_time = time.time()

print("AutoSure Insurance - Part Mapping Solution")
print("Loading data...")

# Load the datasets
surveyor_df = pd.read_csv('surveyor_data.csv', low_memory=False)
garage_df = pd.read_csv('garage_data.csv', low_memory=False)
parts_code_df = pd.read_excel('Primary_Parts_Code.xlsx')

print(f"\nData loaded successfully!")
print(f"Surveyor dataset: {surveyor_df.shape[0]:,} records")
print(f"Garage dataset: {garage_df.shape[0]:,} records")
print(f"Parts code master: {parts_code_df.shape[0]} records")

# Data preparation
print("\nPreparing data for analysis...")

# Function to clean text
def clean_text(text):
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Lowercase and remove special characters
    text = text.lower()
    text = re.sub(r'[^\w\s\|\-]', '', text)
    
    # Replace separators with standard format
    text = re.sub(r'[\|\-\/\\]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Clean the part names
print("Cleaning part descriptions...")
surveyor_df['clean_part_name'] = surveyor_df['TXT_PARTS_NAME'].apply(clean_text)
garage_df['clean_part_desc'] = garage_df['PARTDESCRIPTION'].apply(clean_text)

# Display some examples of the cleaning
print("\nExamples of text cleaning:")
print("\nSurveyor Part Names (Original vs Cleaned):")
examples = surveyor_df[['TXT_PARTS_NAME', 'clean_part_name']].head(5)
for i, row in examples.iterrows():
    print(f"Original: {row['TXT_PARTS_NAME']}")
    print(f"Cleaned: {row['clean_part_name']}")
    print("-" * 40)

print("\nGarage Part Descriptions (Original vs Cleaned):")
examples = garage_df[['PARTDESCRIPTION', 'clean_part_desc']].head(5)
for i, row in examples.iterrows():
    print(f"Original: {row['PARTDESCRIPTION']}")
    print(f"Cleaned: {row['clean_part_desc']}")
    print("-" * 40)

# Extract reference claims for analysis
print("\nExtracting a sample of claims for detailed analysis...")

# Get common claims between both datasets
common_refs = set(surveyor_df['REFERENCE_NUM']).intersection(set(garage_df['REFERENCE_NUM']))
print(f"Found {len(common_refs):,} common reference numbers between datasets")

# Select a sample of reference numbers for detailed analysis
sample_size = min(100, len(common_refs))
sample_refs = list(common_refs)[:sample_size]
print(f"Using {sample_size} reference numbers for sample analysis")

# Create sample dataframes
sample_surveyor = surveyor_df[surveyor_df['REFERENCE_NUM'].isin(sample_refs)]
sample_garage = garage_df[garage_df['REFERENCE_NUM'].isin(sample_refs)]

print(f"Sample surveyor data: {sample_surveyor.shape[0]} records")
print(f"Sample garage data: {sample_garage.shape[0]} records")

# Part Mapping Techniques
print("\nImplementing part mapping techniques...")

# 1. Exact Match
def get_exact_matches(surveyor_parts, garage_parts):
    exact_matches = []
    
    for s_part in surveyor_parts:
        matches = [g_part for g_part in garage_parts if clean_text(s_part) == clean_text(g_part)]
        if matches:
            for match in matches:
                exact_matches.append((s_part, match))
    
    return exact_matches

# 2. Fuzzy Matching
def get_fuzzy_matches(surveyor_parts, garage_parts, threshold=80):
    fuzzy_matches = []
    
    for s_part in surveyor_parts:
        s_part_clean = clean_text(s_part)
        if not s_part_clean:
            continue
            
        best_match, score, _ = process.extractOne(
            s_part_clean, 
            [clean_text(g) for g in garage_parts],
            scorer=fuzz.token_sort_ratio
        )
        
        if score >= threshold:
            idx = [clean_text(g) for g in garage_parts].index(best_match)
            fuzzy_matches.append((s_part, garage_parts[idx], score))
    
    return fuzzy_matches

# 3. TF-IDF and Cosine Similarity
def get_tfidf_matches(surveyor_parts, garage_parts, threshold=0.5):
    # Clean the texts
    s_parts_clean = [clean_text(p) for p in surveyor_parts]
    g_parts_clean = [clean_text(p) for p in garage_parts]
    
    # Filter out empty strings
    valid_s_indices = [i for i, p in enumerate(s_parts_clean) if p]
    valid_g_indices = [i for i, p in enumerate(g_parts_clean) if p]
    
    filtered_s_parts = [s_parts_clean[i] for i in valid_s_indices]
    filtered_g_parts = [g_parts_clean[i] for i in valid_g_indices]
    
    if not filtered_s_parts or not filtered_g_parts:
        return []
    
    # Compute TF-IDF vectors
    vectorizer = TfidfVectorizer()
    all_texts = filtered_s_parts + filtered_g_parts
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Split the matrix into surveyor and garage parts
    s_tfidf = tfidf_matrix[:len(filtered_s_parts)]
    g_tfidf = tfidf_matrix[len(filtered_s_parts):]
    
    # Compute cosine similarity
    similarity = cosine_similarity(s_tfidf, g_tfidf)
    
    # Get best matches
    tfidf_matches = []
    for i, s_idx in enumerate(valid_s_indices):
        best_match_idx = np.argmax(similarity[i])
        best_score = similarity[i][best_match_idx]
        
        if best_score >= threshold:
            g_idx = valid_g_indices[best_match_idx]
            tfidf_matches.append((
                surveyor_parts[s_idx], 
                garage_parts[g_idx], 
                float(best_score)
            ))
    
    return tfidf_matches

# Perform analysis on the sample data
sample_results = []

print("\nAnalyzing mappings for sample claims...")
for ref_num in sample_refs:
    # Get parts for this reference
    s_parts = sample_surveyor[sample_surveyor['REFERENCE_NUM'] == ref_num]['TXT_PARTS_NAME'].tolist()
    g_parts = sample_garage[sample_garage['REFERENCE_NUM'] == ref_num]['PARTDESCRIPTION'].tolist()
    
    if not s_parts or not g_parts:
        continue
    
    # Get exact matches
    exact = get_exact_matches(s_parts, g_parts)
    
    # Get fuzzy matches
    fuzzy = get_fuzzy_matches(s_parts, g_parts)
    
    # Get TF-IDF matches
    tfidf = get_tfidf_matches(s_parts, g_parts)
    
    # Store results for this reference
    sample_results.append({
        'REFERENCE_NUM': ref_num,
        'surveyor_parts': s_parts,
        'garage_parts': g_parts,
        'exact_matches': exact,
        'fuzzy_matches': fuzzy,
        'tfidf_matches': tfidf
    })

# Analyze the mapping results
results_df = []

for res in sample_results:
    ref_num = res['REFERENCE_NUM']
    s_parts = res['surveyor_parts']
    g_parts = res['garage_parts']
    
    # For each surveyor part, find the best match
    for s_part in s_parts:
        best_match = None
        match_type = None
        score = 0
        
        # Check exact matches first
        for exact_s, exact_g in res['exact_matches']:
            if exact_s == s_part:
                best_match = exact_g
                match_type = 'Exact'
                score = 100
                break
        
        # If no exact match, check fuzzy
        if not best_match:
            for fuzzy_s, fuzzy_g, fuzzy_score in res['fuzzy_matches']:
                if fuzzy_s == s_part and fuzzy_score > score:
                    best_match = fuzzy_g
                    match_type = 'Fuzzy'
                    score = fuzzy_score
        
        # If still no match, check TF-IDF
        if not best_match or score < 80:  # Prefer better fuzzy matches over TF-IDF
            for tfidf_s, tfidf_g, tfidf_score in res['tfidf_matches']:
                if tfidf_s == s_part and tfidf_score*100 > score:
                    best_match = tfidf_g
                    match_type = 'TF-IDF'
                    score = tfidf_score*100
        
        # Add to results
        results_df.append({
            'REFERENCE_NUM': ref_num,
            'surveyor_part': s_part,
            'best_garage_match': best_match,
            'match_type': match_type,
            'confidence_score': score
        })

# Convert to DataFrame
mapping_results = pd.DataFrame(results_df)

# Calculate statistics
total_parts = len(mapping_results)
matched_parts = mapping_results['best_garage_match'].notna().sum()
match_rate = matched_parts / total_parts * 100

match_types = mapping_results['match_type'].value_counts(normalize=True) * 100
avg_confidence = mapping_results['confidence_score'].mean()

# Generate report
print("\n" + "="*80)
print(" "*30 + "MAPPING SUMMARY REPORT")
print("="*80)

print(f"\nTotal Surveyor Parts Analyzed: {total_parts}")
print(f"Successfully Mapped Parts: {matched_parts} ({match_rate:.2f}%)")
print(f"Average Confidence Score: {avg_confidence:.2f}%")

print("\nMapping Technique Distribution:")
for match_type, percentage in match_types.items():
    print(f"  - {match_type}: {percentage:.2f}%")

print("\nExample Mappings:")
examples = mapping_results.sort_values('confidence_score', ascending=False).head(10)
for i, row in examples.iterrows():
    print(f"\nReference: {row['REFERENCE_NUM']}")
    print(f"Surveyor Part: {row['surveyor_part']}")
    print(f"Garage Part: {row['best_garage_match']}")
    print(f"Match Type: {row['match_type']}")
    print(f"Confidence: {row['confidence_score']:.2f}%")
    print("-" * 50)

# Save results to CSV
mapping_results.to_csv('part_mapping_results.csv', index=False)
print("\nDetailed results saved to 'part_mapping_results.csv'")

# Generate visualization
plt.figure(figsize=(10, 6))
sns.barplot(x=match_types.index, y=match_types.values)
plt.title('Distribution of Mapping Techniques')
plt.xlabel('Mapping Technique')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('mapping_techniques_distribution.png')

plt.figure(figsize=(10, 6))
sns.histplot(mapping_results['confidence_score'], bins=10)
plt.title('Distribution of Confidence Scores')
plt.xlabel('Confidence Score (%)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('confidence_scores_distribution.png')

# Calculate execution time
end_time = time.time()
execution_time = end_time - start_time

print(f"\nAnalysis completed in {execution_time:.2f} seconds")
print("\nVisualizations saved to:")
print("  - mapping_techniques_distribution.png")
print("  - confidence_scores_distribution.png") 