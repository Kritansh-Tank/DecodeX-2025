# AutoSure Insurance - Part Mapping Solution Technical Documentation

## System Overview

This document provides technical details on the implementation of the AutoSure Insurance part mapping solution, which addresses the challenge of matching inconsistently named automotive parts between surveyor and garage datasets. The solution employs multiple natural language processing (NLP) techniques to establish connections between differently named parts that refer to the same physical component.

## Technology Stack

The solution is implemented using the following technologies:

- **Python 3.x**: Primary programming language
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations
- **scikit-learn**: Machine learning toolkit used for TF-IDF vectorization
- **rapidfuzz**: High-performance fuzzy string matching library
- **matplotlib & seaborn**: Data visualization
- **Regular expressions (re)**: Pattern matching for text cleaning

## Data Processing Pipeline

### 1. Data Loading and Initial Exploration

```python
# Load the datasets
surveyor_df = pd.read_csv('surveyor_data.csv', low_memory=False)
garage_df = pd.read_csv('garage_data.csv', low_memory=False)
parts_code_df = pd.read_excel('Primary_Parts_Code.xlsx')
```

The solution begins by loading three key datasets:
- Surveyor data (945,216 records)
- Garage data (365,086 records)
- Parts code master data (standardized part references)

### 2. Text Preprocessing

Text normalization is a critical step to enable effective matching. We implement the following cleaning operations:

```python
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
```

This function performs several key transformations:
- Conversion to lowercase to neutralize case differences
- Removal of special characters while preserving alphanumeric content
- Standardization of common separators (|, -, /, \) to spaces
- Normalization of whitespace to prevent spacing issues

### 3. Matching Algorithms Implementation

The solution implements three complementary matching techniques:

#### a) Exact Matching

```python
def get_exact_matches(surveyor_parts, garage_parts):
    exact_matches = []
    
    for s_part in surveyor_parts:
        matches = [g_part for g_part in garage_parts if clean_text(s_part) == clean_text(g_part)]
        if matches:
            for match in matches:
                exact_matches.append((s_part, match))
    
    return exact_matches
```

The exact matching algorithm:
- Compares cleaned part names for perfect equality
- Handles multiple matches if they exist
- Provides the highest confidence assignments (100%)

#### b) Fuzzy Matching

```python
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
```

The fuzzy matching algorithm:
- Uses token sort ratio to handle word order differences
- Applies a minimum threshold of 80% similarity
- Returns confidence scores based on string similarity
- Handles missing or empty text values

#### c) TF-IDF with Cosine Similarity

```python
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
```

The TF-IDF algorithm:
- Creates vector representations of part descriptions based on term frequency and importance
- Measures semantic similarity using cosine similarity
- Applies a minimum threshold of 0.5 (50%) similarity
- Handles empty or missing text values
- Maps vector indices back to original part names

### 4. Hierarchical Matching Process

The solution implements a priority-based matching strategy:

```python
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
```

This hierarchical approach:
1. Prioritizes exact matches (highest confidence)
2. Falls back to fuzzy matching if no exact match exists
3. Uses TF-IDF only when fuzzy matching fails to find a good match (< 80%)
4. Normalizes confidence scores to a 0-100% range
5. Records the match type for each match for analysis

## Performance Considerations

The implementation includes several optimizations to handle large datasets:

1. **Selective Processing**: Using a sample of claims for detailed analysis to reduce computation time
2. **Filter Before Processing**: Removing empty strings and invalid values before applying expensive operations
3. **Vectorized Operations**: Using NumPy and pandas to vectorize operations where possible
4. **Memory Management**: Processing data in smaller chunks when needed

## Evaluation Metrics

The solution provides several key metrics to evaluate the matching performance:

1. **Match Rate**: Percentage of surveyor parts successfully mapped to garage parts
2. **Confidence Score Distribution**: Distribution of match confidence levels
3. **Matching Technique Usage**: Breakdown of which techniques were most effective
4. **Example Mappings**: Concrete examples of successful matches with their confidence scores

## Usage Instructions

To run the solution:

1. Ensure all dependencies are installed:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn rapidfuzz
   ```

2. Place the data files in the working directory:
   - surveyor_data.csv
   - garage_data.csv
   - Primary_Parts_Code.xlsx

3. Execute the solution script:
   ```
   python part_mapping_solution.py
   ```

4. Review the generated outputs:
   - Console output with summary statistics
   - part_mapping_results.csv with detailed mapping results
   - Visualization files showing the distribution of techniques and confidence scores

## Extension Points

The current implementation can be extended in several ways:

1. **Custom Scoring Functions**: Implement domain-specific similarity functions that incorporate automotive knowledge
2. **Ensemble Approach**: Combine multiple similarity measures with weighted voting
3. **Interactive Mode**: Add an interactive mode for manual review of low-confidence matches
4. **Incremental Learning**: Implement a feedback mechanism to improve matches based on user corrections
5. **API Integration**: Expose the matching functionality as a web service for integration with existing systems

## Limitations and Constraints

The current solution has a few limitations to be aware of:

1. **Language Dependency**: Optimized for English text and may need adjustments for other languages
2. **Performance with Very Large Datasets**: May require distributed processing for production-scale deployment
3. **Domain Knowledge**: Limited incorporation of automotive-specific terminology knowledge
4. **Match Rate Ceiling**: The fundamental naming inconsistency limits the maximum achievable match rate

## Conclusion

This technical documentation outlines the implementation details of the AutoSure Insurance part mapping solution. By combining multiple text matching techniques and implementing a hierarchical matching process, the solution provides a foundational framework for addressing the part naming inconsistency challenge. Future iterations can build upon this foundation to further improve match rates and integration capabilities. 