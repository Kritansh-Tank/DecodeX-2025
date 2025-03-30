# AutoSure Insurance - Part Mapping Solution Report

## Executive Summary

This report presents a solution to AutoSure Insurance's challenge of inconsistent part naming between surveyor and garage datasets. As detailed in the case background, inconsistency in part nomenclature creates significant operational issues during claim verification. Using advanced text analysis techniques, we've developed a system to match part descriptions between these two data sources, helping to streamline the claim verification process and enhance data integrity.

Our analysis found that approximately **15.42%** of surveyor parts could be confidently mapped to corresponding garage parts using a combination of exact matching, fuzzy matching, and TF-IDF based semantic analysis. This relatively modest success rate highlights the significant naming disparities that exist between how surveyors and garages record the same automotive parts.

## Background and Problem Statement

AutoSure Insurance faces a critical challenge in its claims process. When a vehicle is damaged:

1. A surveyor assesses the damage and manually inputs part details into AutoSure's system
2. The garage submits its list of damaged parts needing repair/replacement
3. The insurance team must manually compare these lists to verify claim accuracy

The primary issue stems from inconsistent naming conventions. For example, a surveyor might write "Left Door" while the garage records "Driver-Side Door" for the same part. This inconsistency creates inefficiencies and potential for errors during claim processing.

## Data Overview

Our analysis utilized three key datasets:

1. **Surveyor Data**: 945,216 records containing information submitted by surveyors
   - Key columns: REFERENCE_NUM, TXT_PARTS_GROUP_NAME, TXT_PARTS_NAME, NUM_PART_CODE, TOTAL_AMOUNT

2. **Garage Data**: 365,086 records containing information submitted by garages
   - Key columns: REFERENCE_NUM, PARTNO, PARTDESCRIPTION, TOTAL_AMOUNT

3. **Parts Code Master**: A reference dataset for standardized part codes and names

We identified 9,496 common reference numbers between the surveyor and garage datasets, indicating claims where both sources submitted information.

## Methodology

Our approach employed multiple natural language processing techniques to match parts between datasets:

### 1. Data Preparation
- Performed data cleaning and normalization on part names:
  - Converted all text to lowercase
  - Removed special characters
  - Standardized separators ("|", "-", "/", "\\") to spaces
  - Removed extra whitespace

### 2. Mapping Techniques
We implemented three complementary techniques to match parts:

**a) Exact Matching**
- Direct string comparison after cleaning and standardization
- Highest confidence matches (100% certainty)
- Found to make up 28.21% of successful matches

**b) Fuzzy Matching**
- Utilizes token-based Levenshtein distance algorithm
- Accounts for minor spelling differences and word order variations
- Assigns confidence scores based on string similarity
- Most prevalent matching technique (40.17% of matches)

**c) TF-IDF with Cosine Similarity**
- Creates vector representations of part descriptions
- Measures semantic similarity between terms
- Valuable for matches where terminology differs but meaning is similar
- Responsible for 31.62% of successful matches

### 3. Matching Process Logic
Our algorithm implemented a hierarchical approach to find the best match for each surveyor part:
1. First attempt exact matches (highest confidence)
2. If no exact match, try fuzzy matching with a threshold of 80% similarity
3. If still no good match, use TF-IDF/cosine similarity with a threshold of 0.5

## Results and Analysis

### Match Success Rate
- Total surveyor parts analyzed: 759 (from 100 sample claims)
- Successfully mapped parts: 117 (15.42%)
- Average confidence score across all matches: 12.90%

### Matching Technique Distribution
- Fuzzy matching: 40.17% of successful matches
- TF-IDF/cosine similarity: 31.62% of successful matches
- Exact matching: 28.21% of successful matches

### Example Part Mappings

Here are some successful part mappings from our analysis:

| Reference Number | Surveyor Part | Garage Part | Match Type | Confidence |
|-----------------|---------------|-------------|------------|------------|
| 202112310018323 | CLIP | CLIP | Exact | 100.00% |
| 202112310018323 | NUT | NUT | Exact | 100.00% |
| 201908240023850 | CROSSMEMBER,FRONT LOWER | CROSSMEMBER,FRONT LOWER | Exact | 100.00% |
| 202205250027807 | TAPE,FRONT DOOR OUTER RR,L | TAPE,FRONT DOOR OUTER RR,L | Exact | 100.00% |
| 202205250027807 | TAPE,REAR DOOR OUTER FR,L | TAPE,REAR DOOR OUTER FR,L | Exact | 100.00% |

These examples demonstrate cases where the naming conventions aligned perfectly between the surveyor and garage datasets.

### Matching Challenges

Our analysis revealed several factors that contribute to the relatively low overall match rate:

1. **Drastically Different Naming Conventions**: Surveyors and garages often use entirely different terminology for the same parts
2. **Varying Levels of Detail**: One dataset might include more specific information about a part than the other
3. **Structural Format Differences**: Garages tend to use more technical/manufacturer-specific terms, while surveyors may use more general descriptions
4. **Inconsistent Abbreviations**: Both sources use abbreviations, but not consistently or in the same way
5. **Missing Part Entries**: In some cases, parts listed by surveyors might not appear in garage records, or vice versa

## Recommendations

Based on our analysis, we recommend the following actions to improve part mapping and claims processing:

### 1. Standardize Part Nomenclature
- Develop a comprehensive standardized parts dictionary that both surveyors and garages must adhere to
- Create a digital input system with dropdown menus or autocomplete to enforce standardized naming

### 2. Implement Automated Mapping System
- Deploy the mapping solution developed in this analysis as part of the claims processing system
- Include confidence scores with matches to flag potential discrepancies requiring manual review
- Continuously train the system with verified matches to improve accuracy over time

### 3. Enhanced Surveyor and Garage Portal
- Create a unified digital platform where both parties select parts from the same standardized catalog
- Implement visual aids (diagrams, images) to ensure accurate part identification

### 4. Feedback Loop System
- Establish a mechanism for operations staff to flag and correct mismatches
- Use this feedback to continuously improve the matching algorithms and part dictionaries

### 5. Gradual Migration to Standardized Part Codes
- Transition from text descriptions to standardized part codes over time
- Provide lookup tools and training for both surveyors and garages

## Conclusion

The part mapping solution developed in this analysis demonstrates the potential for automated matching between surveyor and garage part descriptions. While the current match rate of 15.42% highlights the significant naming inconsistencies in the existing data, this solution provides a foundation for improving the claim verification process.

By implementing our recommendations, AutoSure Insurance can gradually increase match rates, reduce manual verification effort, and accelerate claim processing. This will ultimately lead to enhanced operational efficiency, cost savings, and improved customer satisfaction through faster claim settlements.

---

*Report prepared on June 12, 2023* 