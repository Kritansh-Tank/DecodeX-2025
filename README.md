# AutoSure Insurance - Part Mapping Solution

## Overview

This project addresses AutoSure Insurance's challenge of inconsistent part naming between surveyor and garage datasets. Through advanced text analysis techniques, we've developed a system to match automotive part descriptions between these different sources, helping to streamline the claims verification process.

## Key Files

1. **part_mapping_solution.py** - The main implementation script that:
   - Processes the surveyor and garage datasets
   - Implements three text matching techniques (exact, fuzzy, TF-IDF)
   - Analyzes and outputs mapping results

2. **AutoSure_Part_Mapping_Report.md** - Comprehensive business report including:
   - Executive summary
   - Methodology explanation
   - Results analysis
   - Recommendations for implementation

3. **AutoSure_Technical_Implementation.md** - Technical documentation covering:
   - System architecture
   - Algorithm details
   - Implementation specifics
   - Extension possibilities

4. **Executive_Summary_and_Results.py** - Script that generates visualizations including:
   - Match success rate
   - Technique distribution
   - Confidence score analysis
   - Example mappings

5. **analyze_parts.py** - Initial exploration script for data analysis

## Visualizations

The solution includes several visualizations (in the `figures` directory):
- **executive_summary_dashboard.png** - Dashboard with key metrics and findings
- **matching_technique_distribution.png** - Distribution of matching techniques used
- **confidence_distribution.png** - Distribution of confidence scores by match type
- **matching_success_rate.png** - Overall success rate of the mapping process
- **example_mappings.png** - Examples of successful mappings

## Results Summary

Our analysis found that approximately **15.42%** of surveyor parts could be confidently mapped to corresponding garage parts. This match rate reflects the significant naming inconsistencies between how surveyors and garages record automotive parts.

The most successful mapping techniques were:
1. **Fuzzy Matching** (40.17% of successful matches)
2. **TF-IDF with Cosine Similarity** (31.62% of successful matches)
3. **Exact Matching** (28.21% of successful matches)

## Recommendations

Based on our analysis, we recommend:

1. **Standardize Part Nomenclature** - Develop a standardized parts dictionary for both surveyors and garages
2. **Implement Automated Mapping** - Deploy this matching solution as part of the claims processing system
3. **Create Enhanced Portal** - Develop a unified platform with standardized part selection
4. **Establish Feedback Loop** - Create a system for operations staff to flag and correct mismatches
5. **Migrate to Standardized Codes** - Transition from text descriptions to standardized part codes

## Usage Instructions

1. Ensure all dependencies are installed:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn rapidfuzz
   ```

2. Place the data files in the working directory:
   - surveyor_data.csv
   - garage_data.csv 
   - Primary_Parts_Code.xlsx

3. Run the solution:
   ```
   python part_mapping_solution.py
   ```

4. Generate visualizations:
   ```
   python Executive_Summary_and_Results.py
   ```

## Conclusion

This solution provides AutoSure Insurance with a foundation for addressing the challenge of inconsistent part naming, helping to streamline the claims verification process and improve operational efficiency. 