import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process
import re
import os

# Function to display information about a DataFrame
def display_df_info(df, name):
    print(f"\n{name} DataFrame Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First 5 rows:")
    print(df.head())
    print(f"Missing values per column:")
    print(df.isnull().sum())
    
# Load the data
print("Loading data...")
try:
    # Load with smaller chunk sizes to handle large files more efficiently
    surveyor_df = pd.read_csv('surveyor_data.csv', low_memory=False)
    garage_df = pd.read_csv('garage_data.csv', low_memory=False)
    parts_code_df = pd.read_excel('Primary_Parts_Code.xlsx')
    
    # Display information about each DataFrame
    display_df_info(surveyor_df, "Surveyor")
    display_df_info(garage_df, "Garage")
    display_df_info(parts_code_df, "Parts Code")
    
    # Check if we have common IDs to match records
    common_cols = set(surveyor_df.columns).intersection(set(garage_df.columns))
    print(f"\nCommon columns between surveyor and garage data: {common_cols}")
    
except Exception as e:
    print(f"Error loading or analyzing data: {e}")

# Save this initial analysis to a file for easier viewing
with open('data_analysis_results.txt', 'w') as f:
    import sys
    original_stdout = sys.stdout
    sys.stdout = f
    display_df_info(surveyor_df, "Surveyor")
    display_df_info(garage_df, "Garage")
    display_df_info(parts_code_df, "Parts Code")
    print(f"\nCommon columns between surveyor and garage data: {common_cols}")
    sys.stdout = original_stdout

print("Initial analysis completed and saved to data_analysis_results.txt") 