import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define folder names
folders = ['21006845', '21006846', '21006847']

# Initialize an empty list to store dataframes
dfs = []

# Loop through each folder and read the data
for folder in folders:
    data_path = os.path.join('Sample data', folder, 'VST_RAW.txt')
    
    # Read the data file with latin-1 encoding
    df = pd.read_csv(data_path, sep=';', encoding='latin-1')
    df.columns = ['Time', 'VST_raw']
    
    # Convert Time column to datetime format
    df['Time'] = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M')
    
    # Replace comma with decimal point only for folders that need it (excluding 21006846)
    if folder != '21006846':
        df['VST_raw'] = df['VST_raw'].str.replace(',', '.').astype(float)
    else:
        df['VST_raw'] = df['VST_raw'].astype(float)
    
    # Save standardized version with decimal points
    standardized_path = os.path.join('Sample data', folder, 'VST_RAW.txt')
    df.to_csv(standardized_path, sep=';', index=False, decimal='.')
    
    # Add a column to identify the source folder
    df['Source'] = folder
    
    dfs.append(df)