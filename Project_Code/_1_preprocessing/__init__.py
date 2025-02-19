"""
Preprocessing Package Initialization

This module initializes the preprocessing package which contains tools for data processing
and splitting. It makes the core preprocessing functions available at the package level.

Available Functions:
    - preprocess_data: Handles data preprocessing operations
    - split_data: Manages dataset splitting functionality

Usage:
    from Project_Code._1_preprocessing import preprocess_data, split_data
    # or
    from Project_Code._1_preprocessing import *
"""

from .Processing_data import preprocess_data
from .split import split_data

# Define which functions are exposed when using 'from Project_Code._1_preprocessing import *'
__all__ = [
    'preprocess_data',  # Function for data preprocessing
    'split_data'        # Function for splitting datasets
] 