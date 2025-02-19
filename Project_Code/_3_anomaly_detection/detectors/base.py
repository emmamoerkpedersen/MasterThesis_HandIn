"""Base class for error detection algorithms."""

import pandas as pd
from typing import Dict, Optional

class ErrorDetector:
    """Base class for error detection algorithms."""
    
    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
    
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Base detection method to be implemented by specific detectors.
        
        Args:
            data (pd.DataFrame): Input time series data with columns ['Date', 'Value']
            
        Returns:
            pd.DataFrame: DataFrame with columns:
                - timestamp: Time of measurement
                - value: Original measurement
                - error_type: Type of error detected
                - confidence: Detection confidence score (0-1)
        """
        raise NotImplementedError 