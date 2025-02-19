"""Detector for gaps in time series data."""

import pandas as pd
from .base import ErrorDetector

class GapDetector(ErrorDetector):
    """Detector for gaps in time series data."""
    
    def __init__(self, max_gap: int = 1):
        """
        Args:
            max_gap (int): Maximum allowed gap in hours
        """
        super().__init__()
        self.max_gap = max_gap
    
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect gaps exceeding maximum allowed duration."""
        pass 