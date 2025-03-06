"""Detector for sudden baseline shifts."""

import pandas as pd
from .base import ErrorDetector

class OffsetDetector(ErrorDetector):
    """Detector for sudden baseline shifts."""
    
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect sudden persistent changes.
        
        Strategy:
        1. Compare means of adjacent windows
        2. Use CUSUM for change detection
        3. Verify persistence of change
        4. Consider physical limits
        """
        pass 