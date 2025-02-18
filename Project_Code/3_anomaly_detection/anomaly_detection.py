"""
Module for detecting various types of errors in time series data.
Implements detection algorithms for spikes, gaps, flatlines, and other error types.
"""

import pandas as pd
from typing import Dict, List, Optional, Union

from detectors.base import ErrorDetector
from detectors.spike_detector import SpikeDetector
from detectors.flatline_detector import FlatlineDetector
from detectors.drift_detector import DriftDetector
from detectors.offset_detector import OffsetDetector
from detectors.noise_detector import NoiseDetector
from detectors.gap_detector import GapDetector

class AnomalyDetector:
    """Hub for various error detection algorithms."""
    
    def __init__(self):
        self.detectors = {
            'spike': SpikeDetector(),
            'flatline': FlatlineDetector(),
            'drift': DriftDetector(),
            'offset': OffsetDetector(),
            'noise': NoiseDetector(),
            'gap': GapDetector()
        }
    
    def detect(self, data: pd.DataFrame, 
               detector_types: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        Run specified error detectors on the input data.
        
        Args:
            data (pd.DataFrame): Input time series data
            detector_types (Optional[Union[str, List[str]]]): Specific detector(s) to run.
                If None, runs all detectors.
                
        Returns:
            pd.DataFrame: Combined results from all specified detectors
        """
        if detector_types is None:
            detector_types = list(self.detectors.keys())
        elif isinstance(detector_types, str):
            detector_types = [detector_types]
            
        results = []
        for detector_type in detector_types:
            if detector_type not in self.detectors:
                raise ValueError(f"Unknown detector type: {detector_type}")
            detector_result = self.detectors[detector_type].detect(data)
            results.append(detector_result)
            
        # Combine and deduplicate results
        return pd.concat(results).sort_index().drop_duplicates() 