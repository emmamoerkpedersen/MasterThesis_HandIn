import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add necessary paths
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent.parent
sys.path.append(str(project_dir))

class ForecastVisualizer:
    """
    Class for visualizing water level forecasts and anomalies.
    """
    def __init__(self, config=None):
        """
        Initialize the visualizer with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.anomaly_threshold = config.get('z_score_threshold', 5) if config else 5
    
    # Note: Visualization methods were moved to run_forecast.py for direct integration
    # This class is maintained for compatibility and potential future extensions