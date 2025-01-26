"""
Configuration settings for error detection and imputation.
"""

# Error Detection Parameters
DETECTION_PARAMS = {
    'spike': {
        'window_size': 24,
        'threshold': 3.0
    },
    'gap': {
        'max_gap_hours': 1
    },
    'flatline': {
        'min_duration': 2,
        'max_duration': 48
    }
}

# Imputation Parameters
IMPUTATION_PARAMS = {
    'linear': {
        'max_gap': 24  # hours
    },
    'statistical': {
        'window_size': 168  # hours (1 week)
    }
}

# Validation Parameters
VALIDATION_PARAMS = {
    'train_years': (1970, 2010),
    'test_years': (2010, 2020)
}

# Synthetic Error Generation Parameters
SYNTHETIC_ERROR_PARAMS = {
    'spike_frequency': 0.01,
    'gap_frequency': 0.01,
    'flatline_frequency': 0.01
} 