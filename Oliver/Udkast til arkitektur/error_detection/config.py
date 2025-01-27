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
    'spike': {
        'frequency': 0.01,  # Probability of occurrence
        'magnitude_range': (2, 5),  # Multiple of local std deviation
        'recovery_time': 1,  # Hours to recover to normal
    },
    'flatline': {
        'frequency': 0.005,
        'duration_range': (2, 48),  # Hours
        'value_method': 'last_value',  # or 'mean' or 'custom'
    },
    'drift': {
        'frequency': 0.002,
        'duration_range': (24, 168),  # Hours (1-7 days)
        'drift_rate': (0.1, 0.5),  # Units per hour
        'pattern': 'linear'  # or 'exponential'
    },
    'offset': {
        'frequency': 0.003,
        'magnitude_range': (10, 50),  # Absolute units
        'min_duration': 24  # Minimum hours to maintain offset
    },
    'noise': {
        'frequency': 0.008,
        'duration_range': (4, 24),  # Hours
        'intensity_range': (2, 4)  # Multiple of normal noise level
    }
}

# Physical Constraints
PHYSICAL_LIMITS = {
    'min_value': 0,  # Water level can't be negative
    'max_value': 1000,  # Maximum reasonable water level
    'max_rate_of_change': 50  # Maximum change per hour
} 