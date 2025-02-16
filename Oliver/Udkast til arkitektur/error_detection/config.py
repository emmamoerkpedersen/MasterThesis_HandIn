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
        'frequency': 0.0001,  # Probability of occurrence
        'magnitude_range': (0.4, 0.8),  # 40% to 80% of current value
        'negative_positiv_ratio': 0.5,  # Equal chance of positive/negative spikes
        'recovery_time': 1,  # Hours to recover to normal
    },
    'flatline': {
        'frequency': 0.00003,  # Match offset frequency
        'duration_range': (20, 200),  # Longer periods to be more visible
        'value_method': 'first_value',  # Always use the first value of the period
    },
    'drift': {
        'frequency': 0.00003,  # 1% of data points will have drift
        'duration_range': [24, 168],  # drift duration between 24 and 168 hours
        'magnitude_range': [10, 50],  # maximum drift magnitude (increase these values if drifts are too subtle)
        'negative_positive_ratio': 0.5,  # equal chance of positive and negative drift
    },
    'offset': {
        'frequency': 0.00003,
        'magnitude_range': (50, 500),  # Wider range for more varied offsets
        'negative_positiv_ratio': 0.7,  # More positive offsets than negative
        'min_duration': 24,  # Minimum hours
        'max_duration_multiplier': (15, 20),  # Random multiplier for max duration
        'magnitude_multiplier': (0.8, 1.2),  # Random multiplier for local scaling
    },
    'noise': {
        'frequency': 0.008,
        'duration_range': (4, 24),  # Hours
        'intensity_range': (2, 4)  # Multiple of normal noise level
    }
}

# Add physical limits to SYNTHETIC_ERROR_PARAMS
SYNTHETIC_ERROR_PARAMS['PHYSICAL_LIMITS'] = {
    'min_value': 0,
    'max_value': 3000,  # Increased to allow for larger upward spikes
    'max_rate_of_change': 50
}

# Physical Constraints
PHYSICAL_LIMITS = {
    'min_value': 0,  # Water level can't be negative
    'max_value': 3000,  # Maximum reasonable water level
    'max_rate_of_change': 50  # Maximum change per hour
} 