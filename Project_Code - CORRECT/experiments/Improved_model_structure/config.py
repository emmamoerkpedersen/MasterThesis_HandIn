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

# Imputation Parametersx
IMPUTATION_PARAMS = {
    'linear': {
        'max_gap': 24  # hours
    },
    'statistical': {
        'window_size': 168  # hours (1 week)
    }
}


# Synthetic Error Generation Parameters
SYNTHETIC_ERROR_PARAMS = {
    # Add context-aware toggle
    'use_context_aware': False,
    
    'spike': {
        'frequency': 0.00003,
        # Wider range to allow for more subtle spikes
        'magnitude_range': (0.1, 1.0),  # Changed from (0.4, 0.8)
        'negative_positiv_ratio': 0.5,
        'recovery_time': 1,
        # Add specific context-aware settings for spikes
        'context_aware': {
            'subtle_prob': 0.6,    # Probability of subtle spike
            'medium_prob': 0.3,    # Probability of medium spike
            'obvious_prob': 0.1,   # Probability of obvious spike
            'subtle_factor': 0.5,  # Multiplier for subtle range
            'medium_factor': 2.0,  # Multiplier for medium range
            'obvious_factor': 4.0  # Multiplier for obvious range
        }
    },
    
    'drift': {
        'frequency': 0.00003,
        # Wider range for drift durations
        'duration_range': [12, 168],  # Changed from [24, 168]
        # More varied magnitude range
        'magnitude_range': [5, 50],   # Changed from [10, 50]
        'negative_positive_ratio': 0.5,
        'context_aware': {
            'subtle_prob': 0.5,
            'medium_prob': 0.3,
            'obvious_prob': 0.2
        }
    },
    
    'flatline': {
        'frequency': 0.00003,
        # More varied durations
        'duration_range': (10, 200),  # Changed from (20, 200)
        'value_method': 'first_value'
    },
    
    'offset': {
        'frequency': 0.00003,
        # Wider range for offset magnitudes
        'magnitude_range': (20, 500),  # Changed from (50, 500)
        'negative_positiv_ratio': 0.7,
        'min_duration': 24,
        'max_duration_multiplier': (15, 20),
        'magnitude_multiplier': (0.6, 1.4),  # Changed from (0.8, 1.2)
        'context_aware': {
            'subtle_prob': 0.5,
            'medium_prob': 0.3,
            'obvious_prob': 0.2
        }
    },
    
    'noise': {
        'frequency': 0,  # Still disabled
        'duration_range': (4, 24),
        'intensity_range': (1, 4)  # Changed from (2, 4)
    },
    
    'baseline_shift': {
        'frequency': 0,  # Still disabled
        'magnitude_range': (100, 600),  # Changed from (200, 600)
        'negative_positive_ratio': 0.5
    },
    
    # Updated physical limits
    'PHYSICAL_LIMITS': {
        'min_value': 0,
        'max_value': 3000,
        'max_rate_of_change': 50
    }
}

# Physical Constraints
PHYSICAL_LIMITS = {
    'min_value': 0,  # Water level can't be negative
    'max_value': 4000,  # Maximum reasonable water level
    'max_rate_of_change': 50  # Maximum change per hour
}

# Configuration for LSTM model
LSTM_CONFIG = {
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.273,
    'batch_size': 128,
    'epochs': 500,
    'patience': 15,
    'warmup_length': 100,
    'learning_rate': 0.0058,
    'sequence_length': 10000,
    'feature_cols': [
        'rainfall',
    ],
    'output_features': ['vst_raw'],

    
    'use_time_features': True,
    'use_peak_weighted_loss': False,
    'peak_weight': 2.304,
    'grad_clip_value': 1.148,
    'use_smoothing': True,
    'smoothing_alpha': 0.457,
    'feature_stations': [
        {
            'station_id': '21006845',
            'features': ['vst_raw', 'rainfall']
        },
        {
            'station_id': '21006847',
            'features': ['vst_raw', 'rainfall']
        }
    ]
}
'''
 0.43391583273927786 and parameters: {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2732082146489317, 'learning_rate': 0.005823484689107212,
   'batch_size': 128, 'sequence_length': 1000, 'peak_weight': 2.3047103629304044, 'grad_clip_value': 1.1489855340612218, 'smoothing_alpha': 0.4570339719720078}.
'''