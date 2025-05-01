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


# Synthetic Error Generation Parameters
SYNTHETIC_ERROR_PARAMS = {
    # Add context-aware toggle
    'use_context_aware': False,
    
    'spike': {
        'frequency': 0.0000,
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
        'frequency': 0.0000,
        # Wider range for drift durations
        'duration_range': [500, 1168],  # Changed from [24, 168]
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
        'frequency': 0.000,
        # More varied durations
        'duration_range': (10, 200),  # Changed from (20, 200)
        'value_method': 'first_value'
    },
    
    'offset': {
        'frequency': 0.0000,
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
        'frequency': 0.0000,  # Still disabled
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

# Three-Pass Anomaly Correction Parameters
ANOMALY_CORRECTION_PARAMS = {
    'contamination': 0.05,         # Expected proportion of anomalies (default: 5%)
    'method': 'isolation_forest',  # Anomaly detection method ('isolation_forest', 'one_class_svm', 'statistical')
    'smoothing_window': 5,         # Window size for residual smoothing
    'correction_window': 24,       # Hours to look ahead/behind for context during correction
    'min_segment_length': 3,       # Minimum length of anomaly segment to correct (in hours)
    'visualization': {
        'max_segments': 5,         # Maximum number of individual segments to visualize
        'context_hours': 24        # Hours of context to show before/after anomaly in segment plots
    }
}

# Physical Constraints
PHYSICAL_LIMITS = {
    'min_value': 0,  # Water level can't be negative
    'max_value': 4000,  # Maximum reasonable water level
    'max_rate_of_change': 50  # Maximum change per hour
}

# LSTM Configuration
LSTM_CONFIG = {

    'hidden_size': 128,         
    'num_layers': 3,            
    'dropout': 0.25,             
    'batch_size': 16,
    'sequence_length': 5000,
    'epochs': 100,
    'patience': 15,            

    'warmup_length': 100,
    'learning_rate': 0.001,    

    # 'peak_weighted_loss', 'dynamic_weighted_loss', 'smoothL1_loss', 'mse_loss', 'peak_focused_loss'

    'use_time_features': True,  
    'use_cumulative_features': True, 
    # Add lag features for better prediction
    'use_lagged_features':False,
    'lag_hours': [1, 2, 3, 6, 12, 24],  # Lag periods in hours
    
    'feature_cols': [
        'rainfall',
        'vst_raw_feature'
    ],
    'output_features': ['vst_raw'],

    'feature_stations': [
        {
            'station_id': '21006845',
            'features': ['rainfall']
        },
        {
            'station_id': '21006847',
            'features': ['rainfall']
        }
    ]
}
