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
    'max_value': 3000,  # Maximum reasonable water level
    'max_rate_of_change': 50  # Maximum change per hour
}

# LSTM Model Configuration
LSTM_CONFIG = {
    # Core model parameters
    'sequence_length': 96,  # Default to 24 hours (15-min intervals)
    'n_features': 6,  # Original + 5 engineered features
    'lstm_units': [64, 32],  # Conservative default architecture
    'dropout_rate': 0.3,
    'learning_rate': 0.001,  # More conservative default
    'epochs': 20,
    'batch_size': 32,
    'early_stopping_patience': 5,
    'feature_cols': ['Value'],
    
    # Fine-tuning parameters
    'fine_tuning_epochs': 10,
    'fine_tuning_batch_size': 32,
    'fine_tuning_lr': 0.0001,  # Conservative fine-tuning rate
    
    # Validation parameters
    'validation_split': 0.2,
    'min_samples': 100,  # Minimum number of samples required for training
    'max_samples': 10000,  # Maximum number of samples to use (prevent memory issues)
    
    # Architecture features
    'use_attention': False,
    'use_residual': False,
    'use_layer_norm': False,
    
    # New: Iterative training settings
    'iterative_training': {
        'enabled': False,
        'max_iterations': 3,
        'refinement_params': {
            'confidence_threshold_std': 2.0,  # How many std devs for "normal" data
            'min_data_fraction': 0.7,  # Don't remove more than 30% of data
            'analysis_window': 96,  # 24h window for analyzing patterns
        }
    }
} 