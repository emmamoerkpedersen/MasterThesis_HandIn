"""
Model Registry for managing station-specific LSTM models.
"""

import torch
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Union, Optional

from _3_lstm_model.torch_lstm_model import (
    train_autoencoder, 
    AutoencoderWrapper,
    evaluate_realistic
)

class ModelRegistry:
    """Registry for managing multiple LSTM models for different stations."""
    
    def __init__(self, base_config: Dict, models_dir: Union[str, Path]):
        """
        Initialize the model registry.
        
        Args:
            base_config: Base configuration for models
            models_dir: Directory to store models
        """
        self.models = {}  # station_id -> model
        self.performance = {}  # station_id -> metrics
        self.base_config = base_config
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        
        # Global model (trained on all stations)
        self.global_model = None
        
        print(f"Initialized ModelRegistry with storage at {self.models_dir}")
        
    def train_global_model(self, all_station_data: Dict, validation_data: Optional[Dict] = None) -> Tuple:
        """
        Train a model on all stations combined.
        
        Args:
            all_station_data: Dictionary of all station data
            validation_data: Optional validation data
            
        Returns:
            Tuple of (model, history)
        """
        
        # Train model with base configuration
        self.global_model, history = train_autoencoder(
            train_data=all_station_data,
            validation_data=validation_data,
            config=self.base_config
        )
        
        # Save global model
        self.save_model(self.global_model, "global")
        
        return self.global_model, history
    
    def train_station_model(
        self, 
        station_id: str, 
        station_data: Dict, 
        validation_data: Optional[Dict] = None,
        fine_tune: bool = True
    ) -> Tuple:
        """
        Train or fine-tune a model for a specific station.
        
        Args:
            station_id: ID of the station
            station_data: Training data for the station
            validation_data: Optional validation data
            fine_tune: Whether to fine-tune from global model
            
        Returns:
            Tuple of (model, history)
        """
        if fine_tune and self.global_model is not None:
            # Fine-tune from global model
            print(f"Fine-tuning model for station {station_id} from global model...")
            
            # Create fine-tuning config
            station_config = self.base_config.copy()
            station_config['epochs'] = max(10, station_config.get('epochs', 100) // 3)
            station_config['learning_rate'] = self.base_config.get('learning_rate', 0.001) / 5
            
            # Fine-tune on station data
            station_model, history = train_autoencoder(
                train_data=station_data,
                validation_data=validation_data,
                config=station_config,
                base_model=self.global_model  # Pass global model for initialization
            )
        else:
            # Train from scratch
            print(f"Training new model for station {station_id}...")
            station_config = self.base_config.copy()
            
            station_model, history = train_autoencoder(
                train_data=station_data,
                validation_data=validation_data,
                config=station_config
            )
        
        # Save station model
        self.models[station_id] = station_model
        self.save_model(station_model, station_id)
        
        return station_model, history
    
    def get_model(self, station_id: str) -> AutoencoderWrapper:
        """
        Get the appropriate model for a station.
        
        Args:
            station_id: ID of the station
            
        Returns:
            Model for the station
        """
        # Try to load from memory
        if station_id in self.models:
            return self.models[station_id]
            
        # Try to load from disk
        station_path = self.models_dir / f"{station_id}_model.pt"
        if station_path.exists():
            print(f"Loading saved model for station {station_id}")
            self.models[station_id] = self.load_model(station_id)
            return self.models[station_id]
            
        # Fall back to global model
        if self.global_model is not None:
            print(f"No specific model for {station_id}, using global model")
            return self.global_model
            
        # No models available
        raise ValueError("No models available. Train a global model first.")
    
    def save_model(self, model: AutoencoderWrapper, name: str) -> None:
        """
        Save model to disk.
        
        Args:
            model: Model to save
            name: Name to save the model under (station ID or 'global')
        """
        model_path = self.models_dir / f"{name}_model.pt"
        torch.save(model.model.state_dict(), model_path)
        
        # Save scaler and config
        config_path = self.models_dir / f"{name}_config.json"
        
        # Convert config to JSON-serializable format
        serializable_config = {}
        for k, v in model.config.items():
            if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                serializable_config[k] = v
            else:
                serializable_config[k] = str(v)
        
        with open(config_path, 'w') as f:
            json.dump(serializable_config, f)
            
        # Save scaler
        scaler_path = self.models_dir / f"{name}_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(model.scaler, f)
            
        print(f"Model saved to {model_path}")
    
    def load_model(self, name: str) -> AutoencoderWrapper:
        """
        Load model from disk.
        
        Args:
            name: Name of the model to load
            
        Returns:
            Loaded model
        """
        model_path = self.models_dir / f"{name}_model.pt"
        config_path = self.models_dir / f"{name}_config.json"
        scaler_path = self.models_dir / f"{name}_scaler.pkl"
        
        if not model_path.exists() or not config_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(f"Missing files for model {name}")
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Create model
        model = AutoencoderWrapper(config)
        
        # Load weights
        model.model.load_state_dict(torch.load(model_path, map_location=model.model.device))
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            model.scaler = pickle.load(f)
            model.is_fitted = True
            
        return model
    
    def evaluate_station_performance(
        self, 
        station_id: str, 
        test_data: Dict, 
        ground_truth: Optional[Dict] = None,
        config: Optional[Dict] = None,
        split_datasets = None
    ) -> Dict:
        """
        Evaluate performance of the appropriate model on test data.
        
        Args:
            station_id: Station ID
            test_data: Test data dictionary
            ground_truth: Ground truth anomaly labels (optional)
            config: Configuration override (optional)
            split_datasets: Split datasets information (optional)
            
        Returns:
            Dictionary of evaluation results
        """
        # Use provided config or base config
        eval_config = config if config is not None else self.base_config
        
        # Try to get station-specific model first
        try:
            station_model = self.load_model(station_id)
            print(f"Using station-specific model for {station_id}")
        except Exception as e:
            print(f"Station-specific model not found for {station_id}: {e}")
            print("Falling back to global model")
            try:
                station_model = self.load_model("global")
            except Exception as e2:
                print(f"Global model also not found: {e2}")
                print("No suitable model found for evaluation")
                return {}
        
        # Ensure ground truth is properly formatted
        if ground_truth is None:
            ground_truth = {}
        
        # Use the realistic evaluation approach
        results = evaluate_realistic(
            model=station_model,
            test_data=test_data,
            ground_truth=ground_truth,
            config=eval_config,
            split_datasets=split_datasets
        )
        
        return results 