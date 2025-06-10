import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .feature_scaler import FeatureScaler
from .feature_engineering import FeatureEngineer

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize feature columns list with base features
        self.feature_cols = config['feature_cols'].copy()
        
        # Add feature station columns dynamically
        for station in config['feature_stations']:
            for feature in station['features']:
                feature_name = f"feature_station_{station['station_id']}_{feature}"
                self.feature_cols.append(feature_name)
                
        self.output_features = config['output_features'][0]
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(config)
        
        # Initialize feature scaler
        self.feature_scaler = FeatureScaler(
            feature_cols=self.feature_cols,
            output_features=self.output_features,
            device=self.device
        )
        
    
    
    def update_feature_scaler(self):
        """
        Update the feature scaler with the latest feature columns.
        This should be called after adding new features.
        """
        # If we already have a fitted scaler, preserve its state
        was_fitted = getattr(self.feature_scaler, 'is_fitted', False)
        old_scalers = getattr(self.feature_scaler, 'scalers', {})
        
        # Create new scaler with updated feature columns
        self.feature_scaler = FeatureScaler(
            feature_cols=self.feature_cols,
            output_features=self.output_features,
            device=self.device
        )
        
        # If we had a fitted scaler, restore its state
        if was_fitted and old_scalers:
            self.feature_scaler.scalers = old_scalers
            self.feature_scaler.is_fitted = True
    
    def load_and_split_data(self, project_root, station_id):
        """
        Load and Split data into features and target.
        """
        data_dir = project_root / "data_utils" / "Sample data"
        data = pd.read_pickle(data_dir / "preprocessed_data.pkl")
        # Print the structure of the data
        print(data.keys())
        print(data[station_id].keys())
        # Save vinge data to use for plotting 
        vinge_data = data[station_id]['vinge']
        # Check if station_id exists in the data dictionary, if not return empty dict
        station_data = data.get(station_id)
        if not station_data:
            raise ValueError(f"Station ID {station_id} not found in the data.")

      
        # Concatenate all station data columns
        df = pd.concat(station_data.values(), axis=1)
       
        # Add feature station data
        for station in self.config['feature_stations']:
            feature_station_id = station['station_id']
            feature_station_data = data.get(feature_station_id)
            
            if not feature_station_data:
                raise ValueError(f"Feature station ID {feature_station_id} not found in the data.")
            
            # Add each requested feature from the feature station
            for feature in station['features']:
                if feature not in feature_station_data:
                    raise ValueError(f"Feature {feature} not found in station {feature_station_id}")
                    
                feature_data = feature_station_data[feature][feature].rename(
                    f"feature_station_{feature_station_id}_{feature}"
                )
                df = pd.concat([df, feature_data], axis=1)

        # Print input features information
        print("\nInput Features Summary:")
        print("----------------------")
        print("Base features:")
        for feature in self.config['feature_cols']:
            print(f"  - {feature}")
        
        print("\nFeature stations:")
        for station in self.config['feature_stations']:
            print(f"  - Station {station['station_id']}:")
            for feature in station['features']:
                print(f"    * {feature}")
        
        print("\nFeature engineering settings:")
        print(f"  - Using time features: {self.config.get('use_time_features', False)}")
        print(f"  - Using cumulative features: {self.config.get('use_cumulative_features', False)}")
        print("----------------------\n")

        # Start_date is first rainfall not nan, End_date is last vst_raw not nan
        start_date = pd.Timestamp('2010-01-04')
        end_date = pd.Timestamp('2025-01-07')
        # Cut dataframe
        data = df[(df.index >= start_date) & (df.index <= end_date)]
        vinge_data = vinge_data[(vinge_data.index >= start_date) & (vinge_data.index <= end_date)]
        # Fill NaN in vst_raw_feature, and vst_raw_feature_station_21006845 and 46 with -1
        # data.loc[:, 'feature_station_21006845_vst_raw'] = data['feature_station_21006845_vst_raw'].fillna(-1)
        # data.loc[:, 'feature_station_21006847_vst_raw'] = data['feature_station_21006847_vst_raw'].fillna(-1)
  
        #Aggregate temperature to 30 days = 2880 steps
        data.loc[:, 'temperature'] = data['temperature'].rolling(window=2880, min_periods=1).mean()
       
        # Aggregated feature_station_21006845_temperature to 30 days = 2880 steps
        #data.loc[:, 'feature_station_21006845_temperature'] = data['feature_station_21006845_temperature'].rolling(window=2880, min_periods=1).mean()
        print(f"  - Aggregated temperature to 30 days")

        # Add cumulative rainfall features if enabled in config
        if self.config.get('use_cumulative_features', False):
            data = self._add_cumulative_features(data)
            # Update feature_cols with the new cumulative features
            self.feature_cols = self.feature_engineer.feature_cols.copy()
            # Update the feature scaler with the new feature columns
            self.update_feature_scaler()
        
        # Add time-based features if enabled in config
        if self.config.get('use_time_features', False):
            data = self._add_time_features(data)
            # Update feature_cols with the new time features
            self.feature_cols = self.feature_engineer.feature_cols.copy()
            # Update the feature scaler with the new feature columns
            self.update_feature_scaler()

            
        feature_cols = self.feature_cols
        target_feature = self.output_features
        all_features = list(set(feature_cols + [target_feature]))        
        #Filter data to only include the features and target feature
        data = data[all_features]

        # test_data = data
        # train_data = data
        # val_data = data
        # Split data based on years
        test_data = data[(data.index.year == 2024)]
        val_data = data[(data.index.year >= 2022) & (data.index.year <= 2023)]  # Validation is 2022-2023
        train_data = data[data.index.year < 2022]  # Training is everything before 2022
        
        
        print(f"\nSplit Summary:")
        print(f"Training period: {train_data.index.min().year} - {train_data.index.max().year}")
        print(f"Validation period: {val_data.index.min().year} - {val_data.index.max().year}")
        print(f"Test year: {test_data.index.min().year}")
        
        print(f'\nData shapes:')
        print(f'Total data: {data.shape}')
        print(f'Train data: {train_data.shape}')
        print(f'Validation data: {val_data.shape}')
        print(f'Test data: {test_data.shape}')

        return train_data, val_data, test_data, vinge_data
    
    def prepare_data(self, data, is_training=True):
        """
        Prepare data for training or validation. Scale data and create sequences.
        """
        # Add time and cumulative features if enabled
        if self.config.get('use_time_features', False):
            data = self._add_time_features(data)
        
        if self.config.get('use_cumulative_features', False):
            data = self._add_cumulative_features(data)
            
        
        # Get the most up-to-date feature columns from self.feature_cols
        feature_cols = self.feature_cols
        target_col = self.output_features
        
        # Ensure all feature columns exist in data
        missing_cols = [col for col in feature_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")
        
        # Extract features and target
        features = data[feature_cols]
        target = pd.DataFrame(data[target_col])
        
        # Scale data using FeatureScaler
        if is_training:
            scaled_features, scaled_target = self.feature_scaler.fit_transform(features, target)
        else:
            # For validation/test, ensure the scaler has been fitted
            if not self.feature_scaler.is_fitted:
                raise RuntimeError("Feature scaler has not been fitted. Ensure training data is processed first.")
            scaled_features, scaled_target = self.feature_scaler.transform(features, target)
        
        # Create sequences with overlap
        X, y = self._create_sequences(scaled_features, scaled_target)
        
        # Convert to tensors and move to device
        return torch.FloatTensor(X).to(self.device), torch.FloatTensor(y).to(self.device)
    
    def _create_sequences(self, features, targets):
         """
         Create sequences using the configured sequence length.
         """
         sequence_length = self.config.get('sequence_length', 5000)  # Get from config or default to 5000
         data_length = len(features)
         
         X, y = [], []
 
         # Create sequences based on configured sequence length
         for i in range(0, data_length, sequence_length):
             end_idx = min(i + sequence_length, data_length)
             feature_seq = features[i:end_idx]
             target_seq = targets[i:end_idx]
 
             # Pad sequences if needed
             if end_idx - i < sequence_length:
                 pad_length = sequence_length - (end_idx - i)
                 feature_seq = np.pad(feature_seq, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
                 target_seq = np.pad(target_seq, (0, pad_length), mode='constant', constant_values=np.nan)
 
             X.append(feature_seq)
             y.append(target_seq)
 
         X = np.array(X)  # Shape: (num_sequences, sequence_length, num_features)
         y = np.array(y)[..., np.newaxis]  # Shape: (num_sequences, sequence_length, 1)
 
         return X, y

    def _add_time_features(self, data):
        """
        Add time-based features to the data.
        """
        return self.feature_engineer._add_time_features(data)
        
    def _add_cumulative_features(self, data):
        """
        Add cumulative features to the data.
        """
        return self.feature_engineer._add_cumulative_features(data)