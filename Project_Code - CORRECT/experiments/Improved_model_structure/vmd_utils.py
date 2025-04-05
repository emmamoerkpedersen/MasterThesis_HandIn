import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from vmdpy import VMD
import os
import webbrowser
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class VMD_Processor:
    """
    Utility class for applying Variational Mode Decomposition (VMD) to water level data
    and extracting meaningful features for improved prediction.
    """
    
    def __init__(self, config=None):
        """
        Initialize the VMD processor with configuration parameters
        
        Args:
            config: Dictionary containing VMD parameters
        """
        # Default VMD parameters - tuned for water level data
        self.config = {
            'alpha': 2000,      # Bandwidth constraint
            'tau': 0,           # Noise-tolerance
            'K': 4,             # Number of modes to decompose into
            'DC': 0,            # No DC part imposed
            'init': 1,          # Initialize omegas uniformly
            'tol': 1e-7         # Tolerance for convergence
        }
        
        # Update with provided config if available
        if config:
            self.config.update(config)
            
    def decompose(self, signal, sample_rate=1.0):
        """
        Decompose a signal into its constituent modes using VMD
        
        Args:
            signal: 1D numpy array or pandas Series containing the signal to decompose
            sample_rate: Sample rate of the signal (default: 1.0 - hourly data)
            
        Returns:
            Dictionary containing the modes, reconstructed signal, and frequency info
        """
        # Convert to numpy array if it's a pandas Series
        if isinstance(signal, pd.Series):
            # Keep a copy of the original index for later alignment
            self.original_index = signal.index
            signal_values = signal.values
        else:
            signal_values = signal
            self.original_index = None
            
        # Handle NaN values by linear interpolation
        nan_mask = np.isnan(signal_values)
        if np.any(nan_mask):
            print(f"Warning: {np.sum(nan_mask)} NaN values detected in the signal. Interpolating.")
            signal_values = self._interpolate_nans(signal_values)
            
        # Apply VMD
        u, u_hat, omega = VMD(
            signal_values, 
            alpha=self.config['alpha'],
            tau=self.config['tau'],
            K=self.config['K'],
            DC=self.config['DC'],
            init=self.config['init'],
            tol=self.config['tol']
        )
        
        # Calculate the instantaneous frequencies
        omega_mean = np.mean(omega, axis=1)
        
        # Calculate the power of each mode
        mode_power = np.sum(u**2, axis=1)
        
        # Store the decomposition results
        self.modes = u
        self.mode_freqs = omega_mean * sample_rate
        self.mode_power = mode_power
        
        # Create a dictionary to return
        result = {
            'modes': u,
            'freq': omega_mean * sample_rate,  # Convert to cycles/sample
            'power': mode_power,
            'reconstructed': np.sum(u, axis=0)
        }
        
        return result
    
    def _interpolate_nans(self, signal):
        """
        Interpolate NaN values in the signal
        
        Args:
            signal: 1D numpy array with possible NaN values
            
        Returns:
            Signal with NaN values interpolated
        """
        # Create a mask of NaN values
        nan_mask = np.isnan(signal)
        
        # Create an array of indices
        indices = np.arange(len(signal))
        
        # Get indices and values of non-NaN elements
        valid_indices = indices[~nan_mask]
        valid_values = signal[~nan_mask]
        
        # Interpolate NaN values
        if len(valid_values) > 0:
            interpolated = np.interp(indices, valid_indices, valid_values)
            return interpolated
        else:
            # If all values are NaN, replace with zeros
            return np.zeros_like(signal)
    
    def plot_decomposition(self, signal, result=None, title="VMD Decomposition of Water Level Data"):
        """
        Plot the original signal and its modes
        
        Args:
            signal: Original signal (1D numpy array or pandas Series)
            result: Decomposition result from decompose() method. If None, uses the last decomposition.
            title: Plot title
        """
        if result is None:
            # Use the most recent decomposition
            if not hasattr(self, 'modes'):
                raise ValueError("No decomposition has been performed yet.")
            modes = self.modes
            mode_freqs = self.mode_freqs
            mode_power = self.mode_power
        else:
            modes = result['modes']
            mode_freqs = result['freq']
            mode_power = result['power']
            
        # Get signal values and create time array
        if isinstance(signal, pd.Series):
            signal_values = signal.values
            time_index = signal.index
        else:
            signal_values = signal
            time_index = np.arange(len(signal_values))
            
        # Number of modes plus original signal
        n_plots = modes.shape[0] + 1
        
        # Create figure
        fig = make_subplots(
            rows=n_plots,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=['Original Signal'] + [
                f'Mode {i+1} (Freq: {freq:.2f} cycles/sample, Power: {power:.2f})' 
                for i, (freq, power) in enumerate(zip(mode_freqs, mode_power))
            ]
        )
        
        # Add original signal
        fig.add_trace(
            go.Scatter(
                x=time_index,
                y=signal_values,
                mode='lines',
                line=dict(color='blue', width=1.5),
                name='Original'
            ),
            row=1, col=1
        )
        
        # Add each mode
        for i in range(modes.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=time_index,
                    y=modes[i],
                    mode='lines',
                    line=dict(width=1.5),
                    name=f'Mode {i+1}'
                ),
                row=i+2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=200 * n_plots,
            width=1200,
            showlegend=False
        )
        
        # Save and open in browser
        html_path = 'vmd_decomposition.html'
        fig.write_html(html_path)
        print(f"Opening VMD decomposition plot in browser...")
        webbrowser.open('file://' + os.path.abspath(html_path))
        
    def create_vmd_features(self, data, column='vst_raw', chunk_size=5000):
        """
        Create VMD-based features for model input
        
        Args:
            data: DataFrame containing the data
            column: Column name to apply VMD to
            chunk_size: Size of chunks to process to avoid memory issues
            
        Returns:
            DataFrame with added VMD features
        """
        if column not in data.columns:
            raise ValueError(f"Column {column} not found in data")
        
        # Create a copy of the data to avoid modifying the original
        enhanced_data = data.copy()
        
        # For large datasets, process in chunks to avoid memory issues
        if len(data) > chunk_size:
            print(f"Large dataset detected ({len(data)} points). Processing in chunks of {chunk_size}...")
            
            # Initialize empty mode columns
            for i in range(self.config['K']):
                enhanced_data[f'vmd_mode_{i+1}'] = np.nan
            
            # Process in chunks
            chunk_count = 0
            for i in range(0, len(data), chunk_size):
                chunk_end = min(i + chunk_size, len(data))
                chunk_count += 1
                print(f"Processing chunk {chunk_count}/{(len(data)-1)//chunk_size + 1} (indices {i}-{chunk_end-1})...")
                
                # Get chunk data
                chunk_signal = data.iloc[i:chunk_end][column]
                
                # Skip if all values are NaN
                if chunk_signal.isna().all():
                    continue
                
                # Apply VMD decomposition to chunk
                try:
                    chunk_result = self.decompose(chunk_signal)
                    
                    # Add each mode as a feature for this chunk
                    for j, mode in enumerate(chunk_result['modes']):
                        enhanced_data.iloc[i:chunk_end, enhanced_data.columns.get_loc(f'vmd_mode_{j+1}')] = mode
                        
                except Exception as e:
                    print(f"Warning: Error processing chunk {i}-{chunk_end-1}: {str(e)}")
                    # Continue with next chunk
            
            return enhanced_data
        else:
            # For smaller datasets, process the entire dataset at once
            # Get the signal
            signal = data[column]
            
            # Apply VMD decomposition
            result = self.decompose(signal)
            
            # Add each mode as a new feature
            for i, mode in enumerate(result['modes']):
                enhanced_data[f'vmd_mode_{i+1}'] = mode
                
            return enhanced_data
    
    def downsample_and_decompose(self, signal, target_length=1000, sample_rate=1.0):
        """
        Downsample a signal to reduce memory requirements, then decompose
        
        Args:
            signal: 1D numpy array or pandas Series containing the signal to decompose
            target_length: Target length after downsampling
            sample_rate: Original sample rate of the signal
            
        Returns:
            Dictionary containing the downsampled modes, which can be upsampled later
        """
        # Convert to numpy array if it's a pandas Series
        if isinstance(signal, pd.Series):
            original_index = signal.index
            signal_values = signal.values
        else:
            signal_values = signal
            original_index = None
            
        # Handle NaN values by linear interpolation
        nan_mask = np.isnan(signal_values)
        if np.any(nan_mask):
            print(f"Warning: {np.sum(nan_mask)} NaN values detected in the signal. Interpolating.")
            signal_values = self._interpolate_nans(signal_values)
        
        # Determine downsampling factor
        original_length = len(signal_values)
        if original_length > target_length:
            # Downsample the signal
            factor = original_length // target_length
            if factor > 1:
                print(f"Downsampling signal from {original_length} to ~{original_length//factor} points (factor: {factor})")
                downsampled = signal_values[::factor]
                
                # Apply VMD to downsampled signal
                result = self.decompose(downsampled, sample_rate/factor)
                result['downsample_factor'] = factor
                result['original_length'] = original_length
                return result
        
        # If signal is already small enough, just decompose normally
        return self.decompose(signal_values, sample_rate)
        
if __name__ == "__main__":
    # Generate a sample water level signal with multiple frequencies and noise
    # This simulates water level data with daily, weekly and monthly patterns
    fs = 24  # 24 samples per day (hourly)
    days = 30  # One month of data
    t = np.arange(0, days, 1/fs)  # Time in days
    
    # Create a composite signal resembling water level data:
    # 1. Daily cycle (frequency = 1 cycle/day)
    # 2. Weekly cycle (frequency = 1/7 cycles/day)
    # 3. Monthly trend (slow-changing component)
    # 4. Random noise
    daily = 50 * np.sin(2 * np.pi * 1 * t)  # Daily cycle
    weekly = 100 * np.sin(2 * np.pi * (1/7) * t)  # Weekly cycle
    monthly = 200 * np.sin(2 * np.pi * (1/30) * t)  # Monthly cycle
    trend = 0.5 * t * t  # Increasing trend
    noise = 20 * np.random.randn(len(t))  # Random noise
    
    # Combine all components
    signal = daily + weekly + monthly + trend + noise
    
    # Create example dataset with the signal
    dates = pd.date_range(start='2023-01-01', periods=len(signal), freq='H')
    data = pd.DataFrame({
        'vst_raw': signal,
        'temperature': 20 + 5 * np.sin(2 * np.pi * (1/24) * np.arange(len(signal))),
        'rainfall': np.random.exponential(0.5, size=len(signal))
    }, index=dates)
    
    # Initialize VMD processor
    vmd = VMD_Processor()
    
    # Decompose the signal
    result = vmd.decompose(data['vst_raw'])
    
    # Plot the decomposition
    vmd.plot_decomposition(data['vst_raw'], result)
    
    # Create enhanced features
    enhanced_data = vmd.create_vmd_features(data)
    
    print(f"Original data shape: {data.shape}")
    print(f"Enhanced data shape: {enhanced_data.shape}")
    print(f"New columns: {[col for col in enhanced_data.columns if col not in data.columns]}") 