"""
Integration of Variational Mode Decomposition (VMD) with LSTM model
for improved water level prediction
"""
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import webbrowser

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import project modules
from experiments.Improved_model_structure.vmd_utils import VMD_Processor
from experiments.Improved_model_structure.train_model import DataPreprocessor, LSTM_Trainer
from experiments.Improved_model_structure.model import LSTMModel
from experiments.Improved_model_structure.model_plots import create_full_plot, plot_convergence
from experiments.Improved_model_structure.config import LSTM_CONFIG

def run_vmd_enhanced_pipeline(project_root, station_id='21006846'):
    """
    Run a pipeline that demonstrates VMD-enhanced LSTM model for water level prediction
    
    Args:
        project_root: Path to project root
        station_id: Station ID to process
    """
    print("\n" + "="*80)
    print("Running VMD-Enhanced LSTM Pipeline for Water Level Prediction")
    print("="*80)
    
    # Load configuration and initialize preprocessor
    model_config = LSTM_CONFIG.copy()
    preprocessor = DataPreprocessor(model_config)
    
    # 1. Load and preprocess data
    print(f"\n1. Loading and preprocessing data for station {station_id}...")
    train_data, val_data, test_data = preprocessor.load_and_split_data(project_root, station_id)
    
    # 2. Apply VMD to extract features
    print("\n2. Applying VMD to extract modal features...")
    vmd = VMD_Processor({
        'K': 3,         # Reduce to 3 modes to save memory
        'alpha': 2000,  # Bandwidth constraint
        'tau': 0,       # No noise tolerance
        'tol': 1e-6     # Slightly relaxed tolerance for speed
    })
    
    # 2.1 Visualize VMD decomposition on a small sample
    # Use a 2-day segment for visualization (smaller to save memory)
    sample_data = val_data.iloc[-48:].copy()  # 2 days * 24 hours
    print(f"Visualizing VMD decomposition on {len(sample_data)} sample points...")
    
    # Drop NaN rows for visualization
    sample_data = sample_data.dropna(subset=['vst_raw'])
    
    # Skip visualization if not enough valid data
    if len(sample_data) > 10:
        result = vmd.decompose(sample_data['vst_raw'])
        vmd.plot_decomposition(sample_data['vst_raw'], result, 
                              f"VMD Decomposition of Water Level Data - Station {station_id}")
    else:
        print("Not enough valid data for visualization. Skipping...")
    
    # 2.2 Enhance data with VMD modes using optimized chunking
    print("\nEnhancing data with VMD features using chunked processing...")
    
    # Process in chunks to avoid memory issues - use a manageable chunk size
    chunk_size = 2000  # Set to a reasonable size based on available memory
    
    # Create copies of the datasets
    enhanced_train = train_data.copy()
    enhanced_val = val_data.copy()
    enhanced_test = test_data.copy()
    
    # List of input features to apply VMD to (excluding target variable)
    input_features_for_vmd = [
        'rainfall', 
        'temperature',
        'feature_station_21006845_vst_raw',
        'feature_station_21006845_rainfall',
        'feature_station_21006847_vst_raw',
        'feature_station_21006847_rainfall'
    ]
    
    # Apply VMD only to input features to avoid data leakage
    print("\nApplying VMD only to input features to avoid target data leakage...")
    try:
        for feature in input_features_for_vmd:
            if feature in train_data.columns:
                print(f"\nProcessing feature: {feature}")
                
                # Process validation data
                print("  - Processing validation data...")
                if not val_data[feature].isna().all():
                    val_feature_vmd = vmd.create_vmd_features(val_data, column=feature, chunk_size=chunk_size)
                    # Add VMD modes of this feature to enhanced data
                    for i in range(vmd.config['K']):
                        mode_name = f'vmd_{feature}_mode_{i+1}'
                        enhanced_val[mode_name] = val_feature_vmd[f'vmd_mode_{i+1}']
                        
                # Process test data
                print("  - Processing test data...")
                if not test_data[feature].isna().all():
                    test_feature_vmd = vmd.create_vmd_features(test_data, column=feature, chunk_size=chunk_size)
                    # Add VMD modes of this feature to enhanced data
                    for i in range(vmd.config['K']):
                        mode_name = f'vmd_{feature}_mode_{i+1}'
                        enhanced_test[mode_name] = test_feature_vmd[f'vmd_mode_{i+1}']
                
                # Process training data
                print("  - Processing training data...")
                if not train_data[feature].isna().all():
                    train_feature_vmd = vmd.create_vmd_features(train_data, column=feature, chunk_size=chunk_size)
                    # Add VMD modes of this feature to enhanced data
                    for i in range(vmd.config['K']):
                        mode_name = f'vmd_{feature}_mode_{i+1}'
                        enhanced_train[mode_name] = train_feature_vmd[f'vmd_mode_{i+1}']
                        
                        # Add the new feature to the feature columns list
                        if mode_name not in preprocessor.feature_cols:
                            preprocessor.feature_cols.append(mode_name)
                        
        print("\nVMD feature extraction complete.")
                        
    except MemoryError as e:
        print(f"Memory error: {str(e)}")
        # Implement the memory-saving approach as a fallback
        print("Falling back to processing only most important features...")
        
        # Process only the most important input features
        important_features = ['rainfall', 'feature_station_21006845_vst_raw', 'feature_station_21006847_vst_raw']
        
        for feature in important_features:
            if feature in train_data.columns:
                print(f"\nProcessing important feature: {feature}")
                
                # Process only validation data (often smaller)
                if not val_data[feature].isna().all():
                    val_feature_vmd = vmd.create_vmd_features(val_data, column=feature, chunk_size=1000)
                    # Add only the most significant mode
                    mode_name = f'vmd_{feature}_mode_1'
                    enhanced_val[mode_name] = val_feature_vmd['vmd_mode_1']
                    enhanced_train[mode_name] = np.nan  # Initialize in training data
                    enhanced_test[mode_name] = np.nan   # Initialize in test data
                    
                    # Add to feature columns
                    if mode_name not in preprocessor.feature_cols:
                        preprocessor.feature_cols.append(mode_name)
                
                # For training, just process a small representative sample and fill in
                print("  - Processing sample from training data...")
                sample_train = train_data.sample(min(len(train_data), 1000))
                
                if not sample_train[feature].isna().all():
                    sample_vmd = vmd.create_vmd_features(sample_train, column=feature, chunk_size=1000)
                    # Get mode and index
                    mode_name = f'vmd_{feature}_mode_1'
                    sample_indices = sample_vmd.index
                    
                    # Copy to main datasets
                    enhanced_train.loc[sample_indices, mode_name] = sample_vmd['vmd_mode_1']
                    
                    # Fill missing values
                    enhanced_train[mode_name] = enhanced_train[mode_name].interpolate(method='linear').ffill().bfill()
                    enhanced_val[mode_name] = enhanced_val[mode_name].fillna(0)
                    enhanced_test[mode_name] = enhanced_test[mode_name].fillna(0)
    
    # Print feature counts
    print(f"\nOriginal feature count: {len(model_config['feature_cols'])}")
    print(f"Enhanced feature count: {len(preprocessor.feature_cols)}")
    print(f"Added VMD features: {[col for col in preprocessor.feature_cols if col.startswith('vmd_')]}")
    
    # Check for and fix any NaN values in the VMD features
    print("\nChecking for NaN values in VMD features...")
    for dataset, name in [(enhanced_train, "train"), (enhanced_val, "validation"), (enhanced_test, "test")]:
        # Get only VMD feature columns
        vmd_columns = [col for col in dataset.columns if col.startswith('vmd_')]
        for col in vmd_columns:
            nan_count = dataset[col].isna().sum()
            if nan_count > 0:
                print(f"  - Found {nan_count} NaN values in {name} {col} ({nan_count/len(dataset)*100:.2f}%)")
                # Fill NaN values with 0 (reasonable default for VMD modes)
                dataset[col] = dataset[col].fillna(0)
    
    # Analysis of feature relevance
    print("\nAnalyzing importance of input features and their VMD components...")
    # Calculate correlation of input features and their VMD modes with target
    for dataset_name, dataset in [("train", enhanced_train), ("validation", enhanced_val)]:
        print(f"\nCorrelation with target in {dataset_name} data:")
        
        # Only show correlations for data points where target exists
        valid_indices = dataset['vst_raw'].dropna().index
        if len(valid_indices) > 100:  # Only calculate if enough data
            valid_data = dataset.loc[valid_indices]
            
            # Calculate correlations with target
            correlations = []
            for col in preprocessor.feature_cols:
                if col in valid_data.columns:
                    corr = valid_data[col].corr(valid_data['vst_raw'])
                    correlations.append((col, corr))
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Print top 10 correlations
            print("\nTop 10 features by correlation with target:")
            for feature, corr in correlations[:10]:
                print(f"  {feature:<40}: {corr:>8.4f}")
            
            # Check if any VMD features are in top correlations
            vmd_in_top = [f for f, _ in correlations[:10] if f.startswith('vmd_')]
            if vmd_in_top:
                print(f"\n{len(vmd_in_top)} VMD features are among top 10 correlated with target.")
            else:
                print("\nNo VMD features among top 10 correlated with target.")
    
    # 3. Train LSTM model with enhanced features
    print("\n3. Training LSTM model with VMD-enhanced features...")
    
    # Get updated input size from feature columns
    input_size = len(preprocessor.feature_cols)
    
    # Add gradient clipping and adjust model parameters for stability
    model_config['grad_clip_value'] = 1.0  # Reduce from default
    model_config['learning_rate'] = 0.0005  # Lower learning rate for stability
    model_config['batch_size'] = 64  # Smaller batch size
    model_config['epochs'] = 25
    model_config['patience'] = 5
    # Create the VMD-enhanced model
    model = LSTMModel(
        input_size=input_size,
        sequence_length=model_config['sequence_length'],
        hidden_size=model_config['hidden_size'],
        output_size=len(model_config['output_features']),
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    )
    
    # Apply proper weight initialization for numerical stability
    def init_weights(m):
        if isinstance(m, torch.nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
    
    # Initialize weights for better stability
    model.apply(init_weights)
    
    # Initialize trainer with enhanced model
    trainer = LSTM_Trainer(model_config, preprocessor=preprocessor)
    
    # Print model configuration
    print("\nVMD-Enhanced Model Configuration:")
    print(f"Input Size: {input_size}")
    print(f"Hidden Size: {model_config['hidden_size']}")
    print(f"Number of Layers: {model_config['num_layers']}")
    print(f"Dropout Rate: {model_config['dropout']}")
    print(f"Learning Rate: {model_config.get('learning_rate', 0.001)}")
    print(f"Batch Size: {model_config['batch_size']}")
    print(f"VMD Modes: {vmd.config['K']}")
    
    # Train the model
    try:
        history, val_predictions, val_targets = trainer.train(
            train_data=enhanced_train,
            val_data=enhanced_val,
            epochs=model_config['epochs'],
            batch_size=model_config['batch_size'],
            patience=model_config['patience']
        )
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        print("Trying with reduced complexity...")
        
        # Fall back to a simpler model
        model_config['hidden_size'] = 16  # Reduce from 32
        model_config['learning_rate'] = 0.0001  # Even lower learning rate
        model_config['batch_size'] = 32  # Even smaller batch
        
        # Create a simpler model
        simple_model = LSTMModel(
            input_size=input_size,
            sequence_length=model_config['sequence_length'],
            hidden_size=model_config['hidden_size'],
            output_size=len(model_config['output_features']),
            num_layers=1,  # Force to 1 layer
            dropout=0.1    # Lower dropout
        )
        
        # Re-initialize trainer with simpler model
        simple_trainer = LSTM_Trainer(model_config, preprocessor=preprocessor)
        print("\nFallback to simpler model:")
        print(f"Hidden Size: {model_config['hidden_size']}")
        print(f"Learning Rate: {model_config['learning_rate']}")
        print(f"Batch Size: {model_config['batch_size']}")
        
        # Train the simpler model with gradient monitoring
        history, val_predictions, val_targets = simple_trainer.train(
            train_data=enhanced_train,
            val_data=enhanced_val,
            epochs=model_config['epochs'],
            batch_size=model_config['batch_size'],
            patience=model_config['patience']
        )
    
    # 4. Make and evaluate predictions
    print("\n4. Making predictions and evaluating model...")
    
    # Convert validation predictions to numpy and handle properly
    val_predictions = val_predictions.cpu().numpy()
    val_targets = val_targets.cpu().numpy()
    
    # Transform back to original scale
    predictions_reshaped = val_predictions.reshape(-1, 1)
    predictions_original = preprocessor.scalers['target'].inverse_transform(predictions_reshaped)
    predictions_flattened = predictions_original.flatten()
    
    # Trim predictions to match validation data length
    predictions_flattened = predictions_flattened[:len(val_data)]
    
    # Create DataFrame with aligned predictions and targets
    val_predictions_df = pd.DataFrame(
        predictions_flattened,
        index=val_data.index,
        columns=['vst_raw']
    )
    
    # Create comparison plot
    create_full_plot(val_data, val_predictions_df, str(station_id), model_config)
    
    # Plot training convergence
    plot_convergence(history, str(station_id), title=f"VMD-Enhanced Training - Station {station_id}")
    
    # 5. Compare with baseline model (without VMD)
    print("\n5. Training baseline model (without VMD) for comparison...")
    
    # Reset preprocessor with original features
    baseline_preprocessor = DataPreprocessor(model_config)
    
    # Create baseline model
    baseline_model = LSTMModel(
        input_size=len(baseline_preprocessor.feature_cols),
        sequence_length=model_config['sequence_length'],
        hidden_size=model_config['hidden_size'],
        output_size=len(model_config['output_features']),
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    )
    
    # Initialize baseline trainer
    baseline_trainer = LSTM_Trainer(model_config, preprocessor=baseline_preprocessor)
    
    # Train baseline model with same parameters
    baseline_history, baseline_val_predictions, baseline_val_targets = baseline_trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=model_config['epochs'],
        batch_size=model_config['batch_size'],
        patience=model_config['patience']
    )
    
    # Convert baseline predictions to original scale
    baseline_val_predictions = baseline_val_predictions.cpu().numpy()
    baseline_predictions_reshaped = baseline_val_predictions.reshape(-1, 1)
    baseline_predictions_original = baseline_preprocessor.scalers['target'].inverse_transform(baseline_predictions_reshaped)
    baseline_predictions_flattened = baseline_predictions_original.flatten()
    
    # Trim baseline predictions
    baseline_predictions_flattened = baseline_predictions_flattened[:len(val_data)]
    
    # Create DataFrame with aligned baseline predictions
    baseline_val_predictions_df = pd.DataFrame(
        baseline_predictions_flattened,
        index=val_data.index,
        columns=['vst_raw']
    )
    
    # Plot baseline results
    create_full_plot(val_data, baseline_val_predictions_df, f"{station_id}_baseline", model_config)
    
    # 6. Compare results
    print("\n6. Comparing baseline vs VMD-enhanced models...")
    
    # Calculate performance metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Get actual values
    actual = val_data['vst_raw'].dropna().values
    
    # Get predictions aligned with actual values
    vmd_preds = val_predictions_df.loc[val_data['vst_raw'].dropna().index]['vst_raw'].values
    baseline_preds = baseline_val_predictions_df.loc[val_data['vst_raw'].dropna().index]['vst_raw'].values
    
    # Calculate metrics
    vmd_mse = mean_squared_error(actual, vmd_preds)
    vmd_rmse = np.sqrt(vmd_mse)
    vmd_mae = mean_absolute_error(actual, vmd_preds)
    vmd_r2 = r2_score(actual, vmd_preds)
    
    baseline_mse = mean_squared_error(actual, baseline_preds)
    baseline_rmse = np.sqrt(baseline_mse)
    baseline_mae = mean_absolute_error(actual, baseline_preds)
    baseline_r2 = r2_score(actual, baseline_preds)
    
    # Create comparison results
    print("\nPerformance Comparison:")
    print(f"{'Metric':<10} {'Baseline':<15} {'VMD-Enhanced':<15} {'Improvement':<15}")
    print("-" * 55)
    print(f"{'RMSE':<10} {baseline_rmse:<15.2f} {vmd_rmse:<15.2f} {(baseline_rmse-vmd_rmse)/baseline_rmse*100:<15.2f}%")
    print(f"{'MAE':<10} {baseline_mae:<15.2f} {vmd_mae:<15.2f} {(baseline_mae-vmd_mae)/baseline_mae*100:<15.2f}%")
    print(f"{'R²':<10} {baseline_r2:<15.4f} {vmd_r2:<15.4f} {(vmd_r2-baseline_r2)/abs(baseline_r2)*100:<15.2f}%")
    
    # Create a comparison plot
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Use a 2-week segment for visualization
    comparison_period = val_data.iloc[-336:].index  # 14 days
    
    fig = make_subplots(
        rows=2, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            "Baseline vs VMD-Enhanced Predictions",
            "Prediction Error Comparison"
        ]
    )
    
    # Add actual values
    fig.add_trace(
        go.Scatter(
            x=comparison_period,
            y=val_data.loc[comparison_period, 'vst_raw'],
            mode='lines',
            name='Actual',
            line=dict(color='black', width=2)
        ),
        row=1, col=1
    )
    
    # Add baseline predictions
    fig.add_trace(
        go.Scatter(
            x=comparison_period,
            y=baseline_val_predictions_df.loc[comparison_period, 'vst_raw'],
            mode='lines',
            name='Baseline',
            line=dict(color='blue', width=1.5)
        ),
        row=1, col=1
    )
    
    # Add VMD-enhanced predictions
    fig.add_trace(
        go.Scatter(
            x=comparison_period,
            y=val_predictions_df.loc[comparison_period, 'vst_raw'],
            mode='lines',
            name='VMD-Enhanced',
            line=dict(color='red', width=1.5)
        ),
        row=1, col=1
    )
    
    # Calculate errors
    baseline_error = np.abs(val_data.loc[comparison_period, 'vst_raw'] - 
                          baseline_val_predictions_df.loc[comparison_period, 'vst_raw'])
    vmd_error = np.abs(val_data.loc[comparison_period, 'vst_raw'] - 
                      val_predictions_df.loc[comparison_period, 'vst_raw'])
    
    # Add error plots
    fig.add_trace(
        go.Scatter(
            x=comparison_period,
            y=baseline_error,
            mode='lines',
            name='Baseline Error',
            line=dict(color='blue', width=1.5)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=comparison_period,
            y=vmd_error,
            mode='lines',
            name='VMD-Enhanced Error',
            line=dict(color='red', width=1.5)
        ),
        row=2, col=1
    )
    
    # Add annotations with metrics
    fig.add_annotation(
        x=0.5,
        y=1.12,
        xref="paper",
        yref="paper",
        text=f"Baseline RMSE: {baseline_rmse:.2f} | VMD-Enhanced RMSE: {vmd_rmse:.2f} | Improvement: {(baseline_rmse-vmd_rmse)/baseline_rmse*100:.2f}%",
        showarrow=False,
        font=dict(size=14),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    # Update layout
    fig.update_layout(
        title=f"Baseline vs VMD-Enhanced Model Comparison - Station {station_id}",
        height=800,
        width=1200,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Save and open in browser
    html_path = f'vmd_comparison_{station_id}.html'
    fig.write_html(html_path)
    print(f"\nOpening comparison plot in browser...")
    webbrowser.open('file://' + os.path.abspath(html_path))
    
    return {
        'vmd_metrics': {
            'rmse': vmd_rmse,
            'mae': vmd_mae,
            'r2': vmd_r2
        },
        'baseline_metrics': {
            'rmse': baseline_rmse,
            'mae': baseline_mae,
            'r2': baseline_r2
        },
        'improvement': {
            'rmse': (baseline_rmse-vmd_rmse)/baseline_rmse*100,
            'mae': (baseline_mae-vmd_mae)/baseline_mae*100,
            'r2': (vmd_r2-baseline_r2)/abs(baseline_r2)*100
        }
    }

if __name__ == "__main__":
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    
    # Run VMD-enhanced pipeline
    results = run_vmd_enhanced_pipeline(project_root) 