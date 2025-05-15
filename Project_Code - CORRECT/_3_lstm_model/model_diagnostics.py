import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor

def analyze_individual_residuals(actual, predictions, output_dir=None, station_id='main', features_df=None):
    """
    Create individual plots for each residual analysis component.
    
    Args:
        actual: Series containing actual water level values
        predictions: Series containing predicted water level values
        output_dir: Optional output directory path
        station_id: Station identifier for plot title
        features_df: DataFrame containing additional features like temperature and rainfall
        
    Returns:
        Dictionary of paths to the saved PNG files
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("results/diagnostics")
    
    output_dir = Path(output_dir) / "individual_residuals"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate residuals
    residuals = actual - predictions
    
    # Dictionary to store file paths
    plot_paths = {}
    
    # 1. Time series of residuals
    plt.figure(figsize=(12, 6))
    plt.plot(residuals.index, residuals.values, 'o-', markersize=2, alpha=0.6, color='#1f77b4')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    plt.title(f'Residuals Over Time - Station {station_id}', fontweight='bold')
    plt.ylabel('Residual (Actual - Predicted) [mm]', fontweight='bold')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    time_path = output_dir / f'residuals_time_{station_id}_{timestamp}.png'
    plt.savefig(time_path, dpi=300)
    plt.close()
    plot_paths['time_series'] = time_path
    
    # 2. Histogram of residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='#1f77b4', stat='density')
    plt.title('Distribution of Residuals', fontweight='bold')
    plt.xlabel('Residual [mm]', fontweight='bold')
    plt.ylabel('Density', fontweight='bold')
    
    # Add normal distribution fit
    from scipy import stats
    mu, std = stats.norm.fit(residuals.dropna())
    x = np.linspace(residuals.min(), residuals.max(), 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k--', linewidth=1.5, label=f'Normal: μ={mu:.1f}, σ={std:.1f}')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    plt.legend()
    plt.tight_layout()
    
    hist_path = output_dir / f'residuals_histogram_{station_id}_{timestamp}.png'
    plt.savefig(hist_path, dpi=300)
    plt.close()
    plot_paths['histogram'] = hist_path
    
    # 3. Q-Q plot
    plt.figure(figsize=(10, 6))
    from scipy import stats
    ax = plt.gca()
    stats.probplot(residuals.dropna(), plot=ax)
    plt.title('Q-Q Plot of Residuals', fontweight='bold')
    plt.xlabel('Theoretical Quantiles', fontweight='bold')
    plt.ylabel('Sample Quantiles', fontweight='bold')
    plt.tight_layout()
    
    qq_path = output_dir / f'residuals_qq_{station_id}_{timestamp}.png'
    plt.savefig(qq_path, dpi=300)
    plt.close()
    plot_paths['qq_plot'] = qq_path
    
    # 4. Residuals vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(actual, residuals, alpha=0.3, s=10, color='#1f77b4')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    plt.title('Residuals vs Actual Values', fontweight='bold')
    plt.xlabel('Actual Water Level [mm]', fontweight='bold')
    plt.ylabel('Residual [mm]', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    vs_actual_path = output_dir / f'residuals_vs_actual_{station_id}_{timestamp}.png'
    plt.savefig(vs_actual_path, dpi=300)
    plt.close()
    plot_paths['vs_actual'] = vs_actual_path
    
    # 5. Residuals vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions, residuals, alpha=0.3, s=10, color='#1f77b4')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    plt.title('Residuals vs Predicted Values', fontweight='bold')
    plt.xlabel('Predicted Water Level [mm]', fontweight='bold')
    plt.ylabel('Residual [mm]', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    vs_pred_path = output_dir / f'residuals_vs_predicted_{station_id}_{timestamp}.png'
    plt.savefig(vs_pred_path, dpi=300)
    plt.close()
    plot_paths['vs_predicted'] = vs_pred_path
    
    # 6. Residuals vs features if available
    if features_df is not None and not features_df.empty:
        # Ensure the indices match
        features_aligned = features_df.loc[residuals.index].copy()
        
        # Plot residuals vs temperature if available
        if 'temperature' in features_aligned.columns:
            plt.figure(figsize=(10, 6))
            temp_data = features_aligned['temperature'].dropna()
            res_aligned = residuals.loc[temp_data.index]
            
            plt.scatter(temp_data, res_aligned, alpha=0.3, s=10, color='#1f77b4')
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            plt.title('Residuals vs Temperature', fontweight='bold')
            plt.xlabel('Temperature [°C]', fontweight='bold')
            plt.ylabel('Residual [mm]', fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # Add linear regression line
            if len(temp_data) > 1:
                m, b = np.polyfit(temp_data, res_aligned, 1)
                plt.plot(temp_data, m*temp_data + b, color='green', linestyle='-', linewidth=1.5,
                       label=f'Trend: y = {m:.2f}x + {b:.2f}')
                plt.legend()
            
            plt.tight_layout()
            
            temp_path = output_dir / f'residuals_vs_temp_{station_id}_{timestamp}.png'
            plt.savefig(temp_path, dpi=300)
            plt.close()
            plot_paths['vs_temperature'] = temp_path
        
        # Plot residuals vs rainfall if available
        if 'rainfall' in features_aligned.columns:
            plt.figure(figsize=(10, 6))
            rain_data = features_aligned['rainfall'].dropna()
            res_aligned = residuals.loc[rain_data.index]
            
            # Filter out zero rainfall for better visualization
            nonzero_mask = rain_data > 0
            if nonzero_mask.sum() > 10:  # Only if we have enough non-zero data points
                plt.scatter(rain_data[nonzero_mask], res_aligned[nonzero_mask], 
                       alpha=0.3, s=10, color='#1f77b4')
                plt.xscale('log')  # Log scale for rainfall
                plt.title('Residuals vs Rainfall (non-zero)', fontweight='bold')
            else:
                plt.scatter(rain_data, res_aligned, alpha=0.3, s=10, color='#1f77b4')
                plt.title('Residuals vs Rainfall', fontweight='bold')
            
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            plt.xlabel('Rainfall [mm]', fontweight='bold')
            plt.ylabel('Residual [mm]', fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            rain_path = output_dir / f'residuals_vs_rainfall_{station_id}_{timestamp}.png'
            plt.savefig(rain_path, dpi=300)
            plt.close()
            plot_paths['vs_rainfall'] = rain_path
        
        # Plot residuals vs vst_raw if it's a different column than 'actual'
        if 'vst_raw' in features_aligned.columns and not actual.equals(features_aligned['vst_raw']):
            plt.figure(figsize=(10, 6))
            vst_data = features_aligned['vst_raw'].dropna()
            res_aligned = residuals.loc[vst_data.index]
            
            plt.scatter(vst_data, res_aligned, alpha=0.3, s=10, color='#1f77b4')
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            plt.title('Residuals vs VST Raw', fontweight='bold')
            plt.xlabel('VST Raw [mm]', fontweight='bold')
            plt.ylabel('Residual [mm]', fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # Add linear regression line
            if len(vst_data) > 1:
                m, b = np.polyfit(vst_data, res_aligned, 1)
                plt.plot(vst_data, m*vst_data + b, color='green', linestyle='-', linewidth=1.5,
                       label=f'Trend: y = {m:.2f}x + {b:.2f}')
                plt.legend()
            
            plt.tight_layout()
            
            vst_path = output_dir / f'residuals_vs_vst_{station_id}_{timestamp}.png'
            plt.savefig(vst_path, dpi=300)
            plt.close()
            plot_paths['vs_vst_raw'] = vst_path
    
    print(f"Individual residual plots saved to: {output_dir}")
    return plot_paths

def create_actual_vs_predicted_plot(actual, predictions, output_dir=None, station_id='main'):
    """
    Create a single plot comparing actual vs predicted water levels with additional metrics.
    
    Args:
        actual: Series containing actual water level values
        predictions: Series containing predicted water level values
        output_dir: Optional output directory path
        station_id: Station identifier for plot title
        
    Returns:
        Path to the saved PNG file
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("Project_Code - CORRECT/results/diagnostics")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a new figure
    plt.figure(figsize=(12, 10))
    
    # Create a scatter plot of actual vs predicted
    plt.scatter(actual, predictions, alpha=0.4, s=15, color='#1f77b4')
    
    # Add a perfect prediction line (y=x)
    max_val = max(actual.max(), predictions.max())
    min_val = min(actual.min(), predictions.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Perfect prediction (y=x)')
    
    # Add regression line to show trend
    from sklearn.linear_model import LinearRegression
    import numpy as np
    
    # Filter out NaN values
    mask = ~np.isnan(actual) & ~np.isnan(predictions)
    if sum(mask) > 1:  # Ensure we have at least 2 points for regression
        actual_valid = actual[mask].values.reshape(-1, 1)
        predictions_valid = predictions[mask].values.reshape(-1, 1)
        
        reg = LinearRegression().fit(actual_valid, predictions_valid)
        pred_line = reg.predict(np.array([[min_val], [max_val]]))
        plt.plot([min_val, max_val], [pred_line[0][0], pred_line[1][0]], 'g-', linewidth=1.5, 
                 label=f'Regression line (slope={reg.coef_[0][0]:.2f}, intercept={reg.intercept_[0]:.2f})')
    
    # Add axis labels and title
    plt.xlabel('Actual Water Level [mm]', fontweight='bold', fontsize=12)
    plt.ylabel('Predicted Water Level [mm]', fontweight='bold', fontsize=12)
    plt.title(f'Actual vs Predicted Water Levels - Station {station_id}', fontweight='bold', fontsize=14)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add performance metrics as text box
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actual.dropna(), predictions.loc[actual.dropna().index]))
    mae = mean_absolute_error(actual.dropna(), predictions.loc[actual.dropna().index])
    r2 = r2_score(actual.dropna(), predictions.loc[actual.dropna().index])
    
    # Calculate Nash-Sutcliffe Efficiency (NSE)
    nse = 1 - (np.sum((actual.dropna() - predictions.loc[actual.dropna().index]) ** 2) / 
              np.sum((actual.dropna() - np.mean(actual.dropna())) ** 2))
    
    # Add metrics text box
    stats_text = (
        f"RMSE: {rmse:.2f} mm\n"
        f"MAE: {mae:.2f} mm\n"
        f"R²: {r2:.4f}\n"
        f"NSE: {nse:.4f}"
    )
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
    # Add legend
    plt.legend(loc='lower right')
        
    # Ensure equal aspect ratio
    plt.axis('equal')
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / f'actual_vs_predicted_{station_id}_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved actual vs predicted plot to: {output_path}")
    return output_path

def create_error_heatmap(actual, predictions, output_dir=None, station_id='main'):
    """
    Create a heatmap showing prediction errors by month and day of week.
    
    Args:
        actual: Series containing actual water level values
        predictions: Series containing predicted water level values
        output_dir: Optional output directory path
        station_id: Station identifier for plot title
        
    Returns:
        Path to the saved PNG file
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("Project_Code - CORRECT/results/diagnostics")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate errors
    errors = actual - predictions
    errors_df = pd.DataFrame({'error': errors, 'abs_error': np.abs(errors)})
    
    # Add time features
    errors_df['month'] = errors.index.month
    errors_df['day_of_week'] = errors.index.dayofweek
    errors_df['hour'] = errors.index.hour
    
    # Month-Day of Week Mean Absolute Error
    month_dow_pivot = errors_df.pivot_table(
        values='abs_error', 
        index='month', 
        columns='day_of_week', 
        aggfunc='mean'
    )
    
    # Month labels
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Month-Day of Week heatmap
    sns.heatmap(
        month_dow_pivot, 
        annot=True, 
        fmt='.1f', 
        cmap='YlOrRd', 
        ax=axes[0],
        xticklabels=day_names,
        yticklabels=month_names,
        cbar_kws={'label': 'Mean Absolute Error [mm]'}
    )
    axes[0].set_title('Mean Absolute Error by Month and Day of Week', fontweight='bold')
    axes[0].set_xlabel('Day of Week', fontweight='bold')
    axes[0].set_ylabel('Month', fontweight='bold')
    
    # Create month-only error boxplot
    month_errors = errors_df.groupby('month')['error'].apply(list)
    
    # Plot boxplot of errors by month
    month_plot = axes[1]
    
    # Convert to format needed for boxplot
    month_data = [month_errors[m] for m in range(1, 13)]
    
    # Create boxplot
    boxplot = month_plot.boxplot(
        month_data, 
        labels=month_names,
        patch_artist=True,
        medianprops={'color': 'black'},
        flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 3}
    )
    
    # Customize boxplot colors - gradient from green to red
    colors = plt.cm.YlOrRd(np.linspace(0.1, 0.8, 12))
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
    
    month_plot.set_title('Error Distribution by Month', fontweight='bold')
    month_plot.set_xlabel('Month', fontweight='bold')
    month_plot.set_ylabel('Error (Actual - Predicted) [mm]', fontweight='bold')
    month_plot.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    month_plot.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels
    plt.setp(month_plot.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / f'error_heatmap_{station_id}_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved error heatmap to: {output_path}")
    return output_path

def create_feature_importance_plot(test_data, predictions, feature_cols, output_dir=None, station_id='main'):
    """
    Create a feature importance plot using permutation importance method.
    
    Args:
        test_data: DataFrame containing test data with features and target
        predictions: Series containing predicted water level values
        feature_cols: List of feature column names
        output_dir: Optional output directory path
        station_id: Station identifier for plot title
        
    Returns:
        Path to the saved PNG file
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("Project_Code - CORRECT/results/diagnostics")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract the actual values and prepare features DataFrame
    actual = test_data['vst_raw'] if 'vst_raw' in test_data.columns else None
    if actual is None:
        print("Warning: 'vst_raw' not found in test_data. Feature importance cannot be calculated.")
        return None
    
    # Keep only the feature columns that exist in the data
    valid_feature_cols = [col for col in feature_cols if col in test_data.columns]
    if not valid_feature_cols:
        print("Warning: No valid feature columns found. Feature importance cannot be calculated.")
        return None
    
    # Create features DataFrame
    features = test_data[valid_feature_cols]
    
    # Handle missing values if any
    features = features.fillna(method='ffill').fillna(0)
    
    # Get unique indices where both actual and predictions have valid values
    common_idx = features.index.intersection(actual.index).intersection(predictions.index)
    if len(common_idx) < 10:  # Need enough data for meaningful analysis
        print("Warning: Not enough common data points for feature importance analysis.")
        return None
    
    # Align data on common indices
    X = features.loc[common_idx].values
    y_actual = actual.loc[common_idx].values
    y_pred = predictions.loc[common_idx].values
    
    # Check for NaN or infinite values
    if np.isnan(X).any() or np.isinf(X).any() or np.isnan(y_actual).any() or np.isinf(y_actual).any():
        print("Warning: Data contains NaN or infinite values. Cleaning data for analysis.")
        # Create mask for valid data points
        valid_mask = ~np.isnan(y_actual) & ~np.isinf(y_actual) & ~np.any(np.isnan(X) | np.isinf(X), axis=1)
        X = X[valid_mask]
        y_actual = y_actual[valid_mask]
        y_pred = y_pred[valid_mask]
    
    if len(y_actual) < 10:
        print("Warning: Not enough valid data points after cleaning for feature importance analysis.")
        return None
    
    # Calculate error between actual and predictions (this will be our target)
    errors = np.abs(y_actual - y_pred)
    
    # Train a simple Random Forest model to predict the errors
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    try:
        rf.fit(X, errors)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(rf, X, errors, n_repeats=10, random_state=42)
        
        # Sort features by importance
        sorted_idx = perm_importance.importances_mean.argsort()
        
        # Get feature names
        feature_names = [valid_feature_cols[i] for i in sorted_idx]
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in range(len(sorted_idx))])
        plt.xlabel('Permutation Importance for Prediction Error')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance - Station {station_id}', fontweight='bold')
        plt.tight_layout()
        
        # Save the figure
        output_path = output_dir / f'feature_importance_{station_id}_{timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved feature importance plot to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error calculating feature importance: {e}")
        return None

def create_correlation_analysis(test_data, predictions, feature_cols, output_dir=None, station_id='main'):
    """
    Create correlation analysis plots for features, target, and prediction errors.
    
    Args:
        test_data: DataFrame containing test data with features and target
        predictions: Series containing predicted water level values
        feature_cols: List of feature column names
        output_dir: Optional output directory path
        station_id: Station identifier for plot title
        
    Returns:
        Path to the saved PNG file
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("Project_Code - CORRECT/results/diagnostics")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract the actual values
    actual = test_data['vst_raw'] if 'vst_raw' in test_data.columns else None
    if actual is None:
        print("Warning: 'vst_raw' not found in test_data. Correlation analysis cannot be performed.")
        return None
    
    # Keep only the feature columns that exist in the data
    valid_feature_cols = [col for col in feature_cols if col in test_data.columns]
    if not valid_feature_cols:
        print("Warning: No valid feature columns found. Correlation analysis cannot be performed.")
        return None
    
    # Create a combined DataFrame with features, actual, predictions, and errors
    df_corr = test_data[valid_feature_cols].copy()
    df_corr['Actual'] = actual
    
    # Add predictions to the correlation DataFrame
    prediction_series = pd.Series(predictions, index=test_data.index)
    df_corr['Predicted'] = prediction_series
    
    # Calculate errors
    df_corr['Error'] = df_corr['Actual'] - df_corr['Predicted']
    df_corr['Abs_Error'] = df_corr['Error'].abs()
    
    # Handle missing values
    df_corr = df_corr.fillna(method='ffill').fillna(0)
    
    # Create correlation matrix
    corr_matrix = df_corr.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr_matrix, 
        mask=mask, 
        cmap=cmap, 
        vmax=.3, 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .5},
        annot=True,
        fmt=".2f"
    )
    
    plt.title(f'Feature Correlation Analysis - Station {station_id}', fontweight='bold')
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / f'correlation_analysis_{station_id}_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a separate correlation plot for errors and features only
    error_cols = ['Error', 'Abs_Error']
    error_corr = corr_matrix.loc[error_cols, [col for col in corr_matrix.columns if col not in error_cols]]
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        error_corr,
        cmap='coolwarm',
        annot=True,
        fmt=".2f",
        linewidths=.5
    )
    plt.title(f'Error Correlation with Features - Station {station_id}', fontweight='bold')
    plt.tight_layout()
    
    # Save the error correlation figure
    error_output_path = output_dir / f'error_correlation_{station_id}_{timestamp}.png'
    plt.savefig(error_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlation analysis plots to: {output_path} and {error_output_path}")
    return {'correlation': output_path, 'error_correlation': error_output_path}

def generate_all_diagnostics(actual, predictions, output_dir=None, station_id='main', rainfall=None, features_df=None):
    """
    Generate a simplified set of diagnostic visualizations for a water level prediction model.
    
    Args:
        actual: Series containing actual water level values
        predictions: Series containing predicted water level values
        output_dir: Optional output directory path
        station_id: Station identifier for plot titles
        rainfall: Optional series of rainfall data for the same period
        features_df: DataFrame containing additional features for residual analysis
        
    Returns:
        dict: Dictionary with paths to all generated visualization files
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("Project_Code - CORRECT/results/diagnostics")
    
    # Make sure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store all visualization paths
    visualization_paths = {}
    
    # Generate timestamp for logging
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] Generating diagnostic visualizations for station {station_id}...")
    
    try:
        # 1. Generate individual residual plots
        print("  - Creating individual residual plots...")
        
        # Prepare features dataframe if rainfall is available but no features_df is provided
        if features_df is None and rainfall is not None:
            features_df = pd.DataFrame({'rainfall': rainfall})
            if 'vst_raw' not in features_df.columns:
                features_df['vst_raw'] = actual
        
        residual_plots = analyze_individual_residuals(
            actual=actual,
            predictions=predictions,
            output_dir=output_dir,
            station_id=station_id,
            features_df=features_df
        )
        visualization_paths['residual_plots'] = residual_plots
        
        # 2. Create actual vs predicted plot
        print("  - Creating actual vs predicted plot...")
        actual_pred_path = create_actual_vs_predicted_plot(
                actual=actual,
                predictions=predictions,
                output_dir=output_dir,
                station_id=station_id
            )
        visualization_paths['actual_vs_predicted'] = actual_pred_path
        
        # 3. Create error heatmap by time patterns
        print("  - Creating error pattern heatmap...")
        heatmap_path = create_error_heatmap(
            actual=actual,
            predictions=predictions,
            output_dir=output_dir,
            station_id=station_id
        )
        visualization_paths['error_heatmap'] = heatmap_path
        
        # 4. Create feature importance plot
        print("  - Creating feature importance plot...")
        feature_importance_path = create_feature_importance_plot(
            test_data=pd.DataFrame({'vst_raw': actual, **features_df}),
            predictions=predictions,
            feature_cols=features_df.columns,
            output_dir=output_dir,
            station_id=station_id
        )
        visualization_paths['feature_importance'] = feature_importance_path
        
        # 5. Create correlation analysis plots
        print("  - Creating correlation analysis plots...")
        correlation_analysis_paths = create_correlation_analysis(
            test_data=pd.DataFrame({'vst_raw': actual, **features_df}),
            predictions=predictions,
            feature_cols=features_df.columns,
            output_dir=output_dir,
            station_id=station_id
        )
        visualization_paths['correlation_analysis'] = correlation_analysis_paths
        
        # Success message
        print(f"\nAll diagnostic visualizations saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during diagnostic visualization generation: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return visualization_paths

def generate_comparative_diagnostics(actual, predictions_dict, output_dir=None, station_id='main', rainfall=None, features_df=None):
    """
    Generate comparison diagnostic visualizations between multiple model predictions.
    
    Args:
        actual: Series containing actual water level values
        predictions_dict: Dictionary with model names as keys and prediction Series as values
        output_dir: Optional output directory path
        station_id: Station identifier
        rainfall: Optional series of rainfall data for the same period
        features_df: DataFrame containing additional features for residual analysis
        
    Returns:
        dict: Dictionary with paths to all generated visualization files
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("Project_Code - CORRECT/results/diagnostics")
    
    # Make sure output directory exists
    output_dir = Path(output_dir)
    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store all visualization paths for all models
    all_visualization_paths = {}
    
    # Generate timestamp for logging
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] Generating comparative diagnostic visualizations for {len(predictions_dict)} models...")
    
    try:
        # Generate individual diagnostics for each model
        for model_name, predictions in predictions_dict.items():
            print(f"\nAnalyzing model: {model_name}")
            model_dir = output_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            visualization_paths = generate_all_diagnostics(
                actual=actual,
                predictions=predictions,
                output_dir=model_dir,
                station_id=f"{station_id}_{model_name}",
                rainfall=rainfall,
                features_df=features_df
            )
            
            all_visualization_paths[model_name] = visualization_paths
        
        print(f"\nComparative analysis complete. Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during comparative diagnostic visualization: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return all_visualization_paths
