# Iterative Anomaly Correction Pipeline Explanation

This document explains the exact functionality of the iterative anomaly correction pipeline for water level data, detailing each component and its step-by-step operation.

## Main Components

The pipeline consists of two primary files:
- `iterative_anomaly_correction.py`: Contains the core correction algorithm
- `visualization_utils.py`: Handles visualization of results

## Detection Function: `detect_anomalies_with_nan_handling`

This function detects anomalies in time series data while handling NaN values:

1. **Input**: 
   - `actual`: Series with actual water level values
   - `predicted`: Series with model-predicted values
   - `contamination`: Expected proportion of anomalies (default: 0.05)
   - `magnitude_threshold`: Minimum relative deviation to flag as anomaly (default: 0.5)
   - `direction_bias`: Directionality focus (-1 for drops, 1 for spikes, 0 for both)

2. **Process**:
   - Calculates residuals (actual - predicted)
   - Creates mask for valid (non-NaN) data points
   - For drop detection (direction_bias < 0):
     - Identifies large negative residuals (actual < predicted)
     - Detects sudden drops in data using first derivative
     - Calculates relative drops compared to local rolling mean
     - Combines conditions (a point is anomalous if it meets any condition)
     - Excludes high values (above 80th percentile) to avoid marking peaks as anomalies
     - Connects nearby anomalies (within 3-point window)
   - For other cases (direction_bias >= 0):
     - Uses Isolation Forest on valid residuals
     - Converts predictions to 0/1 indicator
     - Applies magnitude threshold to filter small deviations
     - Applies direction bias if specified

3. **Output**:
   - DataFrame with columns: actual, predicted, residual, anomaly

## `IterativeAnomalyCorrector` Class

### Initialization
1. Stores configuration and preprocessor
2. Creates a FeatureEngineer instance

### `iterative_correction` Method

This is the core method implementing the iterative approach:

1. **Input**:
   - Training, validation, and test data
   - Max iterations, convergence threshold
   - Contamination rate, smoothing window
   - Magnitude threshold, direction bias
   - Output path for visualizations

2. **Initialization**:
   - Creates output directory if needed
   - Makes copies of input data to avoid modifying originals
   - Adds 'vst_raw_corrected' column to each DataFrame (initially same as original)
   - Initializes correction history and anomaly masks

3. **Iterative Process**:
   - For each iteration (up to max_iterations):
     - **Training**:
       - Temporarily changes output feature to 'vst_raw_corrected'
       - Creates new LSTM model
       - Initializes trainer
       - Uses fewer epochs for intermediate iterations
       - Trains model on current corrected data
     
     - **Prediction**:
       - Makes predictions on test data
       - Flattens predictions to match data length
       - Creates predictions Series with matching index
     
     - **Anomaly Detection**:
       - Detects anomalies between original and predicted values
       - Stores anomaly mask for current iteration
       - Counts and reports detected anomalies
     
     - **Correction**:
       - Stores previous corrected values
       - Replaces anomalous points with predictions (where not NaN)
       - Applies smoothing to corrected segments if smoothing_window > 1
     
     - **Training Data Processing** (from second iteration):
       - Makes predictions on training data
       - Detects anomalies in training data
       - Applies corrections to training data
       - Smooths corrected segments
     
     - **Convergence Check**:
       - Calculates change between current and previous corrections
       - Reports max and mean changes
       - Breaks if mean change < convergence_threshold

4. **Final Processing**:
   - Restores original output feature
   - Creates result DataFrame with actual, corrected, and anomaly values
   - Generates visualizations if output_path provided

5. **Output**:
   - Returns result DataFrame, correction history, and anomaly masks

### `_smooth_corrected_segments` Method

This helper method smooths anomalous segments and their boundaries:

1. **Input**:
   - Data series to smooth
   - Anomaly mask indicating which points to smooth
   - Window size for smoothing
   
2. **Process**:
   - Finds indices where anomalies start and end
   - Handles edge cases (anomalies at series boundaries)
   - Groups adjacent anomaly segments if they're close
   - For each segment:
     - Extends segment to include boundary points
     - For extreme anomalies, uses interpolation between good values
     - Applies centered rolling mean with specified window size
     - Applies exponential smoothing if enough data points
     - Blends smoothed values with original at boundaries using sigmoid weighting

3. **Output**:
   - Returns smoothed series

## Pipeline Function: `run_iterative_correction_pipeline`

This function orchestrates the entire process:

1. **Input**:
   - Configuration, preprocessor
   - Training, validation, and test data
   - Output path, max iterations
   - Algorithm parameters (contamination, smoothing window, etc.)

2. **Process**:
   - Initializes corrector
   - Runs iterative correction
   - Calculates metrics for original and corrected data:
     - Difference statistics (mean, median, max)
     - Anomaly count and percentage
     - Mean and max anomaly differences
   - Calculates RMSE between iterations to show convergence

3. **Output**:
   - Returns results, difference statistics, correction history, and anomaly masks

## Visualization Functions

The pipeline uses several visualization functions from `visualization_utils.py`:

### `visualize_all_results`
Generates all visualizations for the correction process:
1. Iteration convergence plot
2. Detailed correction regions
3. Anomaly evolution across iterations
4. RMSE convergence between iterations
5. Final comparison of original, error-injected, and corrected data

### `plot_iteration_convergence`
Shows how corrections evolve across iterations:
1. Plots original data
2. Plots error-injected data if available
3. Plots each iteration's corrections with color gradient

### `plot_correction_regions`
Shows detailed views of significant correction regions:
1. Identifies regions with large corrections
2. Shows up to 3 significant regions
3. Displays original, error, and correction iterations for each region

### `plot_anomaly_evolution`
Visualizes how anomaly detection changes across iterations:
1. Creates subplot for each iteration
2. Shows anomaly masks as filled regions

### `plot_rmse_convergence`
Shows convergence between iterations:
1. Calculates RMSE between consecutive iterations
2. Plots RMSE values to visualize convergence

### `plot_final_comparison`
Compares original, error-injected, and corrected data:
1. Plots all three data series
2. Highlights detected anomalies

## Complete Pipeline Flow

1. Data is split into training, validation, and test sets
2. Initial model is trained on original training data
3. Iterative process begins:
   - Model predicts values for test data
   - Anomalies detected by comparing actual vs. predicted
   - Anomalous points replaced with model predictions
   - Corrected segments smoothed for better transitions
   - Model retrained on partially corrected data
   - Process repeats until convergence or max iterations
4. Final corrections and anomaly masks saved
5. Visualizations generated to show correction process
6. Performance metrics calculated and reported

## Key Algorithm Features

1. **Direction-Specific Detection**: Focus on negative drops vs. positive spikes
2. **Multiple Detection Approaches**:
   - Statistical deviations (residuals)
   - Sudden changes (first derivative)
   - Relative changes compared to local average
3. **Adaptive Smoothing**:
   - Rolling means
   - Interpolation for extreme anomalies
   - Exponential smoothing
   - Sigmoid-weighted boundary blending
4. **Iterative Refinement**:
   - Each iteration builds on previous corrections
   - Convergence tracking
   - Automatic termination when stable
5. **NaN Handling**: Special processing for missing or invalid data
6. **Connected Segments**: Nearby anomalies grouped into coherent segments 