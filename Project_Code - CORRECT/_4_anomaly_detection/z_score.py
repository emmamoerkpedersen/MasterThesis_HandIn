import numpy as np

def calculate_z_scores(y_true, y_pred, window_size=100, threshold=6.0):
    # Calculate residuals, handling NaN values
    residuals = np.full_like(y_true, np.nan, dtype=np.float32)
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    # Calculate absolute residuals
    residuals[valid_mask] = y_true[valid_mask] - y_pred[valid_mask]
    
    z_scores = np.full_like(residuals, np.nan, dtype=np.float32)
    anomalies = np.zeros_like(residuals, dtype=bool)

    # Flag NaNs in y_true as anomalies
    anomalies[np.isnan(y_true)] = True

    for i in range(window_size, len(residuals)):
        # Get window of valid residuals
        window = residuals[i - window_size:i]
        valid_window = window[~np.isnan(window)]
        
        # Skip if not enough valid points in window
        if len(valid_window) < window_size * 0.5:  # Require at least 50% valid points
            continue
            
        mean = np.mean(valid_window)
        std = np.std(valid_window)

        if std == 0:
            continue  # Avoid division by zero

        # Only calculate z-score for valid residuals
        if not np.isnan(residuals[i]):
            z = (residuals[i] - mean) / std
            z_scores[i] = z

            if abs(z) > threshold:
                anomalies[i] = True

    return z_scores, anomalies
