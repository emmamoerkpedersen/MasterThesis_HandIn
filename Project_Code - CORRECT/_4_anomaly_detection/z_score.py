import numpy as np

def calculate_z_scores(y_true, y_pred, window_size=150, threshold=10.0):
    residuals = y_true - y_pred
    z_scores = np.full_like(residuals, np.nan, dtype=np.float32)
    anomalies = np.zeros_like(residuals, dtype=bool)

    for i in range(window_size, len(residuals)):
        window = residuals[i - window_size:i]
        mean = np.mean(window)
        std = np.std(window)

        if std == 0:
            continue  # Avoid division by zero

        z = (residuals[i] - mean) / std
        z_scores[i] = z

        if abs(z) > threshold:
            anomalies[i] = True

    return z_scores, anomalies
