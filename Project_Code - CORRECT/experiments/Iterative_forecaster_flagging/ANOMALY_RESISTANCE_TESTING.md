# Anomaly Resistance Testing Plan

## Problem Statement
Water level forecasting models trained on data containing synthetic errors tend to **follow and predict anomalous patterns** instead of resisting them. We need models that can:
1. Learn from mixed clean/anomalous training data
2. Resist following anomalous patterns during prediction
3. Stay grounded to normal behavior during sustained anomalies (like multi-week offsets)

## Ideas
- ARIMA? Either as a preliminary step or something like that?
- Z-score MAD to find anomalies and flag them BEFORE training. 
  * Perhaps just implement a "perfect" flagging system first. Flag the known injected anomalies, see if this additional flag information + cross entropy loss function combined with mse would work?
