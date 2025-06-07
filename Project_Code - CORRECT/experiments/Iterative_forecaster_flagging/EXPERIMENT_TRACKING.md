# Anomaly Flagging Experiments - Clean Systematic Testing

## 🎯 **Objective**
Test each component incrementally to show the contribution of:
1. **Baseline**: Just anomaly flags as input features
2. **Experiment 1**: + Anomaly aware loss function
3. **Experiment 2**: + Memory protection mechanism  
4. **Experiment 3**: + Lagged VST features

---

## 📋 **Experiment Configurations**

### **BASELINE: Anomaly Flags Only**
**Status**: ❌ **NEEDS TO BE RUN**
**Command**: `--experiment baseline_anomaly_flags_only`
**Config**:
```python
'use_anomaly_flags': True,        # Enable anomaly flagging as input feature
'use_weighted_loss': False,       # Standard MAE loss
'use_memory_protection': False,   # No memory protection
'use_lagged_features': False,     # No lagged features
'use_perfect_flags': True,        # Use perfect flags from synthetic errors
```
**What This Tests**: Just having anomaly flags as input features with standard training

**📊 RESULTS:**
```
Validation Metrics (vs original clean data):
  rmse: [PASTE RESULTS HERE]
  mae: [PASTE RESULTS HERE]
  nse: [PASTE RESULTS HERE]

Validation Metrics (vs corrupted training data):
  rmse: [PASTE RESULTS HERE]
  mae: [PASTE RESULTS HERE]
  nse: [PASTE RESULTS HERE]
```

---

### **EXPERIMENT 1: Anomaly Aware Loss**
**Status**: ✅ **COMPLETED**
**Command**: `--experiment exp1_anomaly_loss_only`
**Config**:
```python
'use_anomaly_flags': True,        # Enable anomaly flagging as input feature
'use_weighted_loss': True,        # ADDED: Weighted loss (0.3 for anomalies)
'anomaly_weight': 0.3,            # Weight for anomalous periods
'use_memory_protection': False,   # No memory protection
'use_lagged_features': False,     # No lagged features
'use_perfect_flags': True,        # Use perfect flags from synthetic errors
```
**What This Tests**: Impact of anomaly-aware loss function (weighted loss + pattern penalty)

**📊 RESULTS:**
```
Validation Metrics (vs original clean data):
  rmse: 51.076322
  mae: 35.802371
  nse: 0.954814

Validation Metrics (vs corrupted training data):
  rmse: 43.625161
  mae: 32.140973
  nse: 0.967561
```

---

### **EXPERIMENT 2: + Memory Protection**
**Status**: ✅ **ABOUT TO RUN**
**Command**: `--experiment exp2_loss_plus_memory`
**Config**:
```python
'use_anomaly_flags': True,        # Enable anomaly flagging as input feature
'use_weighted_loss': True,        # Weighted loss (0.3 for anomalies)
'anomaly_weight': 0.3,            # Weight for anomalous periods
'use_memory_protection': True,    # ADDED: Memory protection mechanism
'use_lagged_features': False,     # No lagged features
'use_perfect_flags': True,        # Use perfect flags from synthetic errors
```
**What This Tests**: Combined impact of anomaly-aware loss + memory protection

**📊 RESULTS:**
```
📊 FLAGGING MODEL RESULTS:
Validation Metrics (vs original clean data):
  rmse: 43.444937
  mae: 32.307560
  nse: 0.967308

Validation Metrics (vs corrupted training data):
  rmse: 51.774879
  mae: 35.902963
  nse: 0.954309
```

---

### **EXPERIMENT 3: + Lagged Features**
**Status**: ❌ **WAITING FOR EXP 2**
**Command**: `--experiment exp3_full_system`
**Config**:
```python
'use_anomaly_flags': True,        # Enable anomaly flagging as input feature
'use_weighted_loss': True,        # Weighted loss (0.3 for anomalies)
'anomaly_weight': 0.3,            # Weight for anomalous periods
'use_memory_protection': True,    # Memory protection mechanism
'use_lagged_features': True,      # ADDED: Lagged VST features
'lag_hours': [24, 48, 72, 168, 336, 672],  # 1d, 2d, 3d, 1w, 2w, 4w
'use_perfect_flags': True,        # Use perfect flags from synthetic errors
```
**What This Tests**: Full system with all anomaly resistance mechanisms

**📊 RESULTS:**
```
Validation Metrics (vs original clean data):
  rmse: 28.517774
  mae: 17.985228
  nse: 0.985914

Validation Metrics (vs corrupted training data):
  rmse: 26.389715
  mae: 17.512098
  nse: 0.988130
```


### **EXPERIMENT 4: + Complex loss**
**Status**: ❌ **WAITING FOR EXP 2**
**Command**: `--experiment exp3_full_system`
**Config**:
```python
'use_anomaly_flags': True,        # Enable anomaly flagging as input feature
'use_weighted_loss': True,        # Weighted loss (0.3 for anomalies)
'anomaly_weight': 0.3,            # Weight for anomalous periods
'use_memory_protection': True,    # Memory protection mechanism
'use_lagged_features': True,      # ADDED: Lagged VST features
'lag_hours': [24, 48, 72, 168, 336, 672],  # 1d, 2d, 3d, 1w, 2w, 4w
'use_perfect_flags': True,        # Use perfect flags from synthetic errors
```
**What This Tests**: Full system with all anomaly resistance mechanisms

**📊 RESULTS:**
```
Validation Metrics (vs original clean data):
  rmse: [PASTE RESULTS HERE]
  mae: [PASTE RESULTS HERE]
  nse: [PASTE RESULTS HERE]

Validation Metrics (vs corrupted training data):
  rmse: [PASTE RESULTS HERE]
  mae: [PASTE RESULTS HERE]
  nse: [PASTE RESULTS HERE]

---

### **EXPERIMENT 5: + Window and potentailly complex loss**
**Status**: ❌ **WAITING FOR EXP 2**
**Command**: `--experiment exp3_full_system`
**Config**:
```python
'use_anomaly_flags': True,        # Enable anomaly flagging as input feature
'use_weighted_loss': True,        # Weighted loss (0.3 for anomalies)
'anomaly_weight': 0.3,            # Weight for anomalous periods
'use_memory_protection': True,    # Memory protection mechanism
'use_lagged_features': True,      # ADDED: Lagged VST features
'lag_hours': [24, 48, 72, 168, 336, 672],  # 1d, 2d, 3d, 1w, 2w, 4w
'use_perfect_flags': True,        # Use perfect flags from synthetic errors
```
**What This Tests**: Full system with all anomaly resistance mechanisms

**📊 RESULTS:**
```
Validation Metrics (vs original clean data):
  rmse: [PASTE RESULTS HERE]
  mae: [PASTE RESULTS HERE]
  nse: [PASTE RESULTS HERE]

Validation Metrics (vs corrupted training data):
  rmse: [PASTE RESULTS HERE]
  mae: [PASTE RESULTS HERE]
  nse: [PASTE RESULTS HERE]

## 📊 **Performance Comparison Table**

| Experiment | Anomaly Flags | Weighted Loss | Memory Protection | Lagged Features | RMSE (Clean) | MAE (Clean) | NSE (Clean) |
|------------|---------------|---------------|-------------------|-----------------|--------------|-------------|-------------|
| Baseline   | ✅            | ❌            | ❌                | ❌              | [RESULTS]    | [RESULTS]   | [RESULTS]   |
| Exp 1      | ✅            | ✅            | ❌                | ❌              | 51.08        | 35.80       | 0.9548      |
| Exp 2      | ✅            | ✅            | ✅                | ❌              | [RESULTS]    | [RESULTS]   | [RESULTS]   |
| Exp 3      | ✅            | ✅            | ✅                | ✅              | [RESULTS]    | [RESULTS]   | [RESULTS]   |

---

## 🔧 **Technical Details**

### **Loss Function**
- **Standard MAE**: `torch.nn.L1Loss()` 
- **Anomaly Aware Loss**: Weighted MAE (0.3 for anomalies) + pattern penalty (0.5 weight)

### **Memory Protection**
- **Mechanism**: Save last good cell state, blend 70% old + 30% new during anomalies
- **Trigger**: Based on anomaly flags (1=anomaly, 0=normal)

### **Lagged Features**
- **Hours**: [24, 48, 72, 168, 336, 672] (1 day to 4 weeks)
- **Created**: After error injection to reflect realistic corrupted history

### **Common Settings**
- **Station**: 21006846
- **Mode**: `quick_mode=True` 
- **Error multiplier**: 1.0
- **Epochs**: 45 (reduced for quick testing)
- **Perfect flags**: From synthetic error injection locations

---

## ✅ **Execution Order**

1. ✅ **Baseline**: Run first to establish performance without any anomaly resistance
2. ✅ **Experiment 1**: COMPLETED - anomaly aware loss only
3. ✅ **Experiment 2**: ABOUT TO RUN - add memory protection
4. ❌ **Experiment 3**: Final - add lagged features

---

## 📝 **Notes for Thesis**

This systematic approach will provide clear evidence of:
- **Individual contribution** of each mechanism
- **Cumulative improvement** as components are added
- **Trade-offs** between anomaly resistance and overall performance
- **Justification** for each design decision in the final model