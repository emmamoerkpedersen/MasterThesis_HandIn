# Anomaly Flagging Experiments - Systematic Testing Plan

## 🎯 **Objective**
Test each component incrementally to show the contribution of:
1. **Experiment 4**: Anomaly flags as input features
2. **Experiment 5**: Weighted loss function 
3. **Experiment 6**: Memory protection mechanism
4. **Experiment 7**: Lagged VST features

---

## 📋 **Experiment Configurations**

### **Experiment 4: Baseline with Anomaly Flags Only**
**Status**: ✅ **COMPLETED** - Results should exist from initial testing
**Config**:
```python
'use_anomaly_flags': True,
'use_weighted_loss': False,     # Standard MAE loss
'use_sliding_window': False,    # No memory protection
'use_lagged_features': False,   # No lagged features
```
**Result Location**: `experiment_flagging_test/` (original results)

---

### **Experiment 5A: Anomaly Flags + Weighted Loss (No Memory Protection)**
**Status**: ❌ **NEEDS TO BE RUN**
**Command**: `--experiment exp5a_weighted_loss_only`
**Config**:
```python
'use_anomaly_flags': True,
'use_weighted_loss': True,      # Weighted loss (0.3 for anomalies)
'anomaly_weight': 0.3,
'use_sliding_window': False,    # NO memory protection
'use_lagged_features': False,   # No lagged features
```
**Notes**: Need to temporarily disable memory protection in model code

---

### **Experiment 5B: Anomaly Flags + Memory Protection (No Weighted Loss)**
**Status**: ❌ **NEEDS TO BE RUN**
**Command**: `--experiment exp5b_memory_protection_only`
**Config**:
```python
'use_anomaly_flags': True,
'use_weighted_loss': False,     # Standard MAE loss
'use_sliding_window': False,    # Basic memory protection (single backup)
'use_lagged_features': False,   # No lagged features
```
**Notes**: Memory protection active, but standard MAE loss

---

### **Experiment 6A: Weighted Loss + Memory Protection (No Lagged Features)**
**Status**: ✅ **RUNNING NOW**
**Command**: `--experiment test_lagged_wo_features`
**Config**:
```python
'use_anomaly_flags': True,
'use_weighted_loss': True,      # Simple weighted loss
'anomaly_weight': 0.3,
'use_sliding_window': False,    # Basic memory protection
'use_lagged_features': False,   # NO lagged features
```
**Result Location**: `test_lagged_features/`

---

### **Experiment 6B: Weighted Loss + Memory Protection + Simple Loss Function**
**Status**: ❌ **NEEDS TO BE RUN**
**Command**: `--experiment exp6b_simple_loss_function`
**Config**:
```python
'use_anomaly_flags': True,
'use_weighted_loss': True,      # Simple loss (pattern penalty = 0.5)
'anomaly_weight': 0.3,
'use_sliding_window': False,    # Basic memory protection
'use_lagged_features': False,   # No lagged features
'use_simple_loss': True,        # Use anomaly_aware_loss_simple()
```
**Notes**: Use the simplified loss function with reduced pattern penalty

---

### **Experiment 7A: Full System Without Lagged Features**
**Status**: ✅ **COMPLETED** (same as 6A above)
**Result Location**: `test_lagged_features/`

📊 FLAGGING MODEL RESULTS:
Validation Metrics (vs original clean data):
  rmse: 62.272513
  mae: 36.698049
  nse: 0.932832

Validation Metrics (vs corrupted training data):
  rmse: 69.412530
  mae: 42.249310
  nse: 0.917863
---

### **Experiment 7B: Full System WITH Lagged Features**
**Status**: ✅ **RUNNING NEXT**
**Command**: `--experiment test_with_lagged_features`
**Config**:
```python
'use_anomaly_flags': True,
'use_weighted_loss': True,      # Simple weighted loss
'anomaly_weight': 0.3,
'use_sliding_window': False,    # Basic memory protection
'use_lagged_features': True,    # 6 lagged VST features added
'lag_hours': [24, 48, 72, 168, 336, 672],
```
**Result Location**: `test_with_lagged_features/`

---

## 🔧 **Code Changes Needed**

### **1. Add Simple Loss Function Toggle**
Add to `alternating_config.py`:
```python
'use_simple_loss': False,  # When True, use anomaly_aware_loss_simple()
```

### **2. Memory Protection Toggle**
Need ability to disable memory protection completely for Experiment 5A.

### **3. Loss Function Selection**
Modify model to choose between:
- `anomaly_aware_loss()` - Complex version (AVOID)
- `anomaly_aware_loss_simple()` - Simple version (USE THIS)
- Standard MAE loss

---

## 📊 **Expected Results**

| Experiment | Anomaly Flags | Weighted Loss | Memory Protection | Lagged Features | Expected Performance |
|------------|---------------|---------------|-------------------|-----------------|---------------------|
| 4          | ✅            | ❌            | ❌                | ❌              | Baseline            |
| 5A         | ✅            | ✅            | ❌                | ❌              | Better anomaly handling |
| 5B         | ✅            | ❌            | ✅                | ❌              | More stable predictions |
| 6A/7A      | ✅            | ✅            | ✅                | ❌              | Best anomaly resistance |
| 6B         | ✅            | ✅ (Simple)   | ✅                | ❌              | Smoother predictions |
| 7B         | ✅            | ✅            | ✅                | ✅              | Best overall performance? |

---

## 📝 **Notes**

- **All experiments** use `quick_mode=True` and `error_multiplier=1`
- **Loss function**: Focus on simple weighted loss, avoid complex version
- **Memory protection**: Basic single backup approach (no sliding window)
- **Station**: 21006846 for consistency
- **Evaluation**: Compare RMSE, oscillation behavior, anomaly resistance

---

## ✅ **Completion Status**

- [x] Experiment 4: Baseline (existing results)
- [x] Experiment 7A: No lagged features (running)
- [x] Experiment 7B: With lagged features (next)
- [ ] Experiment 5A: Weighted loss only
- [ ] Experiment 5B: Memory protection only  
- [ ] Experiment 6B: Simple loss function 