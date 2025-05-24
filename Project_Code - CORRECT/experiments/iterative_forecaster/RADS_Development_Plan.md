# RADS Development Plan: Robust Anomaly Detection System

## 🎯 Core Objectives
Build a **2-in-1 system** that:
1. **Robust Prediction**: Stable water level forecasting even with anomalous training data
2. **Anomaly Detection**: Detect anomalies by comparing observations vs predictions  
3. **Confidence Scoring**: Provide confidence levels for detected anomalies
4. **Real-world Robustness**: Handle both natural sensor issues and synthetic errors

---

## 📋 **4-Step Development Plan**

### **STEP 1: Baseline Performance** 🎯
**Goal**: Achieve excellent prediction accuracy on real-world data  
**Success Criteria**: RMSE <50mm, MAE <30mm, Good peak prediction (slope ≈ 1.0)  
**Focus**: Feature engineering and architecture optimization

### **STEP 2: Robustness Testing** 🛡️  
**Goal**: Test model stability with synthetic error injection  
**Success Criteria**: <20% performance degradation with 20% error injection  
**Focus**: Error correction and training robustness

### **STEP 3: Anomaly Detection** 🔍
**Goal**: Implement MAD-based detection with confidence scoring  
**Success Criteria**: 5-15% detection rate, clear confidence differentiation  
**Focus**: Enhanced z-score and visualization

### **STEP 4: Optimization** ⚙️
**Goal**: Optimize thresholds and finalize system  
**Success Criteria**: Validated on multiple stations  
**Focus**: Adaptive thresholds and cross-validation

---

## 📊 **Experimental Results by Step**

---

## **STEP 1: BASELINE PERFORMANCE** 🔄 **IN PROGRESS**

**Current Status**: Testing feature engineering approaches to solve peak underestimation

### **Experiment 0: Simple Baseline** ✅ **COMPLETED**
```python
# Configuration
{
    'hidden_size': 32, 'objective_function': 'mse_loss',
    'use_time_features': False, 'use_cumulative_features': False
}
```
**Results**: RMSE: 66.47mm, MAE: 25.65mm, R²: 0.8551, **Slope: 0.72** ⚠️  
**Issue**: Systematic peak underestimation - missing 1400mm peaks

### **Experiment 1: Extended Training** ✅ **COMPLETED**  
**Change**: 50 epochs (vs 16)  
**Results**: No improvement in peak prediction  
**Conclusion**: Training time not the issue

### **Experiment 2: Architecture + Loss** ✅ **COMPLETED**
```python
# Configuration  
{
    'hidden_size': 64, 'objective_function': 'mae_loss',
    'use_time_features': False, 'use_cumulative_features': False
}
```
**Results**: RMSE: 42.25mm ✅, MAE: 12.32mm ✅, NSE: 0.9415 ✅  
**Achievement**: 36% RMSE improvement, 52% MAE improvement  
**Issue**: Peak underestimation still persists

### **Experiment 3: Time Features** 🔄 **RUNNING**
```python
# Configuration
{
    'hidden_size': 64, 'objective_function': 'mae_loss', 
    'use_time_features': True,  # ← ADD seasonal patterns
    'use_cumulative_features': False
}
```
**Hypothesis**: Seasonal patterns help predict peak timing  
**Command**: `python run_alternating_model.py --station_id 21006846 --experiment 3`

### **Experiment 4: Cumulative Features** 🔄 **PLANNED**
```python
# Configuration
{
    'hidden_size': 64, 'objective_function': 'mae_loss',
    'use_time_features': True, 'use_cumulative_features': True  # ← ADD rainfall accumulation
}
```
**Hypothesis**: Cumulative rainfall better predicts rain-driven peaks  
**Command**: `python run_alternating_model.py --station_id 21006846 --experiment 4`

### **Step 1 - Future Options (if needed):**
- **Experiment 5**: Peak-weighted loss (3x weight for >800mm)
- **Experiment 6**: Disable alternating during peaks  
- **Experiment 7**: Larger hidden size (128/256)
- **Experiment 8**: Different alternating strategies

**Step 1 Decision Point**: Once slope ≈ 1.0 achieved → Proceed to Step 2

---

## **STEP 2: ROBUSTNESS TESTING** ⏳ **WAITING**

**Prerequisites**: Successful Step 1 completion  
**Approach**: Use best Step 1 config + synthetic error injection

### **Planned Experiments:**

#### **Experiment A: Training Robustness**
```bash
python run_alternating_model.py --station_id 21006846 --error_multiplier 1.0 --error_type train --experiment 2A
```
**Test**: 20% error injection in training only  
**Metrics**: Performance on clean validation data

#### **Experiment B: Validation Robustness**  
```bash
python run_alternating_model.py --station_id 21006846 --error_multiplier 1.0 --error_type validation --experiment 2B
```
**Test**: Clean training, corrupted validation  
**Metrics**: Error correction capability

#### **Experiment C: Full Robustness**
```bash  
python run_alternating_model.py --station_id 21006846 --error_multiplier 1.0 --error_type both --experiment 2C
```
**Test**: Errors in both training and validation  
**Metrics**: Overall system robustness

**Step 2 Success Criteria**:
- Performance degradation <20% with synthetic errors
- Model still converges reliably  
- Predictions smoother than corrupted inputs

---

## **STEP 3: ANOMALY DETECTION** ⏳ **WAITING**

**Prerequisites**: Successful Step 2 completion  
**Approach**: Enhanced MAD-based detection with confidence scoring

### **Planned Experiments:**

#### **Experiment A: Basic Detection**
**Goal**: Implement enhanced z-score with confidence  
**Components**: 
- Modify existing `z_score.py` 
- Add confidence scoring (High/Medium/Low)
- Basic visualization

#### **Experiment B: Confidence Calibration**
**Goal**: Optimize confidence score accuracy  
**Metrics**:
- High confidence → True anomalies correlation
- False positive rate on clean data
- Detection rate optimization (target: 5-15%)

#### **Experiment C: Visualization Enhancement**
**Goal**: Comprehensive anomaly visualization  
**Components**:
- Predictions vs observations
- Anomaly flags with confidence colors
- Residual analysis plots
- Time-series anomaly overview

**Step 3 Success Criteria**:
- Clear confidence differentiation between obvious and subtle anomalies
- Low false positive rate on clean validation data
- High confidence anomalies correspond to obvious errors

---

## **STEP 4: OPTIMIZATION** ⏳ **WAITING**

**Prerequisites**: Successful Step 3 completion  
**Approach**: System optimization and validation

### **Planned Experiments:**

#### **Experiment A: Threshold Optimization**
**Options**:
- Static threshold tuning (optimize MAD multiplier)
- Adaptive thresholds (dynamic based on recent performance)  
- Multi-scale detection (different window sizes)

#### **Experiment B: Cross-Station Validation**
**Test Stations**: 21006845, 21006847, others  
**Metrics**: Generalization performance across different locations

#### **Experiment C: Final System Integration**
**Components**:
- Complete RADS pipeline
- Production-ready configuration
- Documentation and deployment guidelines

---

## 🔧 **Implementation Checklist**

### **Current Status** ✅
- [x] Alternating LSTM model working
- [x] Synthetic error injection system  
- [x] Basic MAD anomaly detection
- [x] Experiment organization system
- [x] CUDA acceleration active

### **To Implement** 📋
- [ ] Enhanced z-score with confidence (`enhanced_anomaly_detection.py`)
- [ ] RADS visualization module (`rads_visualization.py`)  
- [ ] Peak-weighted loss function (if needed)
- [ ] Adaptive threshold system
- [ ] Cross-station validation framework

---

## 📈 **Success Metrics Summary**

| Step | Primary Metric | Target | Current Status |
|------|---------------|---------|----------------|
| **Step 1** | Peak Prediction (Slope) | ≈ 1.0 | 0.72 ⚠️ |
| **Step 1** | RMSE | <50mm | 42.25mm ✅ |  
| **Step 1** | MAE | <30mm | 12.32mm ✅ |
| **Step 2** | Robustness | <20% degradation | TBD |
| **Step 3** | Detection Rate | 5-15% | TBD |
| **Step 4** | Cross-Station | Validated | TBD |

---

## 🚀 **Next Actions**

1. **Complete Experiment 3** (Time Features) → Analyze peak improvement
2. **Run Experiment 4** (Cumulative Features) if needed
3. **Achieve Step 1 success criteria** before proceeding
4. **Implement Step 2** robustness testing
5. **Build enhanced anomaly detection** for Step 3

**Current Priority**: Solve peak underestimation in Step 1 ⚡ 