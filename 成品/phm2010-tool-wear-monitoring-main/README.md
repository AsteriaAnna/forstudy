# PHM 2010 CNC Tool Wear Monitoring System

A predictive health management (PHM) system for CNC milling tool wear monitoring, built using the PHM 2010 Data Challenge dataset.


## Project Overview
This project predicts the remaining useful life (RUL) of CNC milling tools using three complementary approaches:
1. **Physics/Reliability Models** — Power law degradation fitting with uncertainty quantification
2. **Data-Driven Regression** — GradientBoosting with quantile regression (P10/P50/P90)
3. **State-Space Filtering** — Kalman filter for online wear estimation

Additionally, the project implements **Statistical Process Control (SPC)** techniques used in manufacturing:
- Rolling Cp/Cpk trending
- CUSUM drift detection
- Preventive replacement policy

## Dataset
**PHM 2010 CNC Milling Dataset**
- 6 cutters (c1-c6), 315 cuts each
- 7 sensor channels: Force X/Y/Z, Vibration X/Y/Z, AE-RMS
- Sampled at 50 kHz (~219,000 rows per cut)
- Training cutters: c1, c4, c6 (wear measurements available)
- Test cutters: c2, c3, c5 (sensor data only)

## Repository Structure
```
phm2010-portfolio/
├── notebooks/
│   ├── 01_data_exploration.ipynb        # Feature engineering & correlation analysis
│   ├── 02_physics_reliability_models.ipynb  # Power law degradation models
│   ├── 03_data_driven_regression.ipynb  # GradientBoosting + quantile regression
│   ├── 04_kalman_filter.ipynb           # State-space wear estimation
│   └── 05_spc_drift_detection.ipynb     # Cp/Cpk, CUSUM, replacement policy
├── data/processed/
│   ├── train_final.csv                  # 945 cuts × 12 features
│   ├── physics_predictions.csv          # Physics model safe cut estimates
│   └── submission_data_driven.csv       # Challenge submission
├── plots/                               # All generated visualizations
└── README.md
```

## Methods

### 1. Feature Engineering
- Extracted 49 statistical features (mean, std, RMS, kurtosis, skewness, max, p2p) per sensor per cut
- Selected 12 features via correlation analysis (threshold > 0.50) and redundancy removal (threshold < 0.97)
- Top features: vib_x_std (r=0.924), vib_x_max (r=0.890), force_x_p2p (r=0.888)

### 2. Physics Model
- Fitted power law degradation curves (wear = a × t^b) to three training cutters
- Inverted power law to estimate safe cuts for wear thresholds 66–165 × 10⁻³ mm
- Conservative estimate = minimum across three training curves
- Uncertainty band grows from 30 to 250 cuts across scoring range

### 3. Data-Driven Regression
- Leave-one-cutter-out cross-validation (correct approach for time-series manufacturing data)
- GradientBoosting selected over Ridge and RandomForest (lowest average MAE: 17.82)
- Quantile regression provides P10/P50/P90 uncertainty bounds
- P10 (conservative) used for challenge submission to minimize overestimation penalty

### 4. Kalman Filter
- Hidden state = true wear, observation = GB model prediction
- Process noise Q=5.0, observation noise R=18.0 (set from validated MAE)
- Monotonic enforcement ensures physically consistent wear estimates
- Lower bound used for conservative challenge submission

### 5. SPC & Drift Detection
- CTQ characteristic: force_y_mean (correlation 0.859 with wear)
- Baseline spec limits from first 50 cuts: USL=4.69, LSL=1.12 (C1)
- Rolling Cp/Cpk (window=20) — Cpk drops from ~1.5 to -3.3 over tool life
- CUSUM detects drift at cuts 52-53 (wear = 72-102 × 10⁻³ mm, well before failure at 200)

## Key Results

| Method | MAE (wear units) | Notes |
|--------|-----------------|-------|
| Ridge Regression | 24.10 | Linear, weak on non-linear relationships |
| Random Forest | 17.89 | Strong but slightly worse than GB |
| GradientBoosting | 17.82 | Best overall, used for final submission |
| Physics Model | N/A | Baseline, conservative, no sensor data needed |

| Metric | Value |
|--------|-------|
| CUSUM Early Warning | Cut 52-53 |
| Wear at Alert | 72-102 × 10⁻³ mm |
| Cuts Before Failure at Alert | ~260 cuts |
| Features Selected | 12 of 49 |

## Infrastructure
- **Cloud:** Google Cloud Platform (Vertex AI Workbench, Cloud Storage, Cloud Run)
- **Dashboard:** Streamlit deployed on Cloud Run
- **Storage:** GCS bucket `gs://phm2010-data-temilola`

## Requirements
```
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
plotly
streamlit
gcsfs
```

## Author
Temilola Gbadamosi
