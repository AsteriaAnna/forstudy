#!/usr/bin/env python3
"""
PHM 2010 Tool Wear Monitoring Pipeline
Adapted from the public repository for local data processing.

Features:
- Extracts 49 statistical features from 7 sensor channels
- Selects top 12 features via correlation analysis
- Supports local data (no GCS required)
- Outputs processed data ready for deep learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error

# Configuration
DATA_DIR = "data/interim/raw/phm2010_raw"
OUTPUT_DIR = "data/processed/phm2010"
TRAIN_CUTTERS = ["c1", "c4", "c6"]
TEST_CUTTERS = ["c2", "c3", "c5"]
FEATURES_TO_SELECT = 12

def extract_features(filepath):
    """
    Reads one cut CSV file and extracts statistical features
    from each of the 7 sensor signals.
    Returns a dictionary of 49 features.
    """
    df = pd.read_csv(filepath, header=None,
                     names=['force_x', 'force_y', 'force_z',
                            'vib_x', 'vib_y', 'vib_z', 'ae_rms'])
    
    features = {}
    
    for col in df.columns:
        signal = df[col].values
        
        features[f'{col}_mean']     = np.mean(signal)
        features[f'{col}_std']      = np.std(signal)
        features[f'{col}_rms']      = np.sqrt(np.mean(signal**2))
        features[f'{col}_kurtosis'] = stats.kurtosis(signal)
        features[f'{col}_skewness'] = stats.skew(signal)
        features[f'{col}_max']      = np.max(signal)
        features[f'{col}_p2p']      = np.max(signal) - np.min(signal)
    
    return features

def process_cutter(cutter_id, data_dir):
    """
    Processes all cut files for one cutter and returns
    a dataframe with features and wear labels
    """
    print(f"\nProcessing cutter {cutter_id}...")
    
    wear_path = os.path.join(data_dir, cutter_id, f"{cutter_id}_wear.csv")
    
    if os.path.exists(wear_path):
        wear_df = pd.read_csv(wear_path, header=0,
                              names=['cut', 'flute_1', 'flute_2', 'flute_3'])
        
        for col in ['cut', 'flute_1', 'flute_2', 'flute_3']:
            wear_df[col] = pd.to_numeric(wear_df[col], errors='coerce')
        
        wear_df['max_wear'] = wear_df[['flute_1', 'flute_2', 'flute_3']].max(axis=1)
    else:
        wear_df = None
        print(f"  No wear file found for cutter {cutter_id}")
    
    sensor_dir = os.path.join(data_dir, cutter_id, cutter_id)
    sensor_files = sorted(glob.glob(os.path.join(sensor_dir, "*.csv")))
    print(f"  Found {len(sensor_files)} sensor files")
    
    all_features = []
    
    for filepath in sensor_files:
        filename = os.path.basename(filepath)
        cut_num = int(filename.split('_')[-1].replace('.csv', ''))
        
        features = extract_features(filepath)
        features['cut'] = cut_num
        features['cutter'] = cutter_id
        
        if wear_df is not None:
            wear_row = wear_df[wear_df['cut'] == cut_num]
            if not wear_row.empty:
                features['max_wear'] = wear_row['max_wear'].values[0]
            else:
                features['max_wear'] = np.nan
        
        all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    print(f"  Processed {len(features_df)} cuts")
    return features_df

def select_features(df, target_col='max_wear', n_features=12, corr_threshold=0.50, redundancy_threshold=0.97):
    """
    Select features based on correlation with target and redundancy removal
    """
    feature_cols = [col for col in df.columns if col not in ['cut', 'cutter', 'max_wear']]
    
    correlations = df[feature_cols + [target_col]].corr()[target_col].drop(target_col)
    correlations = correlations[abs(correlations) >= corr_threshold].sort_values(ascending=False)
    
    selected = []
    for feature in correlations.index:
        redundant = False
        for sel in selected:
            corr = df[feature].corr(df[sel])
            if abs(corr) >= redundancy_threshold:
                redundant = True
                break
        
        if not redundant:
            selected.append(feature)
            if len(selected) >= n_features:
                break
    
    print(f"\nSelected {len(selected)} features:")
    for i, feat in enumerate(selected, 1):
        corr_val = correlations[feat]
        print(f"  {i}. {feat} (r = {corr_val:.3f})")
    
    return selected

def plot_correlation_heatmap(df, selected_features, output_path):
    """
    Plot correlation heatmap of selected features
    """
    plt.figure(figsize=(12, 10))
    corr_matrix = df[selected_features + ['max_wear']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title('Correlation Heatmap - Selected Features vs Wear')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved correlation heatmap to {output_path}")

def plot_wear_progression(df, output_path):
    """
    Plot wear progression for all cutters
    """
    plt.figure(figsize=(12, 6))
    
    for cutter in TRAIN_CUTTERS:
        cutter_df = df[df['cutter'] == cutter]
        max_wear = cutter_df['max_wear'].max()
        plt.plot(cutter_df['cut'], cutter_df['max_wear'], label=f'{cutter} (max={max_wear:.0f})')
    
    plt.axhline(y=200, color='black', linestyle='--', label='Failure Threshold (200)')
    plt.axhline(y=66, color='gray', linestyle=':', label='Score Range Start (66)')
    plt.axhline(y=165, color='gray', linestyle=':', label='Score Range End (165)')
    
    plt.title('Maximum Wear Progression - All Training Cutters')
    plt.xlabel('Cut Number')
    plt.ylabel('Wear (10⁻³ mm)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved wear progression plot to {output_path}")

def main():
    print("=" * 60)
    print("PHM 2010 Tool Wear Monitoring Pipeline")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nData directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    all_dfs = []
    for cutter in TRAIN_CUTTERS:
        df = process_cutter(cutter, DATA_DIR)
        all_dfs.append(df)
    
    train_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal training dataset shape: {train_df.shape}")
    
    print("\n" + "=" * 40)
    print("Feature Selection")
    print("=" * 40)
    selected_features = select_features(train_df, n_features=FEATURES_TO_SELECT)
    
    print("\n" + "=" * 40)
    print("Visualization")
    print("=" * 40)
    plot_correlation_heatmap(train_df, selected_features, 
                             os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
    plot_wear_progression(train_df, 
                          os.path.join(OUTPUT_DIR, "wear_progression.png"))
    
    final_df = train_df[['cutter', 'cut'] + selected_features + ['max_wear']]
    final_df.to_csv(os.path.join(OUTPUT_DIR, "train_final.csv"), index=False)
    print(f"\nSaved final training data to {os.path.join(OUTPUT_DIR, 'train_final.csv')}")
    
    print("\n" + "=" * 40)
    print("Model Evaluation")
    print("=" * 40)
    X = final_df[selected_features].values
    y = final_df['max_wear'].values
    
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    scores = cross_val_score(gb_model, X, y, cv=3, scoring='neg_mean_absolute_error')
    mae = -scores.mean()
    
    print(f"\nGradientBoosting MAE: {mae:.2f} μm")
    print("(Reference: Public project achieved MAE = 17.82)")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()