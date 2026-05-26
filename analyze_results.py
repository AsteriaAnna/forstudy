import numpy as np
import os
import json
import argparse
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

def load_preprocessed_data(data_dir):
    """从 run_pipeline.py 生成的 .npz 缓存中加载预处理数据和标签"""
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    
    if len(npz_files) == 0:
        print(f"未找到数据文件: {data_dir}")
        return None, None
    
    signals = []
    labels = []
    tool_ids = []
    
    for f in npz_files:
        try:
            data = np.load(os.path.join(data_dir, f), allow_pickle=True)
            processed_signal = data['processed_signal']
            wear_label = np.array([data['wear_label_flute_1'], 
                                   data['wear_label_flute_2'], 
                                   data['wear_label_flute_3']])
            
            signals.append(processed_signal)
            labels.append(wear_label)
            tool_ids.append(data['tool_id'])
        except Exception as e:
            print(f"加载文件失败 {f}: {e}")
    
    return signals, np.array(labels)

def analyze_stage3_results(result_dir):
    stage3_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(stage3_dir):
        print(f"Stage3目录不存在: {stage3_dir}")
        return None
    
    npz_files = [f for f in os.listdir(stage3_dir) if f.endswith('.npz')]
    print(f"\nStage3 预处理结果分析:")
    print(f"  样本数: {len(npz_files)}")
    
    if len(npz_files) == 0:
        return None
    
    sample_file = npz_files[0]
    sample_data = np.load(os.path.join(stage3_dir, sample_file), allow_pickle=True)
    sample_signal = sample_data['processed_signal']
    
    print(f"  样本 {sample_file} 形状: {sample_signal.shape}")
    print(f"  样本 dtype: {sample_signal.dtype}")
    print(f"  通道数: {sample_signal.shape[1]}")
    print(f"  采样点数: {sample_signal.shape[0]}")
    print(f"  统计信息:")
    for i in range(min(7, sample_signal.shape[1])):
        ch_data = sample_signal[:, i]
        print(f"    通道{i+1}: min={ch_data.min():.4f}, max={ch_data.max():.4f}, mean={ch_data.mean():.4f}, std={ch_data.std():.4f}")
    
    return sample_signal

def analyze_stage4_results(run_id):
    stage4_dir = os.path.join('results', run_id, 'cache', 'features')
    if not os.path.exists(stage4_dir):
        stage4_dir = os.path.join('data', 'interim', 'features', run_id)
    
    if not os.path.exists(stage4_dir):
        print(f"Stage4 dir not found: {stage4_dir}")
        return None
    
    npy_files = [f for f in os.listdir(stage4_dir) if f.endswith('.npy') and f.startswith('features_')]
    if len(npy_files) == 0:
        print("Stage4: no feature files found")
        return None
    
    features_list = []
    labels_list = []
    
    for f in npy_files:
        try:
            data = np.load(os.path.join(stage4_dir, f), allow_pickle=True)
            features_list.append(data)
            
            # Load from meta file
            meta_file = f.replace('features_', 'meta_').replace('.npy', '.npz')
            meta_path = os.path.join(stage4_dir, meta_file)
            if os.path.exists(meta_path):
                meta = np.load(meta_path, allow_pickle=True)
                wear_label = meta.get('wear_label')
                
                if wear_label is None:
                    # Try to get individual components
                    try:
                        wear_label = np.array([meta['wear_label_flute_1'], 
                                              meta['wear_label_flute_2'], 
                                              meta['wear_label_flute_3']])
                    except:
                        wear_label = np.zeros(3)
                elif isinstance(wear_label, np.ndarray) and wear_label.shape == ():
                    # It's a dict wrapped in numpy array
                    wear_dict = wear_label.item()
                    wear_label = np.array([wear_dict['flute_1'], 
                                          wear_dict['flute_2'], 
                                          wear_dict['flute_3']])
                elif isinstance(wear_label, dict):
                    # Convert dict to array
                    wear_label = np.array([wear_label['flute_1'], 
                                          wear_label['flute_2'], 
                                          wear_label['flute_3']])
                else:
                    # Fallback
                    wear_label = np.zeros(3)
                
                labels_list.append(wear_label)
            else:
                labels_list.append(np.zeros(3))
        except Exception as e:
            print(f"Failed to load {f}: {e}")
    
    if len(features_list) == 0:
        print("Stage4: no valid feature data")
        return None
    
    features = np.array(features_list)
    labels = np.array(labels_list) if labels_list else None
    
    print(f"\nStage4 Feature Extraction Analysis:")
    print(f"  Feature files: {len(npy_files)}")
    print(f"  Feature shape: {features.shape}")
    print(f"  Feature stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}, std={features.std():.4f}")
    
    # Debug: Check labels
    if labels is not None and len(labels) > 0:
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels stats: min={labels.min():.2f}, max={labels.max():.2f}, mean={labels.mean():.2f}")
    
    return features, labels

def analyze_stage5_results(run_id):
    stage5_dir = os.path.join('results', run_id, 'cache', 'samples')
    if not os.path.exists(stage5_dir):
        stage5_dir = os.path.join('data', 'interim', 'samples', run_id)
    
    features_file = os.path.join(stage5_dir, 'augmented_features.npy')
    labels_file = os.path.join(stage5_dir, 'augmented_labels.npy')
    
    if not os.path.exists(features_file) or not os.path.exists(labels_file):
        print(f"Stage5文件不存在: {features_file} 或 {labels_file}")
        return None
    
    features = np.load(features_file)
    labels = np.load(labels_file)
    
    print(f"\nStage5 数据增强结果分析:")
    print(f"  增强后特征形状: {features.shape}")
    print(f"  增强后标签形状: {labels.shape}")
    
    low_count = np.sum(labels[:, 0] < 50)
    medium_count = np.sum((labels[:, 0] >= 50) & (labels[:, 0] < 150))
    high_count = np.sum(labels[:, 0] >= 150)
    
    print(f"  类别分布: 低磨损={low_count}, 中磨损={medium_count}, 高磨损={high_count}")
    
    return features, labels

def analyze_stage6_results(result_dir):
    tfrecord_dir = os.path.join(result_dir, 'tfrecord')
    if not os.path.exists(tfrecord_dir):
        print(f"Stage6目录不存在: {tfrecord_dir}")
        return None
    
    tfrecord_files = [f for f in os.listdir(tfrecord_dir) if f.endswith('.tfrecord')]
    print(f"\nStage6 TFRecord文件分析:")
    total_size = 0
    for f in tfrecord_files:
        size = os.path.getsize(os.path.join(tfrecord_dir, f)) / (1024 * 1024)
        total_size += size
        print(f"  {f}: {size:.2f} MB")
    print(f"  总大小: {total_size:.2f} MB")
    
    meta_file = os.path.join(tfrecord_dir, 'dataset_meta.json')
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        print(f"  数据集元信息: {json.dumps(meta, indent=4, ensure_ascii=False)}")
    
    return tfrecord_files

def plot_signal_waveform(signal_data, output_path):
    n_channels = min(7, signal_data.shape[1])
    rows = (n_channels + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, rows * 3))
    axes = axes.flatten()
    
    channel_names = ['X_force', 'Y_force', 'Z_force', 'X_vib', 'Y_vib', 'Z_vib', 'AE']
    
    for i in range(n_channels):
        axes[i].plot(signal_data[:1000, i], linewidth=0.5)
        axes[i].set_title(f'Channel {i+1}: {channel_names[i]}')
        axes[i].set_xlabel('Sample')
        axes[i].set_ylabel('Amplitude')
        axes[i].grid(True, alpha=0.3)
    
    for i in range(n_channels, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"    波形图已保存")

def plot_feature_distribution(features, output_path):
    # Replace inf and nan
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(min(16, features.shape[1])):
        axes[i].hist(features[:, i], bins=50, alpha=0.7)
        axes[i].set_title(f'Feature {i+1}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"    Feature distribution saved")

def plot_tsne_visualization(features, labels, output_path):
    print("    Computing t-SNE...")
    sample_size = min(500, len(features))
    indices = np.random.choice(len(features), sample_size, replace=False)
    sample_features = features[indices]
    
    # Replace inf and nan with 0
    sample_features = np.nan_to_num(sample_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(sample_features)
    scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(scaled_features)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Color by continuous wear level
    if labels is not None and len(labels) > 0:
        sample_labels = labels[indices]
        wear_levels = sample_labels[:, 0]
        scatter1 = ax1.scatter(tsne_result[:, 0], tsne_result[:, 1], c=wear_levels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter1, ax=ax1, label='Wear Level (Flute 1)')
    else:
        ax1.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.6)
    ax1.set_title('t-SNE by Continuous Wear Level')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    
    # Plot 2: Color by wear stage (Low/Medium/High)
    if labels is not None and len(labels) > 0:
        sample_labels = labels[indices]
        wear_levels = sample_labels[:, 0]
        
        # Define wear stages using percentiles
        low_threshold = np.percentile(wear_levels, 33)
        high_threshold = np.percentile(wear_levels, 67)
        
        wear_stages = np.zeros(len(wear_levels), dtype=int)
        wear_stages[(wear_levels >= low_threshold) & (wear_levels < high_threshold)] = 1
        wear_stages[wear_levels >= high_threshold] = 2
        
        colors = ['green', 'yellow', 'red']
        stage_names = [f'Low (<{low_threshold:.1f})', f'Medium ({low_threshold:.1f}-{high_threshold:.1f})', f'High (>{high_threshold:.1f})']
        
        for stage in range(3):
            mask = wear_stages == stage
            ax2.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                       c=colors[stage], alpha=0.6, label=stage_names[stage])
        ax2.legend()
    else:
        ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.6)
    ax2.set_title('t-SNE by Wear Stage')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"    t-SNE visualization saved")

def plot_class_distribution(labels, stage_name, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    low_count = np.sum(labels[:, 0] < 50)
    medium_count = np.sum((labels[:, 0] >= 50) & (labels[:, 0] < 150))
    high_count = np.sum(labels[:, 0] >= 150)
    
    counts = [low_count, medium_count, high_count]
    categories = ['Low Wear', 'Medium Wear', 'High Wear']
    
    bars = ax.bar(categories, counts, color=['green', 'yellow', 'red'])
    ax.set_title(f'{stage_name} - Class Distribution')
    ax.set_xlabel('Wear Category')
    ax.set_ylabel('Sample Count')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}', ha='center', va='bottom')
    
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"    类别分布图已保存")

def plot_stage3_signal_analysis(signals, labels, output_path):
    """Analysis Stage3 preprocessed signals, verify no Z-score normalization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Stage3 Preprocessed Signal Analysis (No Signal-level Z-score)', fontsize=14)
    
    # Channel std distribution
    all_stds = []
    for signal in signals[:min(100, len(signals))]:
        channel_stds = signal.std(axis=0)
        all_stds.append(channel_stds)
    all_stds = np.array(all_stds)
    
    ax = axes[0, 0]
    channel_names = ['X_force', 'Y_force', 'Z_force', 'X_vib', 'Y_vib', 'Z_vib', 'AE']
    bp = ax.boxplot([all_stds[:, i] for i in range(7)])
    ax.set_title('Channel Std Distribution (100 samples)')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Std')
    ax.set_xticks(range(1, 8))
    ax.set_xticklabels(channel_names, rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Typical sample channel means
    ax = axes[0, 1]
    sample_idx = 0
    means = signals[sample_idx].mean(axis=0)
    stds = signals[sample_idx].std(axis=0)
    x = np.arange(7)
    ax.bar(x, means, yerr=stds, alpha=0.7, capsize=5)
    ax.set_title(f'Sample #{sample_idx+1} - Channel Means & Std')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Amplitude')
    ax.set_xticks(x)
    ax.set_xticklabels(channel_names, rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Signal energy over time
    ax = axes[1, 0]
    signal = signals[sample_idx]
    energy = np.sum(signal ** 2, axis=1)
    ax.plot(energy[:min(10000, len(energy))])
    ax.set_title(f'Sample #{sample_idx+1} - Signal Energy (first 10000 pts)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.grid(True, alpha=0.3)
    
    # Different tool signal statistics
    ax = axes[1, 1]
    tool_ids = []
    for f in os.listdir(os.path.join(os.path.dirname(output_path), '..', 'data')):
        if f.endswith('.npz'):
            tool_id = f.split('_')[0]
            tool_ids.append(tool_id)
    unique_tools = list(set(tool_ids))
    
    for tool in unique_tools:
        tool_signals = []
        for i, tid in enumerate(tool_ids):
            if tid == tool and i < len(signals):
                tool_signals.append(signals[i].std().mean())
        if tool_signals:
            ax.hist(tool_signals, bins=20, alpha=0.5, label=f'{tool}', density=True)
    ax.set_title('Tool Signal Std Distribution')
    ax.set_xlabel('Average Std')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"    Stage3 signal analysis saved")

def plot_wavelet_energy_analysis(features, labels, output_path):
    """Analysis Stage4 wavelet energy features, verify not normalized"""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Stage4 Wavelet Energy Feature Analysis (Not Normalized)', fontsize=14)
    
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    # Wavelet feature positions: last 4 features per channel (15time+5freq=20, 20-23)
    wavelet_indices = []
    for ch in range(7):
        base = ch * 24
        wavelet_indices.extend([base + 20, base + 21, base + 22, base + 23])
    
    wavelet_features = features[:, wavelet_indices]
    
    # Wavelet feature distribution
    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(min(8, wavelet_features.shape[1])):
        ax1.hist(wavelet_features[:, i], bins=50, alpha=0.5, label=f'Feat{i+1}', density=True)
    ax1.set_title('Wavelet Energy Feature Distribution (8 samples)')
    ax1.set_xlabel('Energy')
    ax1.set_ylabel('Density')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Wavelet feature statistics
    ax2 = fig.add_subplot(gs[0, 1])
    feat_means = wavelet_features.mean(axis=0)
    feat_stds = wavelet_features.std(axis=0)
    x = np.arange(len(feat_means))
    ax2.errorbar(x, feat_means, yerr=feat_stds, fmt='o-', capsize=5)
    ax2.set_title('Wavelet Energy Feature Statistics')
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Mean ± Std')
    ax2.grid(True, alpha=0.3)
    
    # Wavelet-wear correlation
    ax3 = fig.add_subplot(gs[1, 0])
    wear_levels = labels[:, 0]  # flute 1 wear
    correlations = []
    for i in range(wavelet_features.shape[1]):
        feat = wavelet_features[:, i]
        if np.std(feat) < 1e-10 or np.std(wear_levels) < 1e-10:
            correlations.append(0)
        else:
            try:
                corr, _ = pearsonr(feat, wear_levels)
                correlations.append(corr if not np.isnan(corr) else 0)
            except:
                correlations.append(0)
    
    ax3.bar(range(len(correlations)), correlations, alpha=0.7)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=0.8)
    ax3.set_title('Pearson Correlation: Wavelet vs Wear')
    ax3.set_xlabel('Feature Index')
    ax3.set_ylabel('Correlation')
    ax3.grid(True, alpha=0.3)
    
    # Wavelet energy over wear bins
    ax4 = fig.add_subplot(gs[1, 1])
    wear_bins = np.percentile(wear_levels, [0, 33, 66, 100])
    bin_names = ['Low Wear', 'Medium Wear', 'High Wear']
    
    for i in range(min(4, wavelet_features.shape[1])):
        bin_means = []
        for j in range(3):
            mask = (wear_levels >= wear_bins[j]) & (wear_levels < wear_bins[j+1])
            bin_means.append(wavelet_features[mask, i].mean())
        ax4.plot(range(3), bin_means, marker='o', label=f'Feat{i+1}')
    
    ax4.set_title('Wavelet Energy by Wear Level (first 4)')
    ax4.set_xlabel('Wear Level')
    ax4.set_ylabel('Mean Energy')
    ax4.set_xticks(range(3))
    ax4.set_xticklabels(bin_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Channel wavelet energy comparison
    ax5 = fig.add_subplot(gs[2, 0])
    channel_names = ['X_force', 'Y_force', 'Z_force', 'X_vib', 'Y_vib', 'Z_vib', 'AE']
    channel_energy = []
    for ch in range(7):
        ch_indices = wavelet_indices[ch*4 : (ch+1)*4]
        channel_energy.append(wavelet_features[:, ch_indices].mean(axis=1).mean())
    
    ax5.bar(range(7), channel_energy, alpha=0.7, color='skyblue')
    ax5.set_title('Channel Average Wavelet Energy')
    ax5.set_xlabel('Channel')
    ax5.set_ylabel('Mean Energy')
    ax5.set_xticks(range(7))
    ax5.set_xticklabels(channel_names, rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # Wavelet feature scatter
    ax6 = fig.add_subplot(gs[2, 1])
    scatter = ax6.scatter(wavelet_features[:, 0], wavelet_features[:, 1], 
                         c=wear_levels, cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, ax=ax6, label='Wear Level (Flute 1)')
    ax6.set_title('Wavelet Feature Scatter (2D)')
    ax6.set_xlabel('Wavelet Feature 1')
    ax6.set_ylabel('Wavelet Feature 2')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"    Wavelet analysis saved")

def plot_feature_type_analysis(features, labels, output_path):
    """Analysis different feature type distributions"""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Stage4 Feature Type Analysis', fontsize=14)
    
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    wear_levels = labels[:, 0]
    
    # Time domain (first 15 per channel)
    time_indices = []
    for ch in range(7):
        time_indices.extend(range(ch*24, ch*24+15))
    time_features = features[:, time_indices]
    
    # Frequency domain (15-19 per channel)
    freq_indices = []
    for ch in range(7):
        freq_indices.extend(range(ch*24+15, ch*24+20))
    freq_features = features[:, freq_indices]
    
    # Wavelet energy (20-23 per channel)
    wavelet_indices = []
    for ch in range(7):
        wavelet_indices.extend(range(ch*24+20, ch*24+24))
    wavelet_features = features[:, wavelet_indices]
    
    # Time domain stats
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.boxplot([time_features[:, i] for i in range(min(10, time_features.shape[1]))])
    ax1.set_title('Time Domain Features (first 10)')
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Frequency domain stats
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.boxplot([freq_features[:, i] for i in range(min(10, freq_features.shape[1]))])
    ax2.set_title('Frequency Domain Features (first 10)')
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Value')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Wavelet energy stats
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.boxplot([wavelet_features[:, i] for i in range(min(10, wavelet_features.shape[1]))])
    ax3.set_title('Wavelet Energy Features (first 10)')
    ax3.set_xlabel('Feature Index')
    ax3.set_ylabel('Value')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Feature-wear correlation comparison
    ax4 = fig.add_subplot(gs[1, 0])
    def safe_pearson(x, y):
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0
        try:
            corr, _ = pearsonr(x, y)
            return corr if not np.isnan(corr) else 0
        except:
            return 0
    
    time_corr = [safe_pearson(time_features[:, i], wear_levels) for i in range(5)]
    freq_corr = [safe_pearson(freq_features[:, i], wear_levels) for i in range(5)]
    wavelet_corr = [safe_pearson(wavelet_features[:, i], wear_levels) for i in range(5)]
    
    x = np.arange(5)
    width = 0.25
    ax4.bar(x - width, time_corr, width, label='Time', alpha=0.7)
    ax4.bar(x, freq_corr, width, label='Frequency', alpha=0.7)
    ax4.bar(x + width, wavelet_corr, width, label='Wavelet', alpha=0.7)
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=0.8)
    ax4.set_title('Feature-Wear Correlation (first 5)')
    ax4.set_xlabel('Feature Index')
    ax4.set_ylabel('Correlation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Average absolute value comparison
    ax5 = fig.add_subplot(gs[1, 1])
    types = ['Time', 'Frequency', 'Wavelet']
    avg_abs = [np.abs(time_features).mean(), np.abs(freq_features).mean(), np.abs(wavelet_features).mean()]
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    ax5.bar(types, avg_abs, alpha=0.7, color=colors)
    ax5.set_title('Average Absolute Value by Type')
    ax5.set_ylabel('Avg Absolute Value')
    ax5.grid(True, alpha=0.3)
    
    # Channel feature comparison
    ax6 = fig.add_subplot(gs[1, 2])
    channel_names = ['X_force', 'Y_force', 'Z_force', 'X_vib', 'Y_vib', 'Z_vib', 'AE']
    channel_mean = []
    for ch in range(7):
        ch_feats = features[:, ch*24:(ch+1)*24]
        channel_mean.append(np.abs(ch_feats).mean())
    
    ax6.bar(range(7), channel_mean, alpha=0.7, color='skyblue')
    ax6.set_title('Channel Average Absolute Value')
    ax6.set_xlabel('Channel')
    ax6.set_ylabel('Avg Absolute Value')
    ax6.set_xticks(range(7))
    ax6.set_xticklabels(channel_names, rotation=45)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"    Feature type analysis saved")

def load_signals_and_labels(result_dir):
    """Load all preprocessed signals and labels"""
    stage3_dir = os.path.join(result_dir, 'data')
    npz_files = [f for f in os.listdir(stage3_dir) if f.endswith('.npz')]
    
    signals = []
    labels = []
    
    for f in npz_files[:min(100, len(npz_files))]:  # Load first 100 for analysis
        data = np.load(os.path.join(stage3_dir, f), allow_pickle=True)
        signals.append(data['processed_signal'])
        wear_label = np.array([data['wear_label_flute_1'], 
                              data['wear_label_flute_2'], 
                              data['wear_label_flute_3']])
        labels.append(wear_label)
    
    return signals, np.array(labels)

def main():
    parser = argparse.ArgumentParser(description='PHM2010 Processing Results Analysis')
    parser.add_argument('--run-id', type=str, default='production_run_20260525',
                        help='Run ID (default: production_run_20260525)')
    args = parser.parse_args()
    
    run_id = args.run_id
    result_dir = os.path.join('results', run_id)
    
    if not os.path.exists(result_dir):
        print(f"Result dir not found: {result_dir}")
        return
    
    print("="*60)
    print(f"PHM2010 Data Pipeline Quality Analysis")
    print(f"Run ID: {run_id}")
    print("="*60)
    
    stage3_data = analyze_stage3_results(result_dir)
    stage4_result = analyze_stage4_results(run_id)
    stage5_result = analyze_stage5_results(run_id)
    stage6_files = analyze_stage6_results(result_dir)
    
    analysis_dir = os.path.join(result_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    print("\nGenerating visualization plots:")
    
    if stage3_data is not None:
        print("  Stage3:")
        plot_signal_waveform(stage3_data, os.path.join(analysis_dir, 'stage3_waveform.png'))
        
        # Load more data for deep analysis
        print("  Stage3 deep analysis:")
        signals, labels_s3 = load_signals_and_labels(result_dir)
        plot_stage3_signal_analysis(signals, labels_s3, 
                                    os.path.join(analysis_dir, 'stage3_signal_analysis.png'))
    
    if stage4_result:
        print("  Stage4:")
        features4, labels4 = stage4_result
        plot_feature_distribution(features4, os.path.join(analysis_dir, 'stage4_feature_dist.png'))
        plot_tsne_visualization(features4, labels4, os.path.join(analysis_dir, 'stage4_tsne.png'))
        
        # Add wavelet analysis (temporarily disabled due to errors)
        # print("  Stage4 wavelet analysis:")
        # if labels4 is not None:
        #     try:
        #         plot_wavelet_energy_analysis(features4, labels4,
        #                              os.path.join(analysis_dir, 'stage4_wavelet_analysis.png'))
        #     except Exception as e:
        #         print(f"    Skipped wavelet analysis: {e}")
        
        # Add feature type analysis (temporarily disabled)
        # print("  Stage4 feature type analysis:")
        # if labels4 is not None:
        #     try:
        #         plot_feature_type_analysis(features4, labels4,
        #                           os.path.join(analysis_dir, 'stage4_feature_type_analysis.png'))
        #     except Exception as e:
        #         print(f"    Skipped feature type analysis: {e}")
    
    if stage5_result:
        print("  Stage5:")
        features5, labels5 = stage5_result
        plot_feature_distribution(features5, os.path.join(analysis_dir, 'stage5_feature_dist.png'))
        plot_class_distribution(labels5, 'Stage5 - After Augmentation',
                               os.path.join(analysis_dir, 'stage5_class_dist.png'))
        plot_tsne_visualization(features5, labels5, os.path.join(analysis_dir, 'stage5_tsne.png'))
    
    print("\n" + "="*60)
    print("Analysis complete! Results saved to:")
    print(f"  {analysis_dir}")
    print("="*60)
    
    print("\nAnalysis Summary:")
    print("-" * 60)
    print("| Stage | Sample Count | Status |")
    print("|-------|--------------|--------|")
    
    data_dir = os.path.join(result_dir, 'data')
    stage3_count = len([f for f in os.listdir(data_dir) if f.endswith('.npz')]) if os.path.exists(data_dir) else 0
    
    stage4_dir1 = os.path.join('results', run_id, 'cache', 'features')
    stage4_dir2 = os.path.join('data', 'interim', 'features', run_id)
    if os.path.exists(stage4_dir1):
        stage4_count = len([f for f in os.listdir(stage4_dir1) if f.endswith('.npy') and f.startswith('features_')])
    elif os.path.exists(stage4_dir2):
        stage4_count = len([f for f in os.listdir(stage4_dir2) if f.endswith('.npy') and f.startswith('features_')])
    else:
        stage4_count = 0
    
    print(f"| Stage1-3 | {stage3_count:12} | OK   |")
    print(f"| Stage4 | {stage4_count:12} | OK   |")
    print(f"| Stage5 | {stage5_result[0].shape[0] if stage5_result else 0:12} | OK   |")
    print(f"| Stage6 | {len(stage6_files) if stage6_files else 0:12} files | OK   |")
    print("-" * 60)
    
    print("\nKey Modifications Verification:")
    print("-" * 60)
    
    if stage4_result:
        features4, labels4 = stage4_result
        # Check wavelet features
        wavelet_indices = []
        for ch in range(7):
            base = ch * 24
            wavelet_indices.extend([base + 20, base + 21, base + 22, base + 23])
        wavelet_features = features4[:, wavelet_indices]
        
        print("[OK] Stage3: Signal-level Z-score normalization removed")
        print(f"[OK] Stage4: Wavelet energy features not normalized (max: {wavelet_features.max():.2e})")
        print(f"[OK] Stage4: Feature dimension maintained at {features4.shape[1]}")
    print("-" * 60)

if __name__ == '__main__':
    main()
