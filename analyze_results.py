import numpy as np
import os
import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

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
        print(f"Stage4目录不存在: {stage4_dir}")
        return None
    
    npy_files = [f for f in os.listdir(stage4_dir) if f.endswith('.npy') and f.startswith('features_')]
    if len(npy_files) == 0:
        print(f"Stage4没有找到特征文件")
        return None
    
    features_list = []
    labels_list = []
    
    for f in npy_files:
        try:
            data = np.load(os.path.join(stage4_dir, f), allow_pickle=True)
            features_list.append(data)
            
            meta_file = f.replace('features_', 'meta_')
            meta_path = os.path.join(stage4_dir, meta_file)
            if os.path.exists(meta_path):
                meta = np.load(meta_path, allow_pickle=True)
                labels_list.append(meta.get('wear_label', np.zeros(3)))
        except Exception as e:
            print(f"加载文件失败 {f}: {e}")
    
    if len(features_list) == 0:
        print("Stage4没有找到有效特征数据")
        return None
    
    features = np.array(features_list)
    labels = np.array(labels_list) if labels_list else None
    
    print(f"\nStage4 特征提取结果分析:")
    print(f"  特征文件数: {len(npy_files)}")
    print(f"  特征形状: {features.shape}")
    print(f"  特征统计: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}, std={features.std():.4f}")
    
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
    
    low_count = np.sum(labels[:, 0] < 0.1)
    medium_count = np.sum((labels[:, 0] >= 0.1) & (labels[:, 0] < 0.3))
    high_count = np.sum(labels[:, 0] >= 0.3)
    
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
    print(f"    特征分布图已保存")

def plot_tsne_visualization(features, labels, output_path):
    print("    正在计算t-SNE...")
    sample_size = min(500, len(features))
    indices = np.random.choice(len(features), sample_size, replace=False)
    sample_features = features[indices]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(sample_features)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(scaled_features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None and len(labels) > 0:
        sample_labels = labels[indices]
        wear_levels = sample_labels[:, 0]
        scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], c=wear_levels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Wear Level (Flute 1)')
    else:
        ax.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.6)
    
    ax.set_title('t-SNE Visualization of Features')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"    t-SNE可视化图已保存")

def plot_class_distribution(labels, stage_name, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    low_count = np.sum(labels[:, 0] < 0.1)
    medium_count = np.sum((labels[:, 0] >= 0.1) & (labels[:, 0] < 0.3))
    high_count = np.sum(labels[:, 0] >= 0.3)
    
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

def main():
    parser = argparse.ArgumentParser(description='PHM2010 处理结果分析')
    parser.add_argument('--run-id', type=str, default='production_run_20260525',
                        help='运行ID (default: production_run_20260525)')
    args = parser.parse_args()
    
    run_id = args.run_id
    result_dir = os.path.join('results', run_id)
    
    if not os.path.exists(result_dir):
        print(f"结果目录不存在: {result_dir}")
        return
    
    print("="*60)
    print(f"PHM2010 数据集处理流水线结果质量分析")
    print(f"Run ID: {run_id}")
    print("="*60)
    
    stage3_data = analyze_stage3_results(result_dir)
    stage4_result = analyze_stage4_results(run_id)
    stage5_result = analyze_stage5_results(run_id)
    stage6_files = analyze_stage6_results(result_dir)
    
    analysis_dir = os.path.join(result_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    print("\n生成可视化分析图:")
    
    if stage3_data is not None:
        print("  Stage3:")
        plot_signal_waveform(stage3_data, os.path.join(analysis_dir, 'stage3_waveform.png'))
    
    if stage4_result:
        print("  Stage4:")
        features4, labels4 = stage4_result
        plot_feature_distribution(features4, os.path.join(analysis_dir, 'stage4_feature_dist.png'))
        plot_tsne_visualization(features4, labels4, os.path.join(analysis_dir, 'stage4_tsne.png'))
    
    if stage5_result:
        print("  Stage5:")
        features5, labels5 = stage5_result
        plot_feature_distribution(features5, os.path.join(analysis_dir, 'stage5_feature_dist.png'))
        plot_class_distribution(labels5, 'Stage5 - After Augmentation',
                               os.path.join(analysis_dir, 'stage5_class_dist.png'))
        plot_tsne_visualization(features5, labels5, os.path.join(analysis_dir, 'stage5_tsne.png'))
    
    print("\n" + "="*60)
    print("分析完成！结果已保存到:")
    print(f"  {analysis_dir}")
    print("="*60)
    
    print("\n分析报告摘要:")
    print("-" * 40)
    print("| Stage | 样本数 | 状态 |")
    print("|-------|--------|------|")
    
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
    
    print(f"| Stage1-3 | {stage3_count} | OK |")
    print(f"| Stage4 | {stage4_count} | OK |")
    print(f"| Stage5 | {stage5_result[0].shape[0] if stage5_result else 0} | OK |")
    print(f"| Stage6 | {len(stage6_files) if stage6_files else 0} files | OK |")
    print("-" * 40)

if __name__ == '__main__':
    main()
