import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_raw_signal(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    return data

def plot_raw_signal_waveform(signal_data, output_path, title="Raw Signal Waveform"):
    n_channels = signal_data.shape[1]
    rows = (n_channels + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, rows * 3))
    axes = axes.flatten()
    
    channel_names = ['X_acc', 'Y_acc', 'Z_acc', 'X_force', 'Y_force', 'Z_force', 'Spindle']
    
    n_samples = min(1000, signal_data.shape[0])
    
    for i in range(n_channels):
        axes[i].plot(signal_data[:n_samples, i], linewidth=0.5, color='blue')
        axes[i].set_title(f'Channel {i+1}: {channel_names[i]}')
        axes[i].set_xlabel('Sample')
        axes[i].set_ylabel('Amplitude')
        axes[i].grid(True, alpha=0.3)
        
        ch_min = signal_data[:, i].min()
        ch_max = signal_data[:, i].max()
        ch_mean = signal_data[:, i].mean()
        ch_std = signal_data[:, i].std()
        axes[i].text(0.02, 0.98, f'min={ch_min:.3f}\nmax={ch_max:.3f}\nmean={ch_mean:.3f}\nstd={ch_std:.3f}',
                    transform=axes[i].transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    for i in range(n_channels, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  保存: {output_path}")

def plot_comparison_waveform(raw_data, processed_data, output_path, title="Signal Comparison"):
    n_channels = min(raw_data.shape[1], processed_data.shape[1])
    fig, axes = plt.subplots(n_channels, 2, figsize=(14, n_channels * 2.5))
    
    channel_names = ['X_acc', 'Y_acc', 'Z_acc', 'X_force', 'Y_force', 'Z_force', 'Spindle']
    n_samples = min(1000, raw_data.shape[0], processed_data.shape[0])
    
    for i in range(n_channels):
        axes[i, 0].plot(raw_data[:n_samples, i], linewidth=0.5, color='blue')
        axes[i, 0].set_title(f'{channel_names[i]} - Raw (Before Preprocessing)')
        axes[i, 0].set_xlabel('Sample')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].plot(processed_data[:n_samples, i], linewidth=0.5, color='green')
        axes[i, 1].set_title(f'{channel_names[i]} - Processed (After Preprocessing)')
        axes[i, 1].set_xlabel('Sample')
        axes[i, 1].set_ylabel('Amplitude')
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  保存: {output_path}")

def main():
    result_dir = 'results/run_007_20260524_2111'
    analysis_dir = os.path.join(result_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    print("="*60)
    print("绘制预处理前的原始信号波形图")
    print("="*60)
    
    raw_data_dir = 'data/raw/phm2010_raw'
    
    sample_files = [
        ('c1/c1/c_1_001.csv', 'Tool C1, Cut 1'),
        ('c1/c1/c_1_010.csv', 'Tool C1, Cut 10'),
        ('c4/c4/c_4_001.csv', 'Tool C4, Cut 1'),
        ('c6/c6/c_6_001.csv', 'Tool C6, Cut 1'),
    ]
    
    print("\n绘制原始信号波形图:")
    for rel_path, title in sample_files:
        file_path = os.path.join(raw_data_dir, rel_path)
        if os.path.exists(file_path):
            raw_data = load_raw_signal(file_path)
            output_name = rel_path.replace('/', '_').replace('.csv', '_raw.png')
            output_path = os.path.join(analysis_dir, output_name)
            plot_raw_signal_waveform(raw_data, output_path, f"Raw Signal - {title}")
            print(f"    {title}: shape={raw_data.shape}")
        else:
            print(f"    文件不存在: {file_path}")
    
    print("\n绘制预处理前后对比图:")
    stage3_dir = os.path.join(result_dir, 'data', 'cache', 'stage3_dataset_creation')
    
    comparison_samples = [
        ('c1_cut1.npy', 'c1/c1/c_1_001.csv', 'Tool C1, Cut 1'),
    ]
    
    for processed_file, raw_file, title in comparison_samples:
        processed_path = os.path.join(stage3_dir, processed_file)
        raw_path = os.path.join(raw_data_dir, raw_file)
        
        if os.path.exists(processed_path) and os.path.exists(raw_path):
            processed_data = np.load(processed_path, allow_pickle=True)
            raw_data = load_raw_signal(raw_path)
            
            output_path = os.path.join(analysis_dir, 'signal_comparison_c1_cut1.png')
            plot_comparison_waveform(raw_data, processed_data, output_path, f"Signal Comparison - {title}")
            print(f"    对比图已生成: {title}")
    
    print("\n" + "="*60)
    print("完成！所有波形图已保存到:")
    print(f"  {analysis_dir}")
    print("="*60)

if __name__ == '__main__':
    main()
