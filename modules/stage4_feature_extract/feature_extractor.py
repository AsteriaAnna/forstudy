"""
Stage4: 多域特征提取模块
从预处理后的信号中提取时域、频域、时频域特征
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pywt
from scipy import stats
from scipy.fft import fft
from scipy.signal import find_peaks

from core.logger import Logger
from core.utils import ensure_dir, get_interim_dir


class FeatureExtractor:
    """多域特征提取器

    从7通道预处理信号中提取175维特征向量
    - 时域特征: 15个/通道
    - 频域特征: 5个/通道
    - 时频域特征: 5个/通道 (4层能量 + 小波熵)
    """
    
    # 通道名称
    CHANNEL_NAMES = ['force_x', 'force_y', 'force_z', 'vib_x', 'vib_y', 'vib_z', 'ae']
    NUM_CHANNELS = 7
    
    # 时域特征数量
    NUM_TIME_FEATURES = 15
    # 频域特征数量
    NUM_FREQ_FEATURES = 5
    # 时频域特征数量
    NUM_TIME_FREQ_FEATURES = 5
    # 总特征数量
    TOTAL_FEATURES = NUM_CHANNELS * (NUM_TIME_FEATURES + NUM_FREQ_FEATURES + NUM_TIME_FREQ_FEATURES)  # 175
    
    def __init__(self, run_id: str, logger: Optional[Logger] = None):
        """初始化特征提取器
        
        Args:
            run_id: 运行唯一标识符
            logger: 日志记录器实例
        """
        self.run_id = run_id
        self.logger = logger or Logger(run_id)
        self._stats = {
            'total_processed': 0,
            'valid_features': 0,
            'invalid_features': 0,
            'nan_count': 0,
            'inf_count': 0
        }
    
    def extract(self, signal: np.ndarray) -> np.ndarray:
        """提取完整特征向量
        
        Args:
            signal: 输入信号 (N, 7)
            
        Returns:
            168维特征向量
        """
        if signal.ndim != 2 or signal.shape[1] != self.NUM_CHANNELS:
            raise ValueError(f"信号维度错误，期望(N, 7)，实际{signal.shape}")
        
        # 提取三类特征
        time_features = self.extract_time_features(signal)    # (105,)
        freq_features = self.extract_freq_features(signal)     # (35,)
        timefreq_features = self.extract_timefreq_features(signal)  # (28,)
        
        # 拼接所有特征
        feature_vector = np.concatenate([
            time_features,
            freq_features,
            timefreq_features
        ])
        
        # 更新统计
        self._stats['total_processed'] += 1
        if self.validate_features(feature_vector):
            self._stats['valid_features'] += 1
        else:
            self._stats['invalid_features'] += 1
        
        return feature_vector
    
    def extract_time_features(self, signal: np.ndarray) -> np.ndarray:
        """提取时域特征
        
        Args:
            signal: 输入信号 (N, 7)
            
        Returns:
            105维时域特征向量 (15特征 × 7通道)
        """
        time_features = []
        
        for ch in range(self.NUM_CHANNELS):
            ch_signal = signal[:, ch]
            features = self._compute_time_features(ch_signal)
            time_features.extend(features)
        
        return np.array(time_features, dtype=np.float32)
    
    def _compute_time_features(self, x: np.ndarray) -> List[float]:
        """计算单通道时域特征
        
        Args:
            x: 单通道信号 (N,)
            
        Returns:
            15个时域特征列表
        """
        features = []
        
        # 基本统计量
        mean_val = np.mean(x)
        variance = np.var(x)
        std_val = np.std(x)
        
        # 1. mean - 均值
        features.append(mean_val)
        
        # 2. variance - 方差
        features.append(variance)
        
        # 3. std - 标准差
        features.append(std_val)
        
        # 4. rms - 均方根
        rms = np.sqrt(np.mean(x**2))
        features.append(rms)
        
        # 5. peak_to_peak - 峰峰值
        peak_to_peak = np.max(x) - np.min(x)
        features.append(peak_to_peak)
        
        # 6. skewness - 偏度
        if std_val > 1e-10:
            skewness = stats.skew(x)
        else:
            skewness = 0.0
        features.append(skewness)
        
        # 7. kurtosis - 峭度
        if std_val > 1e-10:
            kurtosis = stats.kurtosis(x)
        else:
            kurtosis = 0.0
        features.append(kurtosis)
        
        # 8. crest_factor - 峰值因子 (peak / rms)
        peak = np.max(np.abs(x))
        if rms > 1e-10:
            crest_factor = peak / rms
        else:
            crest_factor = 0.0
        features.append(crest_factor)
        
        # 9. waveform_factor - 波形因子 (rms / mean_abs)
        mean_abs = np.mean(np.abs(x))
        if mean_abs > 1e-10:
            waveform_factor = rms / mean_abs
        else:
            waveform_factor = 0.0
        features.append(waveform_factor)
        
        # 10. impulse_factor - 冲击因子 (peak / mean_abs)
        if mean_abs > 1e-10:
            impulse_factor = peak / mean_abs
        else:
            impulse_factor = 0.0
        features.append(impulse_factor)
        
        # 11. margin_factor - 裕度因子 (peak / rms_of_squared)
        # rms_of_squared = sqrt(mean(x^2)) = rms
        # 实际上裕度因子是 peak / sqrt(mean(squared_signal))
        squared_signal = x ** 2
        rms_squared = np.sqrt(np.mean(squared_signal))
        if rms_squared > 1e-10:
            margin_factor = peak / rms_squared
        else:
            margin_factor = 0.0
        features.append(margin_factor)
        
        # 12. energy - 能量
        energy = np.sum(x**2)
        features.append(energy)
        
        # 13. entropy - 熵 (信号能量分布的熵)
        energy_norm = energy / (len(x) + 1e-10)
        if energy_norm > 1e-10:
            # 使用能量归一化后计算熵
            hist, _ = np.histogram(x, bins=50, density=True)
            hist = hist[hist > 0]  # 去掉零值
            if len(hist) > 0:
                entropy = -np.sum(hist * np.log(hist + 1e-10))
            else:
                entropy = 0.0
        else:
            entropy = 0.0
        features.append(entropy)
        
        # 14. zero_crossing_rate - 零 crossing率
        zero_crossings = np.sum(np.abs(np.diff(np.sign(x)))) / 2
        zcr = zero_crossings / len(x)
        features.append(zcr)
        
        # 15. mean_derivative - 平均变化率
        derivative = np.diff(x)
        mean_derivative = np.mean(np.abs(derivative))
        features.append(mean_derivative)
        
        return features
    
    def extract_freq_features(self, signal: np.ndarray) -> np.ndarray:
        """提取频域特征
        
        Args:
            signal: 输入信号 (N, 7)
            
        Returns:
            35维频域特征向量 (5特征 × 7通道)
        """
        freq_features = []
        
        for ch in range(self.NUM_CHANNELS):
            ch_signal = signal[:, ch]
            features = self._compute_freq_features(ch_signal)
            freq_features.extend(features)
        
        return np.array(freq_features, dtype=np.float32)
    
    def _compute_freq_features(self, x: np.ndarray) -> List[float]:
        """计算单通道频域特征
        
        Args:
            x: 单通道信号 (N,)
            
        Returns:
            5个频域特征列表
        """
        # 计算FFT
        n = len(x)
        fft_vals = fft(x)
        fft_magnitude = np.abs(fft_vals[:n//2])
        fft_magnitude = fft_magnitude / (n + 1e-10)  # 归一化
        
        # 频率轴 - 使用真实物理频率 (PHM2010采样频率为20000Hz)
        fs = 20000
        freqs = np.fft.fftfreq(n, 1/fs)[:n//2]
        
        # 1. spectral_mean - 频谱均值
        spectral_mean = np.mean(fft_magnitude)
        
        # 2. frequency_center - 频率中心(重心)
        if np.sum(fft_magnitude) > 1e-10:
            frequency_center = np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude)
        else:
            frequency_center = 0.0
        
        # 3. spectral_variance - 频谱方差
        if np.sum(fft_magnitude) > 1e-10:
            spectral_variance = np.sum(((freqs - frequency_center)**2) * fft_magnitude) / np.sum(fft_magnitude)
        else:
            spectral_variance = 0.0
        
        # 4. spectral_entropy - 谱熵
        fft_magnitude_norm = fft_magnitude / (np.sum(fft_magnitude) + 1e-10)
        fft_magnitude_norm = fft_magnitude_norm[fft_magnitude_norm > 0]
        if len(fft_magnitude_norm) > 0:
            spectral_entropy = -np.sum(fft_magnitude_norm * np.log(fft_magnitude_norm + 1e-10))
        else:
            spectral_entropy = 0.0
        
        # 5. spectral_peak - 谱峰值
        spectral_peak = np.max(fft_magnitude)
        
        return [spectral_mean, frequency_center, spectral_variance, spectral_entropy, spectral_peak]
    
    def extract_timefreq_features(self, signal: np.ndarray) -> np.ndarray:
        """提取时频域特征
        
        Args:
            signal: 输入信号 (N, 7)
            
        Returns:
            28维时频域特征向量 (4特征 × 7通道)
        """
        timefreq_features = []
        
        for ch in range(self.NUM_CHANNELS):
            ch_signal = signal[:, ch]
            features = self._compute_timefreq_features(ch_signal)
            timefreq_features.extend(features)
        
        return np.array(timefreq_features, dtype=np.float32)
    
    def _compute_timefreq_features(self, x: np.ndarray) -> List[float]:
        """计算单通道时频域特征
        
        使用db4小波进行4层分解
        
        Args:
            x: 单通道信号 (N,)
            
        Returns:
            4个时频域特征列表 (4层能量 + 1个小波熵 = 实际上4能量特征 + 1熵 = 5?
            规格说明是4个特征每通道: 4层能量 + wavelet_entropy = 5个
            但按规格是4个每通道，重新理解: 能量每层(4个) + wavelet_entropy(1个) = 5
            但规格说4个每通道，可能是: 每层能量作为4个特征
            再看规格: "Wavelet decomposition (4 levels, db4 wavelet) energy per level (4 features) + wavelet_entropy"
            看来是4个能量特征 + 1个熵 = 5个，但我们按规格说4个每通道...
            实际上4 levels的energy per level就是4个特征，加上wavelet_entropy是第5个
            但规格说4个每通道，可能是wavelet_entropy不算？或者只算能量？
            重新看: "Time-Frequency Features (4 per channel): Wavelet decomposition (4 levels, db4 wavelet) energy per level (4 features) + wavelet_entropy"
            这看起来是5个...但规格明确说4 per channel
            理解为: 4个能量特征本身就是一个综合特征，或者只取4层能量作为特征
            
            按任务要求是4个每通道，我们取4个能量特征（4层分解的能量）
        """
        # 小波分解 (4层, db4)
        wavelet = 'db4'
        max_level = 4
        
        # 计算分解层数
        actual_level = min(max_level, pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len))
        
        # 进行小波分解
        coeffs = pywt.wavedec(x, wavelet, level=actual_level)
        
        # cA是近似系数, cD是细节系数
        # level 1 = cD[-1], level 2 = cD[-2], etc.
        # 最后一层是cA
        
        features = []
        
        # 细节系数能量 (从高层到低层)，直接使用系数平方和，不归一化
        for i in range(actual_level - 1, -1, -1):
            cD_energy = np.sum(coeffs[i + 1]**2)
            features.append(cD_energy)
        
        # 如果实际层数不足4层，用0填充
        while len(features) < 4:
            features.append(0.0)

        # 计算小波熵 (wavelet_entropy)
        # 使用所有系数的归一化能量分布计算熵
        all_coeffs = []
        for c in coeffs:
            all_coeffs.extend(c.flatten())
        all_coeffs = np.array(all_coeffs)
        
        # 计算各系数的能量
        energy_array = all_coeffs ** 2
        total_energy = np.sum(energy_array)
        
        if total_energy > 1e-10:
            # 归一化能量分布
            p = energy_array / total_energy
            # 移除零值避免 log(0)
            p = p[p > 1e-10]
            # 计算熵 (使用log2为单位)
            wavelet_entropy = -np.sum(p * np.log2(p + 1e-10))
        else:
            wavelet_entropy = 0.0
        
        features.append(wavelet_entropy)

        return features
    
    def validate_features(self, features: np.ndarray) -> bool:
        """验证特征向量的合法性
        
        Args:
            features: 特征向量
            
        Returns:
            True if valid (no NaN/Inf), False otherwise
        """
        if features is None:
            return False
        
        if isinstance(features, list):
            features = np.array(features)
        
        # 检查NaN
        nan_count = np.sum(np.isnan(features))
        self._stats['nan_count'] += nan_count
        
        # 检查Inf
        inf_count = np.sum(np.isinf(features))
        self._stats['inf_count'] += inf_count
        
        if nan_count > 0:
            self.logger.warn(f"特征向量包含{nan_count}个NaN值")
        
        if inf_count > 0:
            self.logger.warn(f"特征向量包含{inf_count}个Inf值")
        
        return nan_count == 0 and inf_count == 0
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表

        Returns:
            175个特征名称列表
        """
        time_feature_names = [
            'mean', 'variance', 'std', 'rms', 'peak_to_peak',
            'skewness', 'kurtosis', 'crest_factor', 'waveform_factor',
            'impulse_factor', 'margin_factor', 'energy', 'entropy',
            'zero_crossing_rate', 'mean_derivative'
        ]
        
        freq_feature_names = [
            'spectral_mean', 'frequency_center', 'spectral_variance',
            'spectral_entropy', 'spectral_peak'
        ]
        
        timefreq_feature_names = [
            'wavelet_energy_level1', 'wavelet_energy_level2',
            'wavelet_energy_level3', 'wavelet_energy_level4',
            'wavelet_entropy'
        ]
        
        feature_names = []
        for ch in self.CHANNEL_NAMES:
            for name in time_feature_names:
                feature_names.append(f"{ch}_{name}")
            for name in freq_feature_names:
                feature_names.append(f"{ch}_{name}")
            for name in timefreq_feature_names:
                feature_names.append(f"{ch}_{name}")
        
        return feature_names
    
    def get_statistics(self) -> Dict:
        """获取特征提取统计信息
        
        Returns:
            统计信息字典
        """
        return dict(self._stats)
    
    def reset_statistics(self):
        """重置统计计数器"""
        self._stats = {
            'total_processed': 0,
            'valid_features': 0,
            'invalid_features': 0,
            'nan_count': 0,
            'inf_count': 0
        }


class BatchFeatureExtractor:
    """批量特征提取器"""
    
    def __init__(
        self, 
        run_id: str, 
        logger: Optional[Logger] = None,
        normalization_method: Optional[str] = None,
        feature_selector=None
    ):
        """初始化批量特征提取器
        
        Args:
            run_id: 运行唯一标识符
            logger: 日志记录器实例
            normalization_method: 归一化方法，'standard' 或 'minmax'，None表示不进行归一化
            feature_selector: 特征选择器实例（如CVOCA），None表示不进行特征选择
        """
        self.run_id = run_id
        self.logger = logger or Logger(run_id)
        self.extractor = FeatureExtractor(run_id, logger)
        self._results = []
        
        # 归一化相关
        self.normalization_method = normalization_method
        self.scaler = self._create_scaler(normalization_method)
        
        # 特征选择相关
        self.feature_selector = feature_selector
        self.selected_indices = None
    
    def _create_scaler(self, normalization_method: Optional[str]):
        """创建归一化器
        
        Args:
            normalization_method: 归一化方法
            
        Returns:
            归一化器实例或None
        """
        if normalization_method is None:
            return None
        
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        if normalization_method == 'standard':
            return StandardScaler()
        elif normalization_method == 'minmax':
            return MinMaxScaler()
        else:
            raise ValueError(f"未知的归一化方法: {normalization_method}")
    
    def process_batch(
        self,
        signal_list: List[np.ndarray],
        cut_ids: Optional[List[str]] = None,
        tool_ids: Optional[List[str]] = None,
        wear_labels: Optional[List[Dict]] = None,
        save_dir: Optional[str] = None
    ) -> List[Dict]:
        """批量处理信号列表
        
        Args:
            signal_list: 信号列表，每个信号形状 (N, 7)
            cut_ids: 切削唯一ID列表
            tool_ids: 刀具ID列表
            wear_labels: 磨损标签列表
            save_dir: 保存目录，若不指定则不保存
            
        Returns:
            特征提取结果列表，包含原始特征、归一化特征（如启用）、选择后的特征（如启用）
        """
        n_samples = len(signal_list)
        
        # 设置默认值
        if cut_ids is None:
            cut_ids = [f"unknown_{i}" for i in range(n_samples)]
        if tool_ids is None:
            tool_ids = ["unknown" for _ in range(n_samples)]
        if wear_labels is None:
            wear_labels = [{"flute_1": 0.0, "flute_2": 0.0, "flute_3": 0.0} for _ in range(n_samples)]
        
        # 确保保存目录存在
        if save_dir is not None:
            ensure_dir(save_dir)
        
        results = []
        feature_matrix = []
        
        # 第一步：批量提取特征
        for i, signal in enumerate(signal_list):
            cut_id = cut_ids[i]
            
            try:
                # 提取特征
                feature_vector = self.extractor.extract(signal)
                
                # 验证特征
                if not self.extractor.validate_features(feature_vector):
                    self.logger.error(f"特征无效: {cut_id}", cut_id)
                    continue
                
                # 构建结果
                result = {
                    'feature_vector': feature_vector,
                    'cut_unique_id': cut_id,
                    'tool_id': tool_ids[i],
                    'wear_label': wear_labels[i]
                }
                results.append(result)
                feature_matrix.append(feature_vector)
                
                self.logger.info(f"特征提取成功: {cut_id}", cut_id)
                
            except Exception as e:
                self.logger.error(f"特征提取失败: {cut_id}", cut_id, exc=e)
                continue
        
        if len(feature_matrix) == 0:
            self.logger.warn("没有有效特征数据")
            return results
        
        # 转换为numpy数组
        feature_matrix = np.array(feature_matrix)
        
        # 第二步：特征归一化（如启用）
        if self.normalization_method is not None:
            feature_matrix = self._normalize_features(feature_matrix)
        
        # 第三步：特征选择（如启用）
        if self.feature_selector is not None:
            # 提取磨损标签作为特征选择的目标
            y = np.array([r['wear_label']['robust_wear'] for r in results])
            feature_matrix = self._select_features(feature_matrix, y)
        
        # 更新结果中的特征向量
        for i, result in enumerate(results):
            result['feature_vector'] = feature_matrix[i]
            
            # 保存单个特征文件
            if save_dir is not None:
                self._save_feature(result, save_dir)
        
        # 输出统计信息
        self.logger.stat(
            f"批量处理完成: 总数={n_samples}, 成功={len(results)}, "
            f"失败={n_samples - len(results)}, "
            f"特征维度={feature_matrix.shape[1]}"
        )
        
        return results
    
    def _normalize_features(self, feature_matrix: np.ndarray) -> np.ndarray:
        """对特征矩阵进行归一化
        
        Args:
            feature_matrix: 特征矩阵，形状 (n_samples, n_features)
            
        Returns:
            归一化后的特征矩阵
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        if self.normalization_method == 'standard':
            self.scaler = StandardScaler()
        elif self.normalization_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"未知的归一化方法: {self.normalization_method}")
        
        normalized_matrix = self.scaler.fit_transform(feature_matrix)
        self.logger.info(f"特征归一化完成: 方法={self.normalization_method}, 特征维度={normalized_matrix.shape[1]}")
        
        return normalized_matrix
    
    def _select_features(self, feature_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
        """使用特征选择器进行特征选择
        
        Args:
            feature_matrix: 特征矩阵，形状 (n_samples, n_features)
            y: 目标变量（磨损标签）
            
        Returns:
            选择后的特征矩阵
        """
        try:
            # 尝试使用fit_transform方法
            if hasattr(self.feature_selector, 'fit_transform'):
                selected_matrix = self.feature_selector.fit_transform(feature_matrix, y)
            elif hasattr(self.feature_selector, 'fit') and hasattr(self.feature_selector, 'transform'):
                self.feature_selector.fit(feature_matrix, y)
                selected_matrix = self.feature_selector.transform(feature_matrix)
            else:
                self.logger.warn("特征选择器不支持fit_transform或fit+transform方法")
                return feature_matrix
            
            # 尝试获取选择的特征索引
            if hasattr(self.feature_selector, 'get_support'):
                self.selected_indices = np.where(self.feature_selector.get_support())[0]
            elif hasattr(self.feature_selector, 'selected_indices'):
                self.selected_indices = self.feature_selector.selected_indices
            
            self.logger.info(f"特征选择完成: 原始维度={feature_matrix.shape[1]}, 选择后维度={selected_matrix.shape[1]}")
            
            return selected_matrix
        
        except Exception as e:
            self.logger.error(f"特征选择失败", exc=e)
            return feature_matrix
    
    def _save_feature(self, result: Dict, save_dir: str):
        """保存单个特征到文件
        
        Args:
            result: 特征结果字典
            save_dir: 保存目录
        """
        cut_id = result['cut_unique_id']
        
        # 保存特征向量
        feature_path = os.path.join(save_dir, f"features_{cut_id}.npy")
        np.save(feature_path, result['feature_vector'])
        
        # 保存元信息
        meta_path = os.path.join(save_dir, f"meta_{cut_id}.npz")
        np.savez(
            meta_path,
            tool_id=result['tool_id'],
            wear_label=result['wear_label']
        )
    
    def load_processed_features(
        self,
        feature_dir: str,
        cut_ids: List[str]
    ) -> List[Dict]:
        """加载已处理的特征
        
        Args:
            feature_dir: 特征文件目录
            cut_ids: 要加载的切削ID列表
            
        Returns:
            特征结果列表
        """
        results = []
        
        for cut_id in cut_ids:
            feature_path = os.path.join(feature_dir, f"features_{cut_id}.npy")
            meta_path = os.path.join(feature_dir, f"meta_{cut_id}.npz")
            
            if not os.path.exists(feature_path):
                self.logger.warn(f"特征文件不存在: {cut_id}", cut_id)
                continue
            
            # 加载特征
            feature_vector = np.load(feature_path)
            
            # 加载元信息
            meta = np.load(meta_path, allow_pickle=True)
            tool_id = str(meta['tool_id'])
            wear_label = meta['wear_label'].item()
            
            result = {
                'feature_vector': feature_vector,
                'cut_unique_id': cut_id,
                'tool_id': tool_id,
                'wear_label': wear_label
            }
            results.append(result)
        
        return results


def extract_features_from_signal(
    signal: np.ndarray,
    cut_unique_id: str,
    tool_id: str,
    wear_label: Dict,
    run_id: str
) -> Dict:
    """从信号提取特征的便捷函数
    
    Args:
        signal: 预处理后的信号 (N, 7)
        cut_unique_id: 切削唯一ID
        tool_id: 刀具ID
        wear_label: 磨损标签字典
        run_id: 运行ID
        
    Returns:
        特征提取结果字典
    """
    extractor = FeatureExtractor(run_id)
    feature_vector = extractor.extract(signal)
    
    return {
        'feature_vector': feature_vector,
        'cut_unique_id': cut_unique_id,
        'tool_id': tool_id,
        'wear_label': wear_label
    }


if __name__ == "__main__":
    # 测试代码
    run_id = Logger.generate_run_id(1)
    logger = Logger(run_id)
    
    # 创建随机测试信号 (1000采样点, 7通道)
    test_signal = np.random.randn(1000, 7)
    
    # 测试特征提取器
    extractor = FeatureExtractor(run_id, logger)
    
    # 提取特征
    features = extractor.extract(test_signal)
    
    print(f"特征向量维度: {features.shape}")
    print(f"预期维度: ({FeatureExtractor.TOTAL_FEATURES},)")
    print(f"验证结果: {extractor.validate_features(features)}")
    
    # 测试批量处理
    batch_extractor = BatchFeatureExtractor(run_id, logger)
    
    # 生成测试数据
    test_signals = [np.random.randn(1000, 7) for _ in range(5)]
    test_cut_ids = [f"c1_cut{i}" for i in range(1, 6)]
    test_tool_ids = ["c1"] * 5
    test_wear_labels = [
        {"flute_1": 0.1 * i, "flute_2": 0.2 * i, "flute_3": 0.15 * i}
        for i in range(1, 6)
    ]
    
    # 批量处理
    results = batch_extractor.process_batch(
        test_signals,
        test_cut_ids,
        test_tool_ids,
        test_wear_labels
    )
    
    print(f"\n批量处理结果数量: {len(results)}")
    
    # 输出统计
    print(f"\n统计信息: {extractor.get_statistics()}")
    
    logger.close()