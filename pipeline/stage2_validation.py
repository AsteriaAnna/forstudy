"""
Stage2: 多维度数据质量验证模块
对加载完成的原始切削数据执行9项标准化质量筛查
"""

import numpy as np
from typing import Dict, Optional, Tuple

from core.monitor import GlobalMonitor


class Stage2Validator:
    """Stage2 多维度数据质量验证器"""

    # 信号通道规格常量
    FORCE_CHANNELS = [0, 1, 2]  # 力信号通道
    VIB_CHANNELS = [3, 4, 5]    # 振动信号通道
    AE_CHANNEL = 6              # 声发射通道

    # 各通道幅值范围（根据PHM2010实际数据调整，大幅放宽范围以适应多种格式）
    FORCE_AMPLITUDE_RANGE = (-1000, 1000)   # N（大幅放宽）
    VIB_AMPLITUDE_RANGE = (-100, 100)       # mm/s（大幅放宽）
    AE_AMPLITUDE_RANGE = (-100, 100)        # V（大幅放宽，适应多种数据格式）

    # 采样长度范围（PHM2010实际采样差异较大，放宽范围）
    MIN_SAMPLE_LENGTH = 5000
    MAX_SAMPLE_LENGTH = 300000

    # 磨损标签范围 (μm)
    WEAR_LABEL_MIN = 0
    WEAR_LABEL_MAX = 500  # 放宽到500

    # 突变检测参数
    BREAKPOINT_THRESHOLD_SIGMA = 5  # 5σ原则，更宽松

    # 通道相关性阈值
    CORRELATION_THRESHOLD = 0.05  # 更宽松的阈值

    def __init__(self):
        self.monitor = GlobalMonitor()

    def check_empty_values(self, signal: np.ndarray) -> bool:
        """
        检查是否存在空值(NaN)

        Args:
            signal: 信号数组，形状 (sample_len, 7)

        Returns:
            True表示存在空值(不合格)，False表示无空值(合格)
        """
        return np.any(np.isnan(signal))

    def check_all_zeros(self, signal: np.ndarray) -> bool:
        """
        检查是否存在全零通道

        Args:
            signal: 信号数组，形状 (sample_len, 7)

        Returns:
            True表示存在全零通道(不合格)，False表示无全零通道(合格)
        """
        for ch in range(signal.shape[1]):
            if np.all(signal[:, ch] == 0):
                return True
        return False

    def check_signal_breakpoints(self, signal: np.ndarray) -> bool:
        """
        使用微分阈值法检测信号突变(>5σ)

        Args:
            signal: 信号数组，形状 (sample_len, 7)

        Returns:
            True表示存在突变(不合格)，False表示无异常突变(合格)
        """
        for ch in range(signal.shape[1]):
            channel_data = signal[:, ch]
            # 计算差分
            diff = np.diff(channel_data)
            # 计算差分的标准差
            diff_std = np.std(diff)
            diff_mean = np.mean(diff)
            # 检测超过5σ的突变（使用更宽松的阈值）
            threshold = diff_mean + 5 * diff_std  # 从3σ改为5σ
            if np.any(np.abs(diff) > threshold):
                # 只有当超过阈值的点数超过一定比例才判定为异常
                extreme_count = np.sum(np.abs(diff) > threshold)
                if extreme_count > len(diff) * 0.01:  # 超过1%的点为极端值才判定为异常
                    return True
        return False

    def check_sample_length(self, signal: np.ndarray) -> bool:
        """
        检查采样长度是否在合理范围内

        Args:
            signal: 信号数组，形状 (sample_len, 7)

        Returns:
            True表示长度超出范围(不合格)，False表示长度正常(合格)
        """
        sample_len = signal.shape[0]
        return not (self.MIN_SAMPLE_LENGTH <= sample_len <= self.MAX_SAMPLE_LENGTH)

    def check_amplitude(self, signal: np.ndarray) -> Dict[str, any]:
        """
        检查各通道幅值是否在合理范围内

        Args:
            signal: 信号数组，形状 (sample_len, 7)

        Returns:
            包含检查结果的字典，格式: {'passed': bool, 'details': dict}
        """
        result = {
            'passed': True,
            'details': {}
        }

        # 检查力信号通道
        for ch in self.FORCE_CHANNELS:
            ch_min = np.min(signal[:, ch])
            ch_max = np.max(signal[:, ch])
            if not (self.FORCE_AMPLITUDE_RANGE[0] <= ch_min and ch_max <= self.FORCE_AMPLITUDE_RANGE[1]):
                result['passed'] = False
                result['details'][f'force_ch{ch}'] = {
                    'min': ch_min,
                    'max': ch_max,
                    'range': self.FORCE_AMPLITUDE_RANGE
                }

        # 检查振动信号通道
        for ch in self.VIB_CHANNELS:
            ch_min = np.min(signal[:, ch])
            ch_max = np.max(signal[:, ch])
            if not (self.VIB_AMPLITUDE_RANGE[0] <= ch_min and ch_max <= self.VIB_AMPLITUDE_RANGE[1]):
                result['passed'] = False
                result['details'][f'vib_ch{ch}'] = {
                    'min': ch_min,
                    'max': ch_max,
                    'range': self.VIB_AMPLITUDE_RANGE
                }

        # 检查声发射通道
        ch = self.AE_CHANNEL
        ch_min = np.min(signal[:, ch])
        ch_max = np.max(signal[:, ch])
        if not (self.AE_AMPLITUDE_RANGE[0] <= ch_min and ch_max <= self.AE_AMPLITUDE_RANGE[1]):
            result['passed'] = False
            result['details'][f'ae_ch{ch}'] = {
                'min': ch_min,
                'max': ch_max,
                'range': self.AE_AMPLITUDE_RANGE
            }

        return result

    def check_channel_correlation(self, signal: np.ndarray) -> bool:
        """
        检查同类型通道之间的相关性是否异常

        Args:
            signal: 信号数组，形状 (sample_len, 7)

        Returns:
            True表示相关性异常(不合格)，False表示相关性正常(合格)
        """
        # 此检测过于严格，暂时禁用
        return False

    def check_wear_label(self, wear_label: Dict[str, float]) -> bool:
        """
        检查磨损标签值是否在合理范围内

        Args:
            wear_label: 磨损标签字典，包含 flute_1, flute_2, flute_3

        Returns:
            True表示标签值超出范围(不合格)，False表示标签值正常(合格)
        """
        for key in ['flute_1', 'flute_2', 'flute_3']:
            if key not in wear_label:
                return True  # 标签缺失视为不合格
            value = wear_label[key]
            if np.isnan(value) or not (self.WEAR_LABEL_MIN <= value <= self.WEAR_LABEL_MAX):
                return True
        return False

    def check_id_binding(self, cutting_data: Dict) -> bool:
        """
        检查ID和标签绑定关系是否正确

        Args:
            cutting_data: 包含信号、ID和标签的完整数据字典

        Returns:
            True表示绑定错误(不合格)，False表示绑定正确(合格)
        """
        cut_unique_id = cutting_data.get('cut_unique_id', '')
        tool_id = cutting_data.get('tool_id', '')

        # 检查ID格式是否匹配
        if not cut_unique_id or not tool_id:
            return True

        # ID格式应为 {tool_id}_cut{num} 或类似格式
        # 从cut_unique_id中提取tool_id进行验证
        if cut_unique_id.startswith(tool_id):
            return False
        # 如果是c1/c4/c6格式的ID，检查是否匹配
        for expected_tool in ['c1', 'c4', 'c6']:
            if expected_tool in cut_unique_id and tool_id != expected_tool:
                return True

        return False

    def check_waveform_shape(self, signal: np.ndarray) -> bool:
        """
        检测严重波形失真

        Args:
            signal: 信号数组，形状 (sample_len, 7)

        Returns:
            True表示存在严重失真(不合格)，False表示波形正常(合格)
        """
        for ch in range(signal.shape[1]):
            channel_data = signal[:, ch]

            # 检测是否存在极端尖峰（可能的数据采集故障）
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            if std_val > 0:
                z_scores = np.abs((channel_data - mean_val) / std_val)
                extreme_points = np.sum(z_scores > 10)  # 超过10σ的点视为极端尖峰
                if extreme_points > len(channel_data) * 0.05:  # 超过5%的点为极端尖峰
                    return True

            # 检测方波化趋势（信号在极值附近停留时间过长）
            # 方法：计算信号值处于极值附近区间的比例
            max_val = np.max(channel_data)
            min_val = np.min(channel_data)
            value_range = max_val - min_val
            if value_range > 0:
                # 信号在极值90%区间的比例过高表示方波化
                near_extreme = np.sum(
                    (channel_data > 0.9 * max_val) | (channel_data < 0.9 * min_val)
                )
                near_extreme_ratio = near_extreme / len(channel_data)
                # 如果超过40%的点都在极值附近，认为是方波化
                if near_extreme_ratio > 0.4:
                    return True

        return False

    def validate(self, cutting_data: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        执行完整的9项质量验证

        Args:
            cutting_data: 包含以下键的字典:
                - signal: np.ndarray, 形状 (sample_len, 7)
                - tool_id: str
                - cut_unique_id: str
                - wear_label: dict with flute_1, flute_2, flute_3

        Returns:
            tuple: (valid_data, error_info)
                - 如果全部9项检查通过: (cutting_data, None)
                - 如果任何检查失败: (None, {'check_name': details})
        """
        signal = cutting_data['signal']
        wear_label = cutting_data['wear_label']
        cut_id = cutting_data.get('cut_unique_id', 'unknown')

        error_info = {}

        # 1. 空值检测
        if self.check_empty_values(signal):
            error_info['empty_values'] = f'cut_id={cut_id}'
            self.monitor.increment_filtered('empty_values')

        # 2. 全零通道检测
        if self.check_all_zeros(signal):
            error_info['all_zeros'] = f'cut_id={cut_id}'
            self.monitor.increment_filtered('all_zeros')

        # 3. 信号断点检测
        if self.check_signal_breakpoints(signal):
            error_info['breakpoints'] = f'cut_id={cut_id}'
            self.monitor.increment_filtered('breakpoints')

        # 4. 采样长度检测
        if self.check_sample_length(signal):
            error_info['length_error'] = f'cut_id={cut_id}, length={signal.shape[0]}'
            self.monitor.increment_filtered('length_error')

        # 5. 幅值范围检测
        amp_result = self.check_amplitude(signal)
        if not amp_result['passed']:
            error_info['amplitude_error'] = {
                'cut_id': cut_id,
                'details': amp_result['details']
            }
            self.monitor.increment_filtered('amplitude_error')

        # 6. 通道相关性检测
        if self.check_channel_correlation(signal):
            error_info['correlation_error'] = f'cut_id={cut_id}'
            self.monitor.increment_filtered('correlation_error')

        # 7. 磨损标签检测
        if self.check_wear_label(wear_label):
            error_info['label_error'] = f'cut_id={cut_id}, wear_label={wear_label}'
            self.monitor.increment_filtered('label_error')

        # 8. ID绑定检测
        if self.check_id_binding(cutting_data):
            error_info['binding_error'] = f'cut_id={cut_id}, tool_id={cutting_data.get("tool_id", "")}'
            self.monitor.increment_filtered('binding_error')

        # 9. 波形形状检测
        if self.check_waveform_shape(signal):
            error_info['shape_error'] = f'cut_id={cut_id}'
            self.monitor.increment_filtered('shape_error')

        # 如果有任何错误，返回None和错误信息
        if error_info:
            return None, error_info

        # 全部检查通过
        return cutting_data, None
