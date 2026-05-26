"""Stage3: 信号预处理模块

该模块对Stage2验证后的切割数据进行信号预处理，包括:
- 带通滤波去除噪声
- 趋势去除
"""

import numpy as np
from scipy.signal import butter, filtfilt

from core.logger import Logger


class Stage3Preprocessor:
    """Stage3信号预处理器"""

    def __init__(self, run_id: str = None):
        """初始化预处理器

        Args:
            run_id: 运行唯一标识符，用于日志记录
        """
        self.logger = Logger(run_id) if run_id else Logger("default_run_id")

    def bandpass_filter(self, signal: np.ndarray, fs: int = 20000) -> np.ndarray:
        """带通滤波

        使用Butterworth 4阶带通滤波器，保留[100, 5000]Hz频率成分

        Args:
            signal: 输入信号，形状为(N, 7)的numpy数组
            fs: 采样频率，默认20000Hz

        Returns:
            滤波后的信号，形状为(N, 7)
        """
        low_freq = 100
        high_freq = 5000
        nyquist = fs / 2

        low = low_freq / nyquist
        high = high_freq / nyquist

        b, a = butter(4, [low, high], btype='band')

        filtered_signal = np.zeros_like(signal)
        for i in range(signal.shape[1]):
            channel_data = signal[:, i]
            filtered_signal[:, i] = filtfilt(b, a, channel_data)

        if not np.all(np.isfinite(filtered_signal)):
            self.logger.error("带通滤波产生NaN或Inf值")
            raise ValueError("Filtering produced NaN or Inf values")

        self.logger.info(
            f"带通滤波完成: 采样率={fs}Hz, 通带=[{low_freq}, {high_freq}]Hz, 阶数=4"
        )

        return filtered_signal

    def remove_trend(self, signal: np.ndarray) -> np.ndarray:
        """趋势去除

        对每个通道进行线性拟合，去除线性趋势

        Args:
            signal: 输入信号，形状为(N, 7)的numpy数组

        Returns:
            去趋势后的信号，形状为(N, 7)
        """
        detrended_signal = np.zeros_like(signal)
        x = np.arange(signal.shape[0])

        for i in range(signal.shape[1]):
            channel = signal[:, i]
            coeffs = np.polyfit(x, channel, 1)
            trend = np.polyval(coeffs, x)
            detrended_signal[:, i] = channel - trend

        self.logger.info("趋势去除完成: 线性拟合阶数=1")

        return detrended_signal

    def preprocess(self, cutting_data: dict) -> dict:
        """完整预处理流程

        依次执行趋势去除、带通滤波

        Args:
            cutting_data: Stage2验证后的切割数据字典，包含:
                - 'signal': 原始信号数组 (N, 7)
                - 'cut_unique_id': 切割唯一标识符
                - 'tool_id': 刀具ID
                - 'wear_label': 磨损标签字典

        Returns:
            预处理后的数据字典，包含:
                - 'processed_signal': 预处理后的信号数组 (N, 7)
                - 'cut_unique_id': 切割唯一标识符
                - 'tool_id': 刀具ID
                - 'wear_label': 磨损标签字典
                - 'status': 处理状态 'PROCESSED'
            如果处理失败返回None
        """
        cut_unique_id = cutting_data.get('cut_unique_id', 'unknown')

        self.logger.info(f"开始预处理: cut_unique_id={cut_unique_id}", cut_unique_id)

        signal = cutting_data.get('signal')
        if signal is None:
            self.logger.error("输入数据缺少'signal'字段", cut_unique_id)
            return None

        try:
            detrended = self.remove_trend(signal)
            self.logger.info(f"步骤1/2 - 趋势去除完成: 信号形状={detrended.shape}", cut_unique_id)

        except Exception as e:
            self.logger.error(f"趋势去除失败", cut_unique_id, exc=e)
            return None

        try:
            filtered = self.bandpass_filter(detrended)
            self.logger.info(f"步骤2/2 - 带通滤波完成: 信号形状={filtered.shape}", cut_unique_id)

        except Exception as e:
            self.logger.error(f"带通滤波失败", cut_unique_id, exc=e)
            return None

        preprocessed_data = {
            'processed_signal': filtered,
            'cut_unique_id': cutting_data.get('cut_unique_id'),
            'tool_id': cutting_data.get('tool_id'),
            'wear_label': cutting_data.get('wear_label'),
            'status': 'PROCESSED'
        }

        self.logger.info(
            f"预处理成功完成: cut_unique_id={cut_unique_id}, "
            f"tool_id={cutting_data.get('tool_id')}, "
            f"信号形状={filtered.shape}, "
            f"状态=PROCESSED",
            cut_unique_id
        )

        return preprocessed_data
