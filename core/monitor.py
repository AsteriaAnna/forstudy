"""
PHM2010 项目全局监控模块
用于跟踪文件加载、处理和过滤的全局统计信息
"""

from typing import Dict


class GlobalMonitor:
    """全局监控器单例类，用于跟踪全局统计数据"""
    
    _instance = None
    
    # Cut status constants
    LOADED = "LOADED"
    VALID = "VALID"
    FILTERED = "FILTERED"
    PROCESSED = "PROCESSED"
    SKIPPED = "SKIPPED"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.reset()
    
    def reset(self):
        """重置所有计数器，为新的运行做准备"""
        self.total_loaded = 0
        self.success_count = 0
        self.filtered_count = 0
        
        # 各种异常计数器
        self.empty_count = 0
        self.zero_count = 0
        self.length_error_count = 0
        self.amplitude_error_count = 0
        self.other_error_count = 0

        # Stage2 9项检测错误计数器
        self.empty_values_count = 0       # 空值检测
        self.all_zeros_count = 0           # 全零通道检测
        self.breakpoints_count = 0         # 信号断点检测
        self.length_error_count = 0        # 采样长度检测
        self.amplitude_error_count = 0    # 幅值范围检测
        self.correlation_error_count = 0  # 通道相关性检测
        self.label_error_count = 0        # 磨损标签检测
        self.binding_error_count = 0      # ID绑定检测
        self.shape_error_count = 0        # 波形形状检测
        
        # 过滤原因详情
        self.filter_reasons: Dict[str, int] = {}
    
    def increment_loaded(self):
        """增加总加载计数"""
        self.total_loaded += 1
    
    def increment_success(self):
        """增加成功计数"""
        self.success_count += 1
    
    def increment_filtered(self, reason: str):
        """
        增加过滤计数
        
        Args:
            reason: 过滤原因
        """
        self.filtered_count += 1
        
        # 记录具体过滤原因
        if reason in self.filter_reasons:
            self.filter_reasons[reason] += 1
        else:
            self.filter_reasons[reason] = 1
        
        # 更新对应的异常计数
        if reason == "empty":
            self.empty_count += 1
        elif reason == "zero":
            self.zero_count += 1
        elif reason == "length_error":
            self.length_error_count += 1
        elif reason == "amplitude_exceeded":
            self.amplitude_error_count += 1
        # Stage2 9项检测错误类型
        elif reason == "empty_values":
            self.empty_values_count += 1
        elif reason == "all_zeros":
            self.all_zeros_count += 1
        elif reason == "breakpoints":
            self.breakpoints_count += 1
        elif reason == "length_error":
            self.length_error_count += 1
        elif reason == "amplitude_error":
            self.amplitude_error_count += 1
        elif reason == "correlation_error":
            self.correlation_error_count += 1
        elif reason == "label_error":
            self.label_error_count += 1
        elif reason == "binding_error":
            self.binding_error_count += 1
        elif reason == "shape_error":
            self.shape_error_count += 1
        else:
            self.other_error_count += 1
    
    def get_summary(self) -> Dict:
        """
        获取所有统计信息的摘要
        
        Returns:
            包含所有统计数据的字典
        """
        return {
            "total_loaded": self.total_loaded,
            "success_count": self.success_count,
            "filtered_count": self.filtered_count,
            "empty_count": self.empty_count,
            "zero_count": self.zero_count,
            "length_error_count": self.length_error_count,
            "amplitude_error_count": self.amplitude_error_count,
            "other_error_count": self.other_error_count,
            "filter_reasons": dict(self.filter_reasons),
            "success_rate": self._calculate_success_rate(),
            # Stage2 9项检测错误统计
            "stage2_empty_values_count": self.empty_values_count,
            "stage2_all_zeros_count": self.all_zeros_count,
            "stage2_breakpoints_count": self.breakpoints_count,
            "stage2_length_error_count": self.length_error_count,
            "stage2_amplitude_error_count": self.amplitude_error_count,
            "stage2_correlation_error_count": self.correlation_error_count,
            "stage2_label_error_count": self.label_error_count,
            "stage2_binding_error_count": self.binding_error_count,
            "stage2_shape_error_count": self.shape_error_count,
        }
    
    def _calculate_success_rate(self) -> float:
        """计算成功率"""
        if self.total_loaded == 0:
            return 0.0
        return self.success_count / self.total_loaded


# 全局单例实例
monitor = GlobalMonitor()
