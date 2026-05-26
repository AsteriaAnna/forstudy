"""
Stage4: 多域特征提取模块

功能:
- 时域特征提取 (15特征/通道)
- 频域特征提取 (5特征/通道)
- 时频域特征提取 (4特征/通道)
- 批量处理支持
- 特征验证

输出: 168维特征向量 (7通道 × 24特征)
"""

from .feature_extractor import (
    FeatureExtractor,
    BatchFeatureExtractor,
    extract_features_from_signal,
)

# 模块版本
__version__ = "1.0.0"

# 导出公共接口
__all__ = [
    'FeatureExtractor',
    'BatchFeatureExtractor',
    'extract_features_from_signal',
]