# Stage2-3-4 修改意见评估报告

## 一、评估概述

本报告基于对代码库中 Stage2（数据质量验证）、Stage3（信号预处理）和 Stage4（特征提取）模块的深入分析，评估用户提出的修改意见的**合理性**、**技术可行性**、**实施难度**以及**对数据集的影响**。

---

## 二、修改意见逐条评估

### 2.1 🔴 最高危错误：信号级 Z-score 标准化

| 评估维度 | 评估结果 |
|---------|---------|
| **合理性** | **完全正确** |
| **影响程度** | **致命** |
| **修改难度** | **低** |
| **建议操作** | **立即修复** |

**详细分析：**

当前代码位置：[stage3_preprocessing.py](file:///C:/Users/Asteria/Documents/%E4%BB%A3%E7%A0%81/%E4%BD%9C%E4%B8%9A/%E8%AE%BA%E6%96%87/project/v16-project/missyou/pipeline/stage3_preprocessing.py#L87-L110)

```python
def zscore_normalize(self, signal: np.ndarray) -> tuple:
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0)
    normalized = (signal - mean) / std
    return normalized, (mean, std)
```

**问题本质：**
- Z-score 标准化将信号转换为零均值、单位方差，**彻底破坏了信号的能量信息**
- 时域特征如 RMS（均方根）、方差、能量直接依赖于信号幅值
- 小波 DWT 能量特征是系数平方和，标准化后能量信息丢失
- 频域能量特征同样依赖原始幅值

**数据集影响：**
- PHM2010 数据集的核心是**刀具磨损与信号能量的相关性**
- 磨损加剧会导致切削力增大、振动增强，信号能量随之增加
- 信号级标准化后，不同磨损状态的能量差异被抹平

**结论：用户的批评完全正确，此错误会导致论文结论不严谨。**

---

### 2.2 🟡 小波能量计算错误

| 评估维度 | 评估结果 |
|---------|---------|
| **合理性** | **正确** |
| **影响程度** | **高** |
| **修改难度** | **低** |
| **建议操作** | **立即修复** |

**详细分析：**

当前代码位置：[feature_extractor.py](file:///C:/Users/Asteria/Documents/%E4%BB%A3%E7%A0%81/%E4%BD%9C%E4%B8%9A/%E8%AE%BA%E6%96%87/project/v16-project/missyou/modules/stage4_feature_extract/feature_extractor.py#L344-L350)

```python
total_energy = np.sum(x**2) + 1e-10
for i in range(actual_level - 1, -1, -1):
    cD_energy = np.sum(coeffs[i + 1]**2)
    features.append(cD_energy / total_energy)  # 问题：除以总能量
```

**问题本质：**
- 用户指出"磨损会让能量变大，归一化会丢失信息"完全正确
- 小波能量归一化后变成"能量占比"，无法反映绝对能量变化
- 正确做法是直接使用系数平方和作为特征

**数据集影响：**
- PHM2010 数据集中，严重磨损阶段的信号能量显著高于初期
- 能量占比无法区分"整体能量都很高"和"整体能量都很低"的情况

**结论：用户的批评正确，此修改能显著提升特征区分度。**

---

### 2.3 🟡 去趋势实现建议

| 评估维度 | 评估结果 |
|---------|---------|
| **合理性** | **正确** |
| **影响程度** | **中** |
| **修改难度** | **低** |
| **建议操作** | **建议采纳** |

**详细分析：**

当前代码位置：[stage3_preprocessing.py](file:///C:/Users/Asteria/Documents/%E4%BB%A3%E7%A0%81/%E4%BD%9C%E4%B8%9A/%E8%AE%BA%E6%96%87/project/v16-project/missyou/pipeline/stage3_preprocessing.py#L63-L85)

```python
def remove_trend(self, signal: np.ndarray) -> np.ndarray:
    detrended_signal = np.zeros_like(signal)
    x = np.arange(signal.shape[0])
    for i in range(signal.shape[1]):
        channel = signal[:, i]
        coeffs = np.polyfit(x, channel, 1)
        trend = np.polyval(coeffs, x)
        detrended_signal[:, i] = channel - trend
    return detrended_signal
```

**评估：**
- 当前实现本质上与 `scipy.signal.detrend` 的 `type='linear'` 相同
- 使用 scipy 标准函数的优势：
  - 代码更简洁
  - 经过充分测试
  - 性能优化（底层用 C 实现）
  - 学术论文更易引用标准方法

**结论：建议采纳，提升代码规范性和可维护性。**

---

### 2.4 🟡 FFT 物理频率问题

| 评估维度 | 评估结果 |
|---------|---------|
| **合理性** | **正确** |
| **影响程度** | **低-中** |
| **修改难度** | **低** |
| **建议操作** | **建议采纳** |

**详细分析：**

当前代码位置：[feature_extractor.py](file:///C:/Users/Asteria/Documents/%E4%BB%A3%E7%A0%81/%E4%BD%9C%E4%B8%9A/%E8%AE%BA%E6%96%87/project/v16-project/missyou/modules/stage4_feature_extract/feature_extractor.py#L251-L258)

```python
n = len(x)
fft_vals = fft(x)
fft_magnitude = np.abs(fft_vals[:n//2])
fft_magnitude = fft_magnitude / (n + 1e-10)
freqs = np.arange(len(fft_magnitude)) / n  # 问题：归一化频率，非物理频率
```

**问题本质：**
- 当前频率轴是归一化频率 [0, 0.5]，而非真实物理频率 [0, 10000]Hz
- 论文中需要报告具体频率值时，归一化频率不规范
- `frequency_center` 等特征值失去物理意义

**数据集影响：**
- PHM2010 采样频率为 20000Hz
- 带通滤波范围 [100, 5000]Hz，与频率轴单位不一致
- 学术论文应使用物理频率

**结论：建议采纳，提升论文规范性。**

---

### 2.5 🟢 缺少特征归一化模块

| 评估维度 | 评估结果 |
|---------|---------|
| **合理性** | **正确** |
| **影响程度** | **高** |
| **修改难度** | **低** |
| **建议操作** | **必须添加** |

**详细分析：**

当前 Pipeline 流程：
```
Stage1 → Stage2 → Stage3 → Stage4(特征提取) → [结束]
```

用户建议的正确流程：
```
Stage1 → Stage2 → Stage3 → Stage4 → 特征归一化 → CVOCA → [结束]
```

**必要性：**
- 不同特征量纲差异极大（如能量可能是 1e6，偏度是无量纲）
- CVOCA 等特征选择算法对特征尺度敏感
- 机器学习模型训练前必须归一化

**结论：必须添加此模块，否则后续特征选择和模型训练效果将受严重影响。**

---

### 2.6 🟢 缺少 CVOCA 特征选择入口

| 评估维度 | 评估结果 |
|---------|---------|
| **合理性** | **正确** |
| **影响程度** | **高** |
| **修改难度** | **中** |
| **建议操作** | **必须添加** |

**详细分析：**

当前代码分析：
- `run_pipeline.py` 只执行到 Stage3（预处理）
- Stage4 的 `FeatureExtractor` 是独立模块，未集成到 Pipeline
- 没有 CVOCA 调用入口

**必要性：**
- 168 维特征存在冗余，需要特征选择降维
- CVOCA 是论文核心方法之一
- 完整流程必须包含特征选择环节

**结论：必须添加，否则无法完成端到端流程。**

---

## 三、修改方案与实施难度评估

### 3.1 修改任务清单

| 序号 | 修改内容 | 文件 | 难度 | 风险 |
|-----|---------|------|-----|-----|
| 1 | 删除 `zscore_normalize` 函数 | stage3_preprocessing.py | 低 | 低 |
| 2 | 修改 `preprocess` 方法，移除标准化步骤 | stage3_preprocessing.py | 低 | 低 |
| 3 | 替换 `remove_trend` 为 scipy.detrend | stage3_preprocessing.py | 低 | 低 |
| 4 | 修改小波能量计算，移除归一化 | feature_extractor.py | 低 | 低 |
| 5 | 修改 FFT 频率轴为真实物理频率 | feature_extractor.py | 低 | 低 |
| 6 | 新建 `FeatureNormalizer` 模块 | modules/ | 低 | 低 |
| 7 | 新建 `FeatureSelector` 模块（CVOCA入口） | modules/ | 中 | 中 |
| 8 | 集成新模块到 Pipeline | run_pipeline.py | 中 | 中 |

### 3.2 实施难度矩阵

```
难度等级定义：
- 低（1-2小时）：简单代码修改，无需重构
- 中（2-4小时）：需要新建模块或调整流程
- 高（4小时+）：需要架构调整或复杂集成

风险等级定义：
- 低：局部修改，不影响其他模块
- 中：可能影响下游流程，需要测试验证
- 高：架构级变更，风险较大
```

---

## 四、对数据集的影响分析

### 4.1 特征质量提升预期

| 修改项 | 修改前问题 | 修改后效果 | 对 PHM2010 的影响 |
|-------|-----------|-----------|------------------|
| 信号级标准化移除 | 能量特征失效 | 能量特征有效 | **显著提升**：磨损与能量相关性恢复 |
| 小波能量不归一化 | 丢失绝对能量信息 | 保留能量绝对值 | **显著提升**：区分不同磨损阶段 |
| FFT 物理频率 | 频率无物理意义 | 频率值可解释 | **中等提升**：论文更规范 |
| 特征级归一化 | 特征量纲混乱 | 特征尺度统一 | **显著提升**：CVOCA 效果提升 |

### 4.2 关键影响评估

**1. 时域特征恢复**
- RMS、方差、能量等特征将正确反映信号强度
- 磨损严重时，切削力增大，这些特征值会显著上升

**2. 频域特征恢复**
- 频谱峰值等特征将与实际频率对应
- 可以观察到磨损相关的频率成分变化

**3. 小波特征恢复**
- 各层能量绝对值保留
- 可以观察到磨损导致的能量分布变化

---

## 五、推荐的实施步骤

### 5.1 第一阶段：修复 Stage3（信号预处理）

```python
# 修改后 stage3_preprocessing.py 的 preprocess 方法
def preprocess(self, cutting_data: dict) -> dict:
    # ... 初始化代码 ...
    
    try:
        detrended = self.remove_trend(signal)  # 步骤1: 去趋势
        self.logger.info(f"步骤1/2 - 趋势去除完成", cut_unique_id)
    except Exception as e:
        self.logger.error(f"趋势去除失败", cut_unique_id, exc=e)
        return None
    
    try:
        filtered = self.bandpass_filter(detrended)  # 步骤2: 带通滤波
        self.logger.info(f"步骤2/2 - 带通滤波完成", cut_unique_id)
    except Exception as e:
        self.logger.error(f"带通滤波失败", cut_unique_id, exc=e)
        return None
    
    # 注意：不再执行 Z-score 标准化！
    
    preprocessed_data = {
        'processed_signal': filtered,  # 使用滤波后信号，而非标准化信号
        # ... 其他字段 ...
    }
```

### 5.2 第二阶段：修复 Stage4（特征提取）

**修改小波能量计算：**
```python
# 修改前
features.append(cD_energy / total_energy)

# 修改后  
features.append(cD_energy)
```

**修改FFT频率轴：**
```python
# 修改前
freqs = np.arange(len(fft_magnitude)) / n

# 修改后
fs = 20000  # PHM2010采样频率
freqs = np.fft.fftfreq(n, 1/fs)[:n//2]
```

### 5.3 第三阶段：添加新模块

**新建 modules/stage5_normalization/feature_normalizer.py**
```python
"""
Stage5: 特征归一化模块
对168维特征做归一化，放在特征提取之后、CVOCA之前
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class FeatureNormalizer:
    def __init__(self, method: str = 'standard'):
        """
        Args:
            method: 'standard' (Z-score) 或 'minmax'
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown method: {method}")
        
    def fit(self, feature_matrix):
        """训练归一化器"""
        self.scaler.fit(feature_matrix)
        
    def transform(self, feature_matrix):
        """应用归一化"""
        return self.scaler.transform(feature_matrix)
        
    def fit_transform(self, feature_matrix):
        """训练并应用"""
        return self.scaler.fit_transform(feature_matrix)
```

**新建 modules/stage6_selection/feature_selector.py**
```python
"""
Stage6: 特征选择模块（CVOCA）
从168维特征中选择最优子集
"""
import numpy as np

class FeatureSelector:
    def __init__(self, selector=None):
        """
        Args:
            selector: CVOCA或其他特征选择器实例
        """
        self.selector = selector
        self.selected_indices = None
        
    def fit(self, X, y):
        """训练特征选择器"""
        if self.selector is not None:
            self.selector.fit(X, y)
            # 假设selector有get_selected_indices方法
            if hasattr(self.selector, 'get_selected_indices'):
                self.selected_indices = self.selector.get_selected_indices()
        
    def transform(self, X):
        """应用特征选择"""
        if self.selected_indices is not None:
            return X[:, self.selected_indices]
        return X
        
    def fit_transform(self, X, y):
        """训练并应用"""
        self.fit(X, y)
        return self.transform(X)
```

### 5.4 第四阶段：集成到 Pipeline

```python
# 修改 run_pipeline.py，添加 Stage4-6
from modules.stage4_feature_extract.feature_extractor import BatchFeatureExtractor
from modules.stage5_normalization.feature_normalizer import FeatureNormalizer
from modules.stage6_selection.feature_selector import FeatureSelector

class DataPipeline:
    def __init__(self, ...):
        # ... 现有初始化 ...
        self.stage4 = BatchFeatureExtractor(run_id, self.logger)
        self.stage5 = FeatureNormalizer()
        self.stage6 = FeatureSelector()
    
    def _process_single_cutting(self, cut_data: dict) -> bool:
        # Stage2: 验证
        valid_data, error_info = self.stage2.validate(cut_data)
        if error_info:
            # ... 处理错误 ...
            
        # Stage3: 预处理（修改后只做去趋势+滤波）
        preprocessed_data = self.stage3.preprocess(valid_data)
        if preprocessed_data is None:
            # ... 处理错误 ...
            
        # Stage4: 特征提取（新增）
        features = self.stage4.extract(preprocessed_data['processed_signal'])
        
        # Stage5: 特征归一化（新增）
        normalized_features = self.stage5.transform(features)
        
        # Stage6: CVOCA特征选择（新增）
        # 需要累积所有样本后进行，或使用增量方式
        # ...
        
        # 保存结果
        # ...
```

---

## 六、结论与建议

### 6.1 核心结论

| 修改项 | 是否必须 | 优先级 |
|-------|---------|-------|
| 移除信号级 Z-score | **是** | P0 |
| 小波能量不归一化 | **是** | P0 |
| 替换去趋势为 scipy | 否 | P2 |
| FFT 使用物理频率 | 否 | P2 |
| 添加特征归一化 | **是** | P0 |
| 添加 CVOCA 入口 | **是** | P0 |

### 6.2 实施建议

1. **立即执行 P0 级修改**（1-2天）：
   - 修复 Stage3 的信号级标准化问题
   - 修复 Stage4 的小波能量计算问题
   - 添加特征归一化模块
   - 添加 CVOCA 特征选择入口

2. **后续优化 P2 级修改**（可选）：
   - 替换去趋势实现
   - 修正 FFT 频率轴

3. **测试验证**：
   - 使用 PHM2010 样本数据测试修改后的流程
   - 验证能量特征是否正确反映磨损状态
   - 验证特征选择效果

### 6.3 预期收益

- **特征质量提升**：能量类特征恢复有效性，与磨损状态相关性增强
- **论文严谨性提升**：方法符合学术规范，实验可复现
- **模型性能提升**：CVOCA 在归一化特征上效果更好，最终模型精度可能提升

---

## 七、代码修改清单

### 需要删除的代码

| 文件 | 位置 | 内容 |
|-----|-----|-----|
| stage3_preprocessing.py | 第87-110行 | `zscore_normalize` 函数 |
| stage3_preprocessing.py | 第159-166行 | `preprocess` 中的标准化调用 |

### 需要修改的代码

| 文件 | 位置 | 修改内容 |
|-----|-----|---------|
| stage3_preprocessing.py | 第63-85行 | 替换为 `scipy.signal.detrend` |
| feature_extractor.py | 第350行 | `cD_energy / total_energy` → `cD_energy` |
| feature_extractor.py | 第257行 | 频率轴使用真实物理频率 |

### 需要新建的文件

| 文件路径 | 功能 |
|---------|-----|
| modules/stage5_normalization/\_\_init\_\_.py | 模块初始化 |
| modules/stage5_normalization/feature_normalizer.py | 特征归一化类 |
| modules/stage6_selection/\_\_init\_\_.py | 模块初始化 |
| modules/stage6_selection/feature_selector.py | CVOCA特征选择类 |

### 需要修改的流程

| 文件 | 修改内容 |
|-----|---------|
| run_pipeline.py | 集成 Stage4-6，扩展 Pipeline |

---

**文档版本**: v1.0  
**生成日期**: 2026-05-26  
**分析对象**: PHM2010 数据处理 Pipeline (Stage2-3-4)