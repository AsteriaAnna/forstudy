# Stage1 数据处理方案修改影响评估报告

## 1. 概述

本报告针对以下修改需求进行全面的影响评估：

| 修改项 | 当前方案 | 目标方案 |
|--------|---------|---------|
| 磨损值处理 | 保留原始三刃值 | 添加 `robust_wear` 字段（自适应加权） |
| 异常处理策略 | 拦截过滤并记录，单条异常不影响整体 | 平均磨损值作为标签，消除个体异常 |
| 幅值范围检查 | 力(-1000,1000)N、振动(-100,100)mm/s、声发射(-100,100)V | 三刃磨损值无明确幅值范围检查 |

---

## 2. 代码结构分析

### 2.1 整体架构

```
Stage1 (加载) → Stage2 (验证) → Stage3 (预处理) → Stage4 (特征) → Stage5 (增强) → Stage6 (TFRecord)
```

### 2.2 关键文件清单

| 文件 | 路径 | 职责 |
|------|------|------|
| `stage1_loading.py` | `pipeline/stage1_loading.py` | 原始数据加载，磨损标签解析 |
| `stage2_validation.py` | `pipeline/stage2_validation.py` | 9项质量验证 |
| `stage3_preprocessing.py` | `pipeline/stage3_preprocessing.py` | 信号预处理 |
| `run_pipeline.py` | `pipeline/run_pipeline.py` | Stage1-3流式处理 |
| `main.py` | `main.py` | 完整流水线入口 |
| `feature_extractor.py` | `modules/stage4_feature_extract/feature_extractor.py` | 特征提取 |
| `augmenter.py` | `modules/stage5_augmentation/augmenter.py` | 数据增强 |
| `tfrecord_builder.py` | `modules/stage6_tfrecord_build/tfrecord_builder.py` | TFRecord构建 |

---

## 3. 数据契约分析

### 3.1 当前数据契约

**wear_label 结构（Stage1输出）：**
```python
wear_label = {
    'flute_1': float,    # 刀刃1磨损值
    'flute_2': float,    # 刀刃2磨损值
    'flute_3': float     # 刀刃3磨损值
}
```

**数据流转：**

| 阶段 | wear_label格式 | 修改情况 |
|------|--------------|---------|
| Stage1 | `dict` | 读取原始数据 |
| Stage2 | `dict` | 验证三刃值范围 |
| Stage3 | `dict` | 原样传递 |
| Stage4 | `dict` | 原样传递 |
| Stage5 | `np.ndarray(N, 3)` | 转换为数组 |
| Stage6 | `np.ndarray(N, 3)` | TFRecord存储 |

### 3.2 修改后数据契约

**wear_label 结构（修改后）：**
```python
wear_label = {
    'flute_1': float,       # 刀刃1磨损值（保留）
    'flute_2': float,       # 刀刃2磨损值（保留）
    'flute_3': float,       # 刀刃3磨损值（保留）
    'robust_wear': float    # 新增：自适应加权稳健磨损值
}
```

---

## 4. 修改影响评估

### 4.1 Stage1 修改分析

**文件：** `pipeline/stage1_loading.py`

**修改内容：**
1. 在 `load_wear_labels()` 方法中添加 `robust_wear` 计算
2. 添加辅助方法 `_calculate_robust_wear()`

**影响评估：**
| 维度 | 影响 | 风险 |
|------|------|------|
| 数据结构 | 新增字段，向后兼容 | 低 |
| 内存占用 | 增加约4字节/切削 | 可忽略 |
| 计算开销 | 每次加载增加约10次浮点运算 | 可忽略 |
| ID规范 | 无影响 | 无 |

**代码修改点：**
- 第77-96行：`load_wear_labels()` 方法
- 需要添加 `_calculate_robust_wear()` 辅助方法

---

### 4.2 Stage2 修改分析

**文件：** `pipeline/stage2_validation.py`

**修改内容：**
1. 移除 `check_wear_label()` 中的幅值范围检查
2. 改为检查 `robust_wear` 字段是否存在

**影响评估：**
| 维度 | 影响 | 风险 |
|------|------|------|
| 验证逻辑 | 简化磨损标签验证 | 低 |
| 数据质量 | 依赖Stage1的稳健值计算 | 中 |
| 向后兼容 | 需检查字段存在性 | 低 |

**代码修改点：**
- 第174-190行：`check_wear_label()` 方法

---

### 4.3 Stage3 修改分析

**文件：** `pipeline/stage3_preprocessing.py`

**影响评估：**
| 维度 | 影响 | 风险 |
|------|------|------|
| 预处理逻辑 | 无变化，原样传递 | 无 |
| 数据结构 | 透明处理新增字段 | 无 |

**结论：无需修改**

---

### 4.4 Stage4 修改分析

**文件：** `modules/stage4_feature_extract/feature_extractor.py`

**影响评估：**
| 维度 | 影响 | 风险 |
|------|------|------|
| 特征提取 | 无变化，wear_label仅传递 | 无 |
| 数据结构 | 透明处理新增字段 | 无 |

**结论：无需修改**

---

### 4.5 Stage5 修改分析

**文件：** `modules/stage5_augmentation/augmenter.py`

**修改内容：**
1. `bin_wear_category()` 方法需要适配新的 `wear_label` 结构
2. 可以选择使用 `robust_wear` 或三刃平均值

**影响评估：**
| 维度 | 影响 | 风险 |
|------|------|------|
| 类别划分 | 使用稳健值更准确 | 低 |
| 向后兼容 | 需处理两种格式 | 低 |

**代码修改点：**
- 第140-161行：`bin_wear_category()` 方法

---

### 4.6 Stage6 修改分析

**文件：** `modules/stage6_tfrecord_build/tfrecord_builder.py`

**修改内容：**
1. 可选：扩展 `WEAR_LABEL_DIM` 为4（包含 `robust_wear`）
2. 或保持3维，使用 `robust_wear` 替代三刃值

**影响评估：**
| 维度 | 影响 | 风险 |
|------|------|------|
| 数据集格式 | 可能需要调整维度 | 中 |
| 模型输入 | 需同步修改标签输入 | 中 |

**代码修改点：**
- 第26行：`WEAR_LABEL_DIM` 常量
- 第97-106行：`_create_feature_spec()` 方法
- 第108-140行：`_serialize_example()` 方法

---

### 4.7 Pipeline 集成层分析

**文件：** `pipeline/run_pipeline.py`、`main.py`

**影响评估：**
| 维度 | 影响 | 风险 |
|------|------|------|
| 数据保存 | JSON序列化自动处理新增字段 | 无 |
| 缓存机制 | 透明处理 | 无 |
| 统计信息 | 无变化 | 无 |

**结论：无需修改**

---

## 5. 修改优先级与风险矩阵

### 5.1 修改优先级

| 优先级 | 文件 | 修改内容 | 必要性 |
|--------|------|---------|--------|
| P0 | `stage1_loading.py` | 添加 `robust_wear` 计算 | 必需 |
| P1 | `stage2_validation.py` | 调整磨损标签验证 | 必需 |
| P2 | `stage5_augmentation/augmenter.py` | 使用 `robust_wear` 进行类别划分 | 推荐 |
| P3 | `stage6_tfrecord_build/tfrecord_builder.py` | 调整标签维度 | 可选 |

### 5.2 风险评估矩阵

| 风险类型 | 描述 | 严重程度 | 缓解措施 |
|----------|------|---------|---------|
| 数据泄露 | 稳健值计算使用测试集数据 | 高 | 在模型训练阶段计算 |
| 向后兼容 | 现有代码依赖三刃值格式 | 中 | 保留原始字段 |
| 验证缺失 | 移除幅值检查可能引入异常值 | 中 | 依赖Stage1的稳健值计算 |
| 模型适配 | Stage6修改影响下游模型 | 中 | 提供两种格式选项 |

---

## 6. 推荐实现方案

### 6.1 方案A：最小侵入式（推荐）

**核心原则：** 保持向后兼容性，新增字段可选使用

**修改内容：**

1. **Stage1**：添加 `robust_wear` 字段，保留三刃值
2. **Stage2**：仅检查字段存在性，不检查幅值范围
3. **Stage5**：优先使用 `robust_wear` 进行类别划分
4. **Stage6**：保持3维标签，使用三刃平均值（向后兼容）

**优点：**
- 最小代码改动
- 完全向后兼容
- 风险可控

### 6.2 方案B：完全迁移

**修改内容：**

1. **Stage1**：添加 `robust_wear` 字段
2. **Stage2**：简化验证逻辑
3. **Stage5**：使用 `robust_wear`
4. **Stage6**：将标签维度改为1维（仅 `robust_wear`）

**优点：**
- 数据结构更简洁
- 专注于稳健值

**缺点：**
- 破坏向后兼容
- 需要修改所有下游模块

---

## 7. 实施路线图

### 阶段1：Stage1 修改（1天）

```python
# 修改 stage1_loading.py
def load_wear_labels(self, tool_id: str) -> dict:
    # ... 现有代码 ...
    wear_data[cut_num] = {
        'flute_1': float(parts[1]),
        'flute_2': float(parts[2]),
        'flute_3': float(parts[3]),
        'robust_wear': self._calculate_robust_wear(parts[1:4])  # 新增
    }

def _calculate_robust_wear(self, flute_values: list) -> float:
    """自适应加权稳健磨损值计算"""
    values = np.array([float(v) for v in flute_values])
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    if std_val == 0:
        return mean_val
    
    # 置信度计算：离均值越近，权重越高
    confidence = 1 - np.abs(values - mean_val) / (3 * std_val)
    confidence = np.clip(confidence, 0.1, 1.0)
    weights = confidence / np.sum(confidence)
    
    return float(np.sum(values * weights))
```

### 阶段2：Stage2 修改（0.5天）

```python
# 修改 stage2_validation.py
def check_wear_label(self, wear_label: Dict[str, float]) -> bool:
    """检查磨损标签（简化版）"""
    # 检查必要字段存在
    required_fields = ['flute_1', 'flute_2', 'flute_3', 'robust_wear']
    for key in required_fields:
        if key not in wear_label:
            return True  # 缺失字段视为不合格
        value = wear_label[key]
        if np.isnan(value):
            return True
    return False
```

### 阶段3：Stage5 修改（0.5天）

```python
# 修改 augmenter.py
def bin_wear_category(self, wear_label) -> str:
    """使用稳健磨损值进行类别划分"""
    # 优先使用 robust_wear
    if isinstance(wear_label, dict) and 'robust_wear' in wear_label:
        mean_wear = wear_label['robust_wear']
    else:
        # 兼容旧格式
        if isinstance(wear_label, dict):
            wear_values = [wear_label['flute_1'], wear_label['flute_2'], wear_label['flute_3']]
            mean_wear = np.mean(wear_values)
        else:
            mean_wear = np.mean(wear_label)
    
    if mean_wear < self.WEAR_LOW_THRESHOLD:
        return 'low'
    elif mean_wear < self.WEAR_HIGH_THRESHOLD:
        return 'medium'
    else:
        return 'high'
```

### 阶段4：测试与验证（1天）

1. 运行完整流水线验证数据流转
2. 检查 `robust_wear` 字段是否正确计算
3. 验证Stage2验证逻辑正常工作
4. 确认数据增强使用正确的磨损值

---

## 8. 总结

### 8.1 修改影响摘要

| 文件 | 修改类型 | 风险等级 | 工作量 |
|------|---------|---------|--------|
| `stage1_loading.py` | 新增功能 | 低 | 1天 |
| `stage2_validation.py` | 逻辑调整 | 低 | 0.5天 |
| `stage5_augmentation/augmenter.py` | 适配修改 | 低 | 0.5天 |
| `stage6_tfrecord_build/tfrecord_builder.py` | 可选修改 | 中 | 0.5天 |
| **总计** | - | - | **2.5天** |

### 8.2 关键结论

1. **修改可行**：所有修改都在现有架构范围内，不会破坏核心流程
2. **风险可控**：通过保留原始字段实现向后兼容
3. **收益显著**：使用稳健磨损值提升数据质量，减少异常值影响
4. **实施建议**：采用方案A（最小侵入式），逐步迁移

### 8.3 后续建议

1. 在Stage1添加配置选项，支持选择不同的稳健值计算方法
2. 在Stage5添加配置选项，支持选择使用 `robust_wear` 或原始三刃值
3. 添加单元测试验证稳健值计算的正确性

---

**报告生成日期**：2026-05-25  
**版本**：v1.0