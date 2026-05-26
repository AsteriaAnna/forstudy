
# PHM2010 刀具磨损预测数据处理流水线

## 项目概述

PHM2010 刀具磨损预测项目是一个端到端的数据处理流水线，用于处理 PHM2010 数据集的传感器信号数据，提取多域特征，进行数据增强，最终生成可用于深度学习模型训练的 TFRecord 数据集。

---

## 目录结构

```
missyou/
├── main.py                          # 项目全局入口
├── analyze_results.py               # 结果分析与可视化脚本
├── plot_raw_signal.py               # 原始信号绘图脚本
├── .gitignore
├── core/                            # 全局公共工具模块
│   ├── __init__.py
│   ├── id_generator.py              # ID生成与解析工具
│   ├── logger.py                    # 日志记录器
│   ├── monitor.py                   # 全局监控器
│   └── utils.py                     # 通用工具函数
├── pipeline/                        # Stage1-3 串行流水线（不可拆分）
│   ├── __init__.py
│   ├── run_pipeline.py              # Pipeline集成运行器
│   ├── stage1_loading.py            # 原始数据加载
│   ├── stage2_validation.py         # 数据质量验证
│   └── stage3_preprocessing.py      # 信号预处理
├── modules/                         # Stage4-6 独立可运行模块
│   ├── __init__.py
│   ├── stage4_feature_extract/
│   │   ├── __init__.py
│   │   └── feature_extractor.py     # 多域特征提取
│   ├── stage5_augmentation/
│   │   ├── __init__.py
│   │   └── augmenter.py             # 数据增强
│   └── stage6_tfrecord_build/
│       ├── __init__.py
│       └── tfrecord_builder.py      # TFRecord构建
├── study/                           # 研究参考资料
│   ├── reference/
│   │   ├── 2026-CIRP-Jiang-Transformer-Physics.pdf
│   │   └── 2026-CIRP-Jiang-Transformer-Physics.txt
│   ├── 论文数据处理方法拆解报告.md
│   └── 设计方案与论文方法对比分析.md
├── docs/                            # 文档目录
│   ├── readme.md                    # 本文件
│   ├── bulid.md                     # 开发规范文档
│   └── stage1修改/                  # Stage1修改文档
├── data/                            # 数据目录
│   ├── raw/                         # 原始数据（只读）
│   │   └── phm2010_raw/
│   │       ├── c1/                  # 刀具c1数据
│   │       │   ├── c1/              # 信号文件目录
│   │       │   │   ├── c_1_001.csv  # 格式1: c_{tool}_{cut}.csv
│   │       │   │   └── c_c1_1.csv   # 格式2: c_c{tool}_{cut}.csv
│   │       │   └── c1_wear.csv      # 磨损标签文件
│   │       ├── c4/                  # 刀具c4数据
│   │       └── c6/                  # 刀具c6数据
│   ├── interim/                     # 中间数据
│   │   ├── features/                # Stage4特征输出
│   │   └── samples/                 # Stage5样本输出
│   └── processed/                   # 最终数据
├── results/                         # 运行结果目录
│   └── run_xxx/                     # 单次运行版本目录
│       ├── data/                    # 运行时数据
│       ├── logs/                    # 日志文件
│       └── figures/                 # 可视化图表
```

---

## 程序运行逻辑

### 整体架构

本项目采用六阶段流水线架构，分为两个主要部分：

1. **Stage1-3 (pipeline/)**：串行执行，不可拆分
   - 逐切削流式原始数据加载
   - 多维度数据质量验证
   - 信号清洗与标准化预处理

2. **Stage4-6 (modules/)**：独立可运行模块
   - 多域特征提取
   - 数据增强与样本重构
   - TFRecord数据集构建

### 程序入口

通过 `main.py` 入口文件运行程序：

```bash
# 运行完整流水线 (Stage1-6)
python main.py --stage all

# 仅运行Stage1-3
python main.py --stage 1-3

# 仅运行Stage4
python main.py --stage 4

# 指定运行ID和恢复
python main.py --stage all --run-id run_001_20250524_1530
```

### 辅助脚本

项目还包含两个辅助分析脚本：

- `analyze_results.py`：分析处理结果，生成特征分布、t-SNE可视化等图表
- `plot_raw_signal.py`：绘制原始信号波形图，用于数据探索

---

## 模块关系图

```mermaid
graph TB
    subgraph 数据源
        RAW[原始CSV数据&lt;br/&gt;data/raw/phm2010_raw/]
        WEAR[磨损标签&lt;br/&gt;cX_wear.csv]
    end

    subgraph Stage1_3["Stage1-3 (pipeline/) - 串行执行"]
        S1[Stage1Loader&lt;br/&gt;原始数据加载]
        S2[Stage2Validator&lt;br/&gt;数据质量验证]
        S3[Stage3Preprocessor&lt;br/&gt;信号预处理]
    end

    subgraph Stage4["Stage4 (modules/stage4_feature_extract/)"]
        FE[FeatureExtractor&lt;br/&gt;多域特征提取]
        BFE[BatchFeatureExtractor&lt;br/&gt;批量特征提取]
    end

    subgraph Stage5["Stage5 (modules/stage5_augmentation/)"]
        DA[DataAugmenter&lt;br/&gt;数据增强]
    end

    subgraph Stage6["Stage6 (modules/stage6_tfrecord_build/)"]
        TF[TFRecordBuilder&lt;br/&gt;数据集构建]
    end

    subgraph 输出
        TFRECORD[TFRecord文件&lt;br/&gt;train.tfrecord&lt;br/&gt;test.tfrecord]
        META[元数据&lt;br/&gt;dataset_meta.json]
    end

    RAW --&gt; S1
    WEAR --&gt; S1
    S1 --&gt; S2
    S2 --&gt; S3
    S3 --&gt; FE
    FE --&gt; BFE
    BFE --&gt; DA
    DA --&gt; TF
    TF --&gt; TFRECORD
    TF --&gt; META
```

### 数据流向

```mermaid
flowchart LR
    subgraph 输入
        CSV[CSV信号文件]
        CSV_WEAR[磨损标签CSV]
    end

    subgraph Stage1
        S1_OUT[signal, tool_id&lt;br/&gt;cut_unique_id&lt;br/&gt;wear_label]
    end

    subgraph Stage2
        S2_OUT[valid_data&lt;br/&gt;或None]
    end

    subgraph Stage3
        S3_OUT[processed_signal&lt;br/&gt;filter_params]
    end

    subgraph Stage4
        S4_OUT[feature_vector&lt;br/&gt;168维]
    end

    subgraph Stage5
        S5_OUT[augmented_features&lt;br/&gt;augmented_labels]
    end

    subgraph Stage6
        S6_OUT[train.tfrecord&lt;br/&gt;test.tfrecord]
    end

    CSV &amp; CSV_WEAR --&gt; S1_OUT --&gt; S2_OUT --&gt; S3_OUT --&gt; S4_OUT --&gt; S5_OUT --&gt; S6_OUT
```

---

## 核心模块详解

### 1. core/id_generator.py

**功能**：解析和生成切削唯一标识ID

**关键函数**：

| 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `parse_filename(filename)` | 文件名字符串 | `(tool_id, cut_num)` | 解析CSV文件名为刀具ID和切削编号 |
| `generate_cut_unique_id(tool_id, cut_num)` | 刀具ID, 切削编号 | 唯一ID字符串 | 生成标准格式的唯一标识 |

**支持的文件名格式**：
- `c_1_001.csv` → tool_id=`c1`, cut_num=`1`
- `c_c1_1.csv` → tool_id=`c1`, cut_num=`1`

**数据契约**：
```python
# 输出格式
{
    "tool_id": str,      # "c1", "c4", "c6"
    "cut_num": int,      # 切削编号
    "cut_unique_id": str # "c1_cut1", "c4_cut215" 等
}
```

---

### 2. core/logger.py

**功能**：统一日志记录，支持控制台和文件双输出，单例模式

**日志级别**：
- `INFO`：正常流程日志
- `WARN`：轻微异常警告
- `ERROR`：严重错误日志
- `STAT`：统计摘要日志

**关键函数**：

| 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(run_id, log_dir, console_output, file_output)` | 运行ID等 | Logger实例 | 初始化日志记录器 |
| `info(message, cut_id)` | 消息, 可选cut_id | None | 输出普通日志 |
| `warn(message, cut_id)` | 消息, 可选cut_id | None | 输出警告日志 |
| `error(message, cut_id, exc)` | 消息, cut_id, 异常 | None | 输出错误日志 |
| `stat(message)` | 消息 | None | 输出统计日志 |
| `generate_run_id(run_seq)` | 序号 | run ID字符串 | 生成运行标识符 |

**输出格式**：
```
2025-05-24 15:30:00 [INFO] c1_cut12 - Signal loaded successfully
2025-05-24 15:30:00 [WARN] c1_cut15 - Missing metadata
2025-05-24 15:30:00 [ERROR] c1_cut15 - File not found
2025-05-24 15:30:00 [STAT] Total loaded: 315, Success: 300, Failed: 15
```

---

### 3. core/monitor.py

**功能**：全局监控器单例，跟踪数据处理统计信息

**状态常量**：
- `LOADED`：已加载
- `VALID`：验证通过
- `FILTERED`：异常过滤
- `PROCESSED`：处理完成
- `SKIPPED`：跳过

**关键函数**：

| 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `increment_loaded()` | None | None | 增加总加载计数 |
| `increment_success()` | None | None | 增加成功计数 |
| `increment_filtered(reason)` | 过滤原因 | None | 增加过滤计数并记录原因 |
| `get_summary()` | None | Dict | 获取所有统计摘要 |

**监测的异常类型**：
- 空值检测 (empty_values)
- 全零通道检测 (all_zeros)
- 信号断点检测 (breakpoints)
- 采样长度检测 (length_error)
- 幅值范围检测 (amplitude_error)
- 通道相关性检测 (correlation_error)
- 磨损标签检测 (label_error)
- ID绑定检测 (binding_error)
- 波形形状检测 (shape_error)

---

### 4. core/utils.py

**功能**：通用工具函数，路径管理

**关键函数**：

| 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `ensure_dir(path)` | 路径 | None | 创建目录（如果不存在） |
| `get_run_id()` | None | str | 生成运行ID（自动递增） |
| `get_data_dir(run_id)` | run_id | str | 获取运行数据目录路径 |
| `get_results_dir(run_id)` | run_id | str | 获取运行结果目录路径 |
| `get_cache_path(run_id, stage)` | run_id, stage | str | 获取缓存目录路径 |
| `get_interim_dir(subdir)` | 子目录名 | str | 获取中间数据目录 |
| `get_raw_data_dir()` | None | str | 获取原始数据目录 |

**路径映射规则**：
```
Stage1-3: results/{run_id}/data/cache/stageX/
Stage4:   data/interim/features/{run_id}/
Stage5:   data/interim/samples/{run_id}/
Stage6:   data/processed/
```

---

### 5. pipeline/stage1_loading.py

**功能**：以生成器方式流式加载PHM2010原始数据

**关键类**：`Stage1Loader`

**关键函数**：

| 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `load_wear_labels(tool_id)` | 刀具ID | Dict | 加载磨损标签文件 |
| `scan_signal_files(tool_id)` | 刀具ID | List[Tuple] | 扫描信号文件 |
| `load_single_cutting(filepath, tool_id, cut_num)` | 文件路径等 | Dict或None | 加载单个切削数据 |
| `generate_cut_data(tool_ids)` | 刀具ID列表 | Generator | 流式生成切削数据 |

**输入数据契约**：

| 字段 | 类型 | 说明 |
|------|------|------|
| 信号文件目录 | `data/raw/phm2010_raw/{c1,c4,c6}/{c1,c4,c6}/` | CSV格式，无表头 |
| 磨损标签文件 | `data/raw/phm2010_raw/{c1,c4,c6}/{c1,c4,c6}_wear.csv` | CSV格式，含cut,flute_1,flute_2,flute_3 |

**磨损标签文件格式**：
```csv
cut,flute_1,flute_2,flute_3
1,0.05,0.03,0.04
2,0.15,0.12,0.13
...
```

**输出数据契约**：

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `signal` | np.ndarray | (N, 7) | 7通道时序信号 |
| `tool_id` | str | - | 刀具标识 (c1/c4/c6) |
| `cut_num` | int | - | 切削编号 |
| `cut_unique_id` | str | - | 唯一标识符 |
| `wear_label` | dict | - | 磨损标签（含robust_wear和wear_stage） |
| `status` | str | - | "LOADED" |

**磨损标签增强**：
- `robust_wear`：三刃磨损值的加权平均
- `wear_stage`：通过EM算法或固定阈值划分的磨损阶段（initial/normal/severe）

---

### 6. pipeline/stage2_validation.py

**功能**：执行9项数据质量验证

**关键类**：`Stage2Validator`

**信号通道规格**：
- 力信号通道 (0-2)：force_x, force_y, force_z
- 振动信号通道 (3-5)：vib_x, vib_y, vib_z
- 声发射通道 (6)：ae

**幅值范围**：
- 力信号：(-1000, 1000) N
- 振动信号：(-100, 100) mm/s
- 声发射：(-100, 100) V

**采样长度范围**：5000 - 300000 点

**关键函数**：

| 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `check_empty_values(signal)` | 信号数组 | bool | 检测NaN空值 |
| `check_all_zeros(signal)` | 信号数组 | bool | 检测全零通道 |
| `check_signal_breakpoints(signal)` | 信号数组 | bool | 检测信号突变 |
| `check_sample_length(signal)` | 信号数组 | bool | 检测采样长度 |
| `check_amplitude(signal)` | 信号数组 | dict | 检测通道幅值 |
| `check_channel_correlation(signal)` | 信号数组 | bool | 检测通道相关性（当前禁用） |
| `check_wear_label(wear_label)` | 标签字典 | bool | 检测磨损标签 |
| `check_id_binding(cutting_data)` | 数据字典 | bool | 检测ID绑定 |
| `check_waveform_shape(signal)` | 信号数组 | bool | 检测波形形状 |
| `validate(cutting_data)` | 数据字典 | Tuple | 完整验证流程 |

**输入数据契约**（继承Stage1）：
```python
{
    "signal": np.ndarray,      # (N, 7)
    "tool_id": str,
    "cut_unique_id": str,
    "wear_label": dict
}
```

**输出数据契约**：
```python
# 验证通过
(valid_data, None)

# 验证失败
(None, error_info)
# error_info = {
#     "empty_values": "cut_id=...",
#     "all_zeros": "cut_id=...",
#     ...
# }
```

---

### 7. pipeline/stage3_preprocessing.py

**功能**：信号预处理：带通滤波、趋势去除、Z-score标准化

**关键类**：`Stage3Preprocessor`

**滤波参数**：
- 滤波器类型：Butterworth 4阶带通
- 通带频率：[100, 5000] Hz
- 采样频率：20000 Hz

**关键函数**：

| 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `bandpass_filter(signal, fs)` | 信号, 采样率 | 滤波后信号 | 带通滤波 |
| `remove_trend(signal)` | 信号 | 去趋势信号 | 线性趋势去除 |
| `zscore_normalize(signal)` | 信号 | (标准化信号, 统计量) | Z-score标准化 |
| `preprocess(cutting_data)` | 切削数据 | 预处理后数据或None | 完整预处理流程 |

**预处理流程**：
```
原始信号 → 带通滤波 → 趋势去除 → Z-score标准化 → 预处理后信号
```

**输入数据契约**（继承Stage2验证通过数据）：
```python
{
    "signal": np.ndarray,      # (N, 7) 原始信号
    "cut_unique_id": str,
    "tool_id": str,
    "wear_label": dict
}
```

**输出数据契约**：
```python
{
    "processed_signal": np.ndarray,  # (N, 7) 标准化后信号
    "cut_unique_id": str,
    "tool_id": str,
    "wear_label": dict,
    "filter_params": {
        "mean": np.ndarray,    # (7,) 每通道均值
        "std": np.ndarray      # (7,) 每通道标准差
    },
    "status": "PROCESSED"
}
```

---

### 8. modules/stage4_feature_extract/feature_extractor.py

**功能**：从预处理信号中提取168维多域特征向量

**关键类**：`FeatureExtractor`, `BatchFeatureExtractor`

**特征构成**（168维 = 7通道 × 24特征）：

| 特征类型 | 每通道数量 | 总维度 | 说明 |
|----------|-----------|--------|------|
| 时域特征 | 15 | 105 | 均值、方差、标准差、RMS、峰峰值等 |
| 频域特征 | 5 | 35 | 频谱均值、频率中心、谱方差等 |
| 时频域特征 | 4 | 28 | db4小波4层能量 |

**时域特征 (15个/通道)**：
1. mean - 均值
2. variance - 方差
3. std - 标准差
4. rms - 均方根
5. peak_to_peak - 峰峰值
6. skewness - 偏度
7. kurtosis - 峭度
8. crest_factor - 峰值因子
9. waveform_factor - 波形因子
10. impulse_factor - 冲击因子
11. margin_factor - 裕度因子
12. energy - 能量
13. entropy - 熵
14. zero_crossing_rate - 零交叉率
15. mean_derivative - 平均变化率

**频域特征 (5个/通道)**：
1. spectral_mean - 频谱均值
2. frequency_center - 频率中心
3. spectral_variance - 频谱方差
4. spectral_entropy - 谱熵
5. spectral_peak - 谱峰值

**时频域特征 (4个/通道)**：
1. wavelet_energy_level1 - 第1层小波能量
2. wavelet_energy_level2 - 第2层小波能量
3. wavelet_energy_level3 - 第3层小波能量
4. wavelet_energy_level4 - 第4层小波能量

**关键函数**：

| 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `extract(signal)` | (N,7)信号 | (168,)特征向量 | 提取完整特征 |
| `extract_time_features(signal)` | (N,7)信号 | (105,)时域特征 | 提取时域特征 |
| `extract_freq_features(signal)` | (N,7)信号 | (35,)频域特征 | 提取频域特征 |
| `extract_timefreq_features(signal)` | (N,7)信号 | (28,)时频特征 | 提取时频特征 |
| `validate_features(features)` | 特征向量 | bool | 验证特征合法性 |

**BatchFeatureExtractor 关键函数**：

| 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `process_batch(signal_list, cut_ids, tool_ids, wear_labels, save_dir)` | 信号列表等 | 结果列表 | 批量特征提取 |

**输入数据契约**：
```python
{
    "signal": np.ndarray,  # (N, 7) 预处理后信号
    "cut_unique_id": str,
    "tool_id": str,
    "wear_label": dict
}
```

**输出数据契约**：
```python
{
    "feature_vector": np.ndarray,  # (168,) 特征向量
    "cut_unique_id": str,
    "tool_id": str,
    "wear_label": dict
}
```

**保存文件格式**：
- 特征文件：`features_{cut_unique_id}.npy`
- 元数据文件：`meta_{cut_unique_id}.npz`

---

### 9. modules/stage5_augmentation/augmenter.py

**功能**：数据增强与类别平衡

**关键类**：`DataAugmenter`

**磨损等级划分**：
- 低磨损 (low)：&lt; 50 μm
- 中磨损 (medium)：50-150 μm
- 高磨损 (high)：&gt; 150 μm

**增强方法**：
1. 高斯噪声叠加
2. 幅值缩放
3. 时间偏移（循环移位）
4. SMOTE风格类别平衡

**关键函数**：

| 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `add_gaussian_noise(features, sigma_range)` | 特征, sigma范围 | 增强特征 | 添加高斯噪声 |
| `amplitude_scaling(features, scale_range)` | 特征, 缩放范围 | 增强特征 | 幅值缩放 |
| `time_shift(features, shift_range)` | 特征, 偏移范围 | 增强特征 | 循环移位时间偏移 |
| `sliding_window(signal, window_size, step)` | 信号, 窗口, 步长 | 窗口列表 | 滑动窗口切片 |
| `bin_wear_category(wear_label)` | 磨损标签 | category字符串 | 磨损类别分类 |
| `balance_classes(features, labels)` | 特征, 标签 | 平衡后数据 | SMOTE类别平衡 |
| `augment_dataset(features_list, labels_list, tool_ids, cut_unique_ids, n_augmentations, balance)` | 完整数据集 | 增强后数据集 | 完整增强流程 |

**输入数据契约**：
```python
{
    "features_list": List[np.ndarray],  # 每个元素 (168,)
    "labels_list": List[np.ndarray],      # 每个元素 (3,)
    "tool_ids": List[str],
    "cut_unique_ids": List[str],
    "n_augmentations": int,              # 默认2
    "balance": bool                      # 默认True
}
```

**输出数据契约**：
```python
{
    "features": np.ndarray,              # (N, 168) 增强后特征
    "labels": np.ndarray,                # (N, 3) 增强后标签
    "tool_ids": List[str],
    "cut_unique_ids": List[str],
    "source_indices": List[int]
}
```

**增强后数据保存**：
- `augmented_features.npy`
- `augmented_labels.npy`

---

### 10. modules/stage6_tfrecord_build/tfrecord_builder.py

**功能**：数据集划分与TFRecord格式构建

**关键类**：`TFRecordBuilder`

**数据集划分规则**：
- 训练集 (Train)：刀具 c1, c4
- 测试集 (Test)：刀具 c6

**关键函数**：

| 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `split_dataset(features, labels, tool_ids)` | 特征, 标签, 刀具ID | train/test字典 | 按刀具ID划分 |
| `_create_feature_spec()` | None | dict | 创建TFRecord特征规范 |
| `_serialize_example(...)` | 各字段 | tf.train.Example | 序列化单个样本 |
| `create_tfrecord(samples, output_path, cut_unique_ids)` | 样本数据, 路径 | 样本数量 | 创建TFRecord文件 |
| `read_tfrecord(input_path)` | 路径 | tf.data.Dataset | 读取TFRecord |
| `read_tfrecord_as_arrays(input_path)` | 路径 | Tuple | 读取为numpy数组 |
| `generate_metadata(train_count, test_count, version)` | 统计值 | 元数据字典 | 生成元数据 |
| `build(features, labels, tool_ids, cut_unique_ids, version)` | 完整数据 | 元数据 | 完整构建流程 |

**TFRecord特征规范**：
```python
{
    "features": tf.io.FixedLenFeature([168], dtype=tf.float32),
    "tool_id": tf.io.FixedLenFeature([], dtype=tf.string),
    "wear_label": tf.io.FixedLenFeature([3], dtype=tf.float32),
    "cut_unique_id": tf.io.FixedLenFeature([], dtype=tf.string)
}
```

**输入数据契约**：
```python
{
    "features": np.ndarray,     # (N, 168)
    "labels": np.ndarray,        # (N, 3)
    "tool_ids": List[str],
    "cut_unique_ids": List[str],
    "version": str
}
```

**输出数据契约**：
```python
# build() 返回的元数据
{
    "version": str,              # 版本标识
    "created_at": str,           # 创建时间
    "train_samples": int,        # 训练样本数
    "test_samples": int,         # 测试样本数
    "feature_dim": int,          # 特征维度 (168)
    "split_ratio": str,          # 划分比例 "80:20"
    "split_method": str,         # "by_tool_id"
    "train_tools": List[str],   # ["c1", "c4"]
    "test_tools": List[str]     # ["c6"]
}
```

**输出文件**：
- `data/processed/train.tfrecord`
- `data/processed/test.tfrecord`
- `data/processed/dataset_meta.json`

---

## 数据契约汇总

### 阶段间数据流

| 阶段 | 输入 | 输出 |
|------|------|------|
| **Stage1** | CSV文件, wear.csv | `{signal, tool_id, cut_unique_id, wear_label, status}` |
| **Stage2** | Stage1输出 | `{signal, tool_id, cut_unique_id, wear_label}` 或 `None` |
| **Stage3** | Stage2输出 | `{processed_signal, cut_unique_id, tool_id, wear_label, filter_params, status}` |
| **Stage4** | Stage3输出 | `{feature_vector, cut_unique_id, tool_id, wear_label}` |
| **Stage5** | Stage4输出 | `{features, labels, tool_ids, cut_unique_ids, source_indices}` |
| **Stage6** | Stage5输出 | `{train.tfrecord, test.tfrecord, dataset_meta.json}` |

---

## 全局配置常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `SUPPORTED_TOOL_IDS` | `['c1', 'c4', 'c6']` | 支持的刀具ID |
| `NUM_CHANNELS` | `7` | 信号通道数 |
| `FEATURE_DIM` | `168` | 特征向量维度 |
| `WEAR_LABEL_DIM` | `3` | 磨损标签维度（三刃） |
| `SAMPLE_RATE` | `20000` Hz | 采样频率 |
| `BANDPASS_FREQ` | `[100, 5000]` Hz | 带通滤波频率范围 |
| `WEAR_LOW_THRESHOLD` | `50` μm | 低磨损阈值 |
| `WEAR_HIGH_THRESHOLD` | `150` μm | 高磨损阈值 |

---

## 运行示例

```bash
# 完整流水线
python main.py --stage all

# 仅运行Stage1-3
python main.py --stage 1-3

# 运行特定阶段（需要按依赖顺序）
python main.py --stage 4 --run-id run_001_20250524_1530

# 从检查点恢复
python main.py --stage all --resume --run-id run_001_20250524_1530

# 分析处理结果
python analyze_results.py

# 绘制原始信号
python plot_raw_signal.py
```

---

## 依赖环境

```
numpy
scipy
tensorflow
pywt (PyWavelets)
scikit-learn
matplotlib
```

---

## 开发规范

详见 [bulid.md](bulid.md) 文档。

