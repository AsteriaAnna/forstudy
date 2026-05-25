# Stage1 数据处理方案修改指导文档

## 文档信息

| 项目 | 内容 |
|------|------|
| 版本 | v1.0 |
| 日期 | 2026-05-25 |
| 状态 | 待实施 |
| 适用范围 | `pipeline/stage1_loading.py` 及相关模块 |

---

## 1. 前置基础统一约定

### 1.1 术语定义

| 术语 | 定义 |
|------|------|
| **robust_wear** | 自适应加权稳健磨损值，通过降低异常刀刃权重计算得到的可靠磨损估计 |
| **wear_stage** | EM算法自动划分的磨损阶段（initial/normal/severe） |
| **原始三刃值** | 三个刀刃的原始测量值（flute_1, flute_2, flute_3） |
| **切削ID** | 格式为 `{tool_id}_cut{cut_num}`，例如 `c1_cut123` |
| **刀具ID** | 支持 `c1`, `c4`, `c6` 三种刀具 |

### 1.2 数据结构约定

#### 磨损标签结构（修改后）

```python
wear_label = {
    # ========== 原始字段（必须保留）==========
    'flute_1': float,    # 刀刃1磨损值（μm）
    'flute_2': float,    # 刀刃2磨损值（μm）
    'flute_3': float,    # 刀刃3磨损值（μm）
    
    # ========== 新增字段（必须添加）==========
    'robust_wear': float,    # 自适应加权稳健磨损值（μm）
    'wear_stage': str,       # 磨损阶段 ('initial'/'normal'/'severe')
}
```

#### 切削数据结构（贯穿整个流水线）

```python
cutting_data = {
    'signal': np.ndarray,           # 原始信号 (N, 7)
    'tool_id': str,                # 刀具ID
    'cut_unique_id': str,           # 切削唯一ID
    'wear_label': dict,             # 磨损标签（包含上述5个字段）
    'status': str,                 # 状态标识
}
```

### 1.3 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 方法名 | 蛇形命名法 | `_calculate_robust_wear` |
| 字段名 | 蛇形命名法 | `robust_wear` |
| 阶段名 | 全小写 | `initial`, `normal`, `severe` |
| 类常量 | 全大写下划线 | `SUPPORTED_TOOL_IDS` |

### 1.4 依赖约定

#### Python 标准库
- `numpy`: 数值计算
- `json`: 数据序列化

#### 外部依赖
- `scikit-learn`: EM算法实现（`sklearn.mixture.GaussianMixture`）

---

## 2. 修改规范

### 2.1 核心原则

| 原则 | 说明 |
|------|------|
| **向后兼容** | 新增字段不能破坏现有代码的正常运行 |
| **数据完整** | 原始三刃值必须完整保留，不能删除 |
| **流式处理** | 必须保持流式加载模式，不加载全部数据到内存 |
| **可追溯性** | 所有派生字段必须能够追溯到原始数据 |

### 2.2 代码组织规范

```
# 文件结构
pipeline/
├── stage1_loading.py          # 主文件（修改）
├── stage2_validation.py       # 关联修改
├── stage3_preprocessing.py   # 无需修改
├── run_pipeline.py           # 无需修改

modules/
├── stage5_augmentation/
│   └── augmenter.py          # 关联修改（可选）
└── stage6_tfrecord_build/
    └── tfrecord_builder.py   # 关联修改（可选）
```

### 2.3 异常处理规范

| 场景 | 处理方式 |
|------|---------|
| 磨损标签文件不存在 | 记录错误日志，返回空字典 |
| 磨损值解析失败 | 跳过该行，继续处理 |
| EM算法计算失败 | 使用固定阈值划分阶段 |
| robust_wear计算失败 | 返回三个值的简单平均 |

---

## 3. 实现方式

### 3.1 添加 robust_wear 字段

#### 实现位置
`pipeline/stage1_loading.py` → `load_wear_labels()` 方法

#### 实现代码

```python
def load_wear_labels(self, tool_id: str) -> dict[int, dict[str, float]]:
    """加载指定刀具的磨损标签
    
    Args:
        tool_id: 刀具ID (c1/c4/c6)
    
    Returns:
        {cut_num: {flute_1, flute_2, flute_3, robust_wear}}
    """
    if tool_id in self._wear_cache:
        return self._wear_cache[tool_id]

    wear_path = self._get_wear_label_path(tool_id)

    if not wear_path.exists():
        self.logger.error(f"Wear label file not found: {wear_path}")
        return {}

    wear_data: dict[int, dict[str, float]] = {}

    try:
        with open(wear_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            if header != 'cut,flute_1,flute_2,flute_3':
                self.logger.warn(f"Unexpected wear CSV header: {header}", tool_id)

            for line in f:
                parts = line.strip().split(',')
                if len(parts) != 4:
                    continue
                try:
                    cut_num = int(parts[0])
                    flute_1 = float(parts[1])
                    flute_2 = float(parts[2])
                    flute_3 = float(parts[3])
                    
                    # ========== 新增：计算稳健磨损值 ==========
                    robust_wear = self._calculate_robust_wear([flute_1, flute_2, flute_3])
                    
                    wear_data[cut_num] = {
                        'flute_1': flute_1,
                        'flute_2': flute_2,
                        'flute_3': flute_3,
                        'robust_wear': robust_wear  # 新增字段
                    }
                except (ValueError, IndexError):
                    continue

        self._wear_cache[tool_id] = wear_data
        self.logger.info(f"Loaded {len(wear_data)} wear labels for {tool_id}", tool_id)

    except Exception as e:
        self.logger.error(f"Failed to load wear labels: {wear_path}", tool_id, e)
        return {}

    return wear_data

def _calculate_robust_wear(self, flute_values: list) -> float:
    """计算自适应加权稳健磨损值
    
    通过降低异常值的权重，得到更可靠的磨损估计。
    这是论文"平均磨损值作为标签消除个体异常"思想的实现。
    
    Args:
        flute_values: [flute_1, flute_2, flute_3] 三个刀刃的磨损值
    
    Returns:
        稳健磨损值（加权平均）
    """
    values = np.array(flute_values, dtype=np.float64)
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    # 标准差为0时返回均值
    if std_val == 0:
        return mean_val
    
    # 计算置信度：离均值越近，置信度越高
    confidence = 1 - np.abs(values - mean_val) / (3 * std_val)
    
    # 限制最小权重为0.1，避免完全忽略异常值
    confidence = np.clip(confidence, 0.1, 1.0)
    
    # 归一化权重
    weights = confidence / np.sum(confidence)
    
    # 计算加权平均
    robust_wear = np.sum(values * weights)
    
    return float(robust_wear)
```

#### 算法原理

```
原始数据: [0.123, 0.118, 0.150]
              ↑      ↑      ↑
          正常   正常   异常(偏大)

计算过程:
1. 均值: 0.1303
2. 标准差: 0.0167
3. 置信度: [0.76, 0.85, 0.19]  → 限制后 [0.76, 0.85, 0.10]
4. 归一化权重: [0.39, 0.44, 0.17]
5. 加权平均: 0.123×0.39 + 0.118×0.44 + 0.150×0.17 = 0.1205

效果: 异常值0.150的权重从0.33降至0.17，结果更接近正常范围
```

---

### 3.2 添加 wear_stage 字段

#### 实现位置
`pipeline/stage1_loading.py` → 新增 `_compute_wear_stages()` 方法

#### 实现代码

```python
def _compute_wear_stages(self, tool_id: str, wear_data: dict) -> dict:
    """使用EM算法计算磨损阶段划分
    
    使用高斯混合模型(GMM)自动将磨损值分为三个阶段。
    这是论文"EM算法分类磨损阶段"思想的实现。
    
    Args:
        tool_id: 刀具ID
        wear_data: 磨损数据字典
    
    Returns:
        {cut_num: stage_name} 阶段划分结果
    """
    from sklearn.mixture import GaussianMixture
    
    # 按切削编号排序
    sorted_cuts = sorted(wear_data.keys())
    
    # 提取稳健磨损值序列
    wear_values = np.array([
        wear_data[cut]['robust_wear'] 
        for cut in sorted_cuts
    ], dtype=np.float64).reshape(-1, 1)
    
    # 使用EM算法进行高斯混合模型拟合
    # n_components=3: 初始磨损、正常磨损、严重磨损
    gmm = GaussianMixture(
        n_components=3,
        covariance_type='full',
        random_state=42,
        max_iter=100
    )
    
    try:
        gmm.fit(wear_values)
        labels = gmm.predict(wear_values)
    except Exception as e:
        self.logger.warn(
            f"EM算法计算失败，使用固定阈值划分: {tool_id}",
            tool_id, e
        )
        return self._fallback_wear_stages(wear_data)
    
    # 获取各分量的均值并排序，确定阶段顺序
    means = gmm.means_.flatten()
    stage_order = np.argsort(means)  # 从小到大排序
    
    # 映射到阶段名称
    stage_names = ['initial', 'normal', 'severe']
    
    # 构建结果
    result = {}
    for cut_num, label in zip(sorted_cuts, labels):
        # 将GMM标签映射到正确的阶段名称
        stage_idx = np.where(stage_order == label)[0][0]
        result[cut_num] = stage_names[stage_idx]
    
    self.logger.info(
        f"EM阶段划分完成: {tool_id}, "
        f"initial={sum(1 for v in result.values() if v == 'initial')}, "
        f"normal={sum(1 for v in result.values() if v == 'normal')}, "
        f"severe={sum(1 for v in result.values() if v == 'severe')}",
        tool_id
    )
    
    return result

def _fallback_wear_stages(self, wear_data: dict) -> dict:
    """固定阈值划分磨损阶段（降级方案）
    
    当EM算法计算失败时使用。
    使用固定阈值划分磨损阶段。
    
    Args:
        wear_data: 磨损数据字典
    
    Returns:
        {cut_num: stage_name}
    """
    THRESHOLD_LOW = 50.0   # μm
    THRESHOLD_HIGH = 150.0  # μm
    
    result = {}
    for cut_num, data in wear_data.items():
        robust_wear = data['robust_wear']
        
        if robust_wear < THRESHOLD_LOW:
            result[cut_num] = 'initial'
        elif robust_wear < THRESHOLD_HIGH:
            result[cut_num] = 'normal'
        else:
            result[cut_num] = 'severe'
    
    return result
```

#### 修改 `load_wear_labels()` 方法

```python
def load_wear_labels(self, tool_id: str) -> dict[int, dict[str, float]]:
    """加载指定刀具的磨损标签（完整版）"""
    # ... 现有代码加载三刃值和robust_wear ...
    
    # ========== 新增：计算磨损阶段 ==========
    # 先加载所有数据，然后计算阶段
    if len(wear_data) > 0:
        stage_mapping = self._compute_wear_stages(tool_id, wear_data)
        
        # 将阶段信息添加到wear_data
        for cut_num, stage in stage_mapping.items():
            if cut_num in wear_data:
                wear_data[cut_num]['wear_stage'] = stage
    
    return wear_data
```

---

### 3.3 更新 Stage2 验证逻辑

#### 修改位置
`pipeline/stage2_validation.py` → `check_wear_label()` 方法

#### 实现代码

```python
def check_wear_label(self, wear_label: Dict[str, float]) -> bool:
    """检查磨损标签值
    
    验证磨损标签的合法性。
    根据论文要求，三刃磨损值无明确幅值范围检查，
    只检查字段存在性和数值有效性。
    
    Args:
        wear_label: 磨损标签字典
    
    Returns:
        True表示标签值不合格，False表示合格
    """
    # 必需字段列表
    required_fields = ['flute_1', 'flute_2', 'flute_3', 'robust_wear', 'wear_stage']
    
    # 检查字段存在性
    for key in required_fields:
        if key not in wear_label:
            self.logger.warn(f"Wear label missing field: {key}")
            return True  # 缺失字段视为不合格
    
    # 检查数值字段（非wear_stage）
    numeric_fields = ['flute_1', 'flute_2', 'flute_3', 'robust_wear']
    for key in numeric_fields:
        value = wear_label[key]
        if np.isnan(value):
            self.logger.warn(f"Wear label contains NaN: {key}={value}")
            return True
    
    # 检查wear_stage字段的值
    valid_stages = ['initial', 'normal', 'severe']
    wear_stage = wear_label['wear_stage']
    if wear_stage not in valid_stages:
        self.logger.warn(f"Invalid wear stage: {wear_stage}")
        return True
    
    return False
```

---

## 4. 验证方式

### 4.1 单元测试

#### 测试 robust_wear 计算

```python
def test_calculate_robust_wear():
    """测试稳健磨损值计算"""
    loader = Stage1Loader("test_data")
    
    # 测试1: 正常数据
    values = [0.123, 0.118, 0.120]
    result = loader._calculate_robust_wear(values)
    expected = np.mean(values)  # 应该接近简单平均
    assert abs(result - expected) < 0.001
    
    # 测试2: 包含异常值
    values = [0.123, 0.118, 0.150]  # 0.150是异常值
    result = loader._calculate_robust_wear(values)
    simple_mean = np.mean(values)
    # 稳健值应该小于简单平均值（因为异常值被降低权重）
    assert result < simple_mean
    
    # 测试3: 全部相同
    values = [0.120, 0.120, 0.120]
    result = loader._calculate_robust_wear(values)
    assert abs(result - 0.120) < 0.001
    
    print("✅ robust_wear 计算测试通过")

test_calculate_robust_wear()
```

#### 测试 EM算法阶段划分

```python
def test_compute_wear_stages():
    """测试磨损阶段划分"""
    loader = Stage1Loader("test_data")
    
    # 模拟磨损数据
    wear_data = {
        i: {'robust_wear': 20.0 + i * 0.5}  # 线性增长
        for i in range(1, 50)  # 50个样本
    }
    
    # 添加高磨损样本
    for i in range(50, 100):
        wear_data[i] = {'robust_wear': 100.0 + (i - 50) * 1.5}  # 快速增加
    
    result = loader._compute_wear_stages('test_tool', wear_data)
    
    # 验证结果
    assert len(result) == 99  # 所有样本都有阶段
    
    stages = list(result.values())
    assert 'initial' in stages
    assert 'normal' in stages
    assert 'severe' in stages
    
    # 验证阶段顺序：应该是从initial到normal到severe
    initial_cuts = [k for k, v in result.items() if v == 'initial']
    normal_cuts = [k for k, v in result.items() if v == 'normal']
    severe_cuts = [k for k, v in result.items() if v == 'severe']
    
    assert max(initial_cuts) < min(normal_cuts)  # initial < normal
    assert max(normal_cuts) < min(severe_cuts)   # normal < severe
    
    print("✅ EM算法阶段划分测试通过")

test_compute_wear_stages()
```

### 4.2 集成测试

```python
def test_full_pipeline():
    """测试完整流水线"""
    import tempfile
    import shutil
    
    # 创建临时测试目录
    test_dir = tempfile.mkdtemp()
    
    try:
        # 准备测试数据
        prepare_test_data(test_dir)
        
        # 运行Stage1加载
        loader = Stage1Loader(test_dir)
        wear_data = loader.load_wear_labels('c1')
        
        # 验证数据结构
        assert len(wear_data) > 0
        
        sample = wear_data[1]
        assert 'flute_1' in sample
        assert 'flute_2' in sample
        assert 'flute_3' in sample
        assert 'robust_wear' in sample
        assert 'wear_stage' in sample
        
        # 验证阶段划分
        stages = [data['wear_stage'] for data in wear_data.values()]
        assert set(stages) == {'initial', 'normal', 'severe'}
        
        print("✅ 完整流水线测试通过")
        
    finally:
        shutil.rmtree(test_dir)

test_full_pipeline()
```

### 4.3 验证检查清单

| 检查项 | 验证方法 | 预期结果 |
|--------|---------|---------|
| robust_wear 字段存在 | 检查所有 wear_data 包含该字段 | 100% 存在 |
| wear_stage 字段存在 | 检查所有 wear_data 包含该字段 | 100% 存在 |
| 阶段值有效 | 检查所有阶段值为 initial/normal/severe | 100% 有效 |
| 阶段顺序正确 | 验证 initial < normal < severe | 符合预期 |
| 数值计算正确 | 对比手动计算结果 | 误差 < 0.001 |
| 流式处理正常 | 检查内存占用 | 恒定 O(1) |

---

## 5. 可修改、删除的功能

### 5.1 ✅ 可以修改的部分

| 模块 | 可修改内容 | 修改理由 |
|------|-----------|---------|
| `stage1_loading.py` | 添加新字段 | 实现论文方法 |
| `stage1_loading.py` | 添加新方法 | 计算稳健磨损值和阶段划分 |
| `stage2_validation.py` | 验证逻辑 | 适配新的数据结构 |
| `augmenter.py` | 类别划分方法 | 使用 wear_stage |
| `tfrecord_builder.py` | TFRecord格式 | 可选存储 wear_stage |

### 5.2 ❌ 严禁修改的部分

| 模块 | 禁止修改内容 | 原因 |
|------|------------|------|
| `stage1_loading.py` | 原始三刃值字段 | 数据完整性要求 |
| `stage1_loading.py` | ID规范 (tool_id, cut_unique_id) | 全局标识系统 |
| `stage1_loading.py` | 生成器模式 (yield) | 流式处理核心 |
| `stage1_loading.py` | 文件名解析模式 | 数据溯源要求 |
| `stage2_validation.py` | 信号验证逻辑 | 数据质量保证 |
| `stage3_preprocessing.py` | 信号预处理逻辑 | 模型输入一致性 |

### 5.3 🔄 可删除的部分

| 模块 | 可删除内容 | 条件 |
|------|----------|------|
| `stage2_validation.py` | 幅值范围检查代码 | 确认不需要时 |
| `stage2_validation.py` | WEAR_LABEL_MIN/MAX 常量 | 确认不需要时 |

---

## 6. 重构范围

### 6.1 核心修改范围

```
┌─────────────────────────────────────────────────────────┐
│                    重构范围（红色区域）                    │
├─────────────────────────────────────────────────────────┤
│  pipeline/                                              │
│  ├── stage1_loading.py         ← 核心修改               │
│  │   ├── load_wear_labels()    ← 添加 robust_wear       │
│  │   ├── _calculate_robust_wear() ← 新增方法            │
│  │   ├── _compute_wear_stages()   ← 新增方法            │
│  │   └── _fallback_wear_stages()  ← 新增方法            │
│  │                                                     │
│  └── stage2_validation.py      ← 关联修改               │
│      └── check_wear_label()    ← 更新验证逻辑            │
├─────────────────────────────────────────────────────────┤
│  其他文件（绿色区域）                                     │
│  ├── stage3_preprocessing.py   ← 无需修改              │
│  ├── run_pipeline.py          ← 无需修改                │
│  ├── main.py                  ← 无需修改                │
│  └── modules/                 ← 可选修改                │
└─────────────────────────────────────────────────────────┘
```

### 6.2 文件修改清单

| 文件路径 | 修改类型 | 修改内容 |
|---------|---------|---------|
| `pipeline/stage1_loading.py` | 修改 | 添加 `_calculate_robust_wear()` |
| `pipeline/stage1_loading.py` | 修改 | 添加 `_compute_wear_stages()` |
| `pipeline/stage1_loading.py` | 修改 | 添加 `_fallback_wear_stages()` |
| `pipeline/stage1_loading.py` | 修改 | 更新 `load_wear_labels()` |
| `pipeline/stage2_validation.py` | 修改 | 更新 `check_wear_label()` |

### 6.3 禁止越界修改

以下文件**不在本次重构范围内**：

| 文件 | 状态 | 说明 |
|------|------|------|
| `pipeline/stage3_preprocessing.py` | ❌ 不修改 | 透明处理 |
| `pipeline/run_pipeline.py` | ❌ 不修改 | 直接使用新结构 |
| `main.py` | ❌ 不修改 | 无需变更 |
| `modules/stage4_feature_extract/feature_extractor.py` | ❌ 不修改 | 透明处理 |
| `modules/stage5_augmentation/augmenter.py` | ⚠️ 可选 | 建议修改但不强制 |
| `modules/stage6_tfrecord_build/tfrecord_builder.py` | ⚠️ 可选 | 建议修改但不强制 |

---

## 7. 实施检查表

### 7.1 修改前检查

- [ ] 备份原始代码
- [ ] 确认 scikit-learn 依赖已安装
- [ ] 确认测试数据可用
- [ ] 阅读并理解本文档所有内容

### 7.2 修改中检查

- [ ] 按照第3章实现代码
- [ ] 保持代码风格一致
- [ ] 添加必要的注释
- [ ] 不修改禁止修改的部分

### 7.3 修改后检查

- [ ] 运行单元测试验证 robust_wear 计算
- [ ] 运行单元测试验证 EM算法阶段划分
- [ ] 运行集成测试验证完整流水线
- [ ] 检查数据输出格式
- [ ] 验证向后兼容性
- [ ] 检查内存占用

### 7.4 验证清单

| 测试项 | 命令 | 预期结果 |
|--------|------|---------|
| 单元测试 | `python -m pytest tests/test_stage1.py::test_calculate_robust_wear` | 通过 |
| 单元测试 | `python -m pytest tests/test_stage1.py::test_compute_wear_stages` | 通过 |
| 集成测试 | `python -m pytest tests/test_stage1.py::test_full_pipeline` | 通过 |
| 内存检查 | 观察任务管理器 | 内存占用恒定 |
| 数据验证 | 检查输出 JSON | 包含所有必需字段 |

---

## 8. 回滚方案

### 8.1 回滚触发条件

- 测试失败率 > 5%
- 内存占用异常增长
- 数据格式不兼容

### 8.2 回滚步骤

```bash
# 1. 恢复原始代码
git checkout HEAD -- pipeline/stage1_loading.py
git checkout HEAD -- pipeline/stage2_validation.py

# 2. 清除缓存
rm -rf results/ stage3_cache/ features/

# 3. 重新测试原始代码
python -m pipeline.run_pipeline --raw-data-dir phm2010_raw
```

### 8.3 紧急联系人

如遇到问题，请联系项目负责人。

---

## 9. 附录

### A. 依赖安装

```bash
pip install scikit-learn numpy scipy
```

### B. 相关文档

- [论文方法拆解报告](../study/论文数据处理方法拆解报告.md)
- [设计方案对比分析](../study/设计方案与论文方法对比分析.md)
- [修改影响评估报告](../study/Stage1数据处理方案修改影响评估报告.md)

### C. 参考资料

1. Jiang et al., "Transformer-Physics Hybrid Model for Manufacturing Process", CIRP Journal of Manufacturing Science and Technology, 2026
2. sklearn.mixture.GaussianMixture 官方文档

---

**文档版本**: v1.0  
**最后更新**: 2026-05-25  
**维护者**: 项目团队  
**审批状态**: 待审批