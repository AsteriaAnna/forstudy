"""Stage5: 数据增强与样本重构模块

该模块对Stage4提取的特征数据进行数据增强，包括:
- 高斯噪声增强
- 幅值缩放增强
- 时间偏移增强
- 滑动窗口切片
- SMOTE风格类别平衡
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from core.logger import Logger


class DataAugmenter:
    """Stage5 数据增强器"""

    # 磨损等级分界线 (μm)
    WEAR_LOW_THRESHOLD = 50
    WEAR_HIGH_THRESHOLD = 150

    # 默认增强参数
    DEFAULT_SIGMA_RANGE = (0.01, 0.05)
    DEFAULT_SCALE_RANGE = (0.9, 1.1)
    DEFAULT_SHIFT_RANGE = (-50, 50)
    DEFAULT_WINDOW_SIZE = 1024
    DEFAULT_STEP = 512

    def __init__(self, run_id: str = None, random_seed: int = 42):
        """初始化数据增强器

        Args:
            run_id: 运行唯一标识符，用于日志记录
            random_seed: 随机种子，确保可复现性
        """
        self.logger = Logger(run_id) if run_id else Logger("default_run_id")
        self.rng = np.random.default_rng(random_seed)

        # 统计信息
        self.original_sample_count = 0
        self.augmented_sample_count = 0
        self.expansion_ratio = 0.0
        self.class_distribution_before = {}
        self.class_distribution_after = {}

    def add_gaussian_noise(
        self,
        features: np.ndarray,
        sigma_range: Tuple[float, float] = (0.01, 0.05)
    ) -> np.ndarray:
        """添加高斯噪声增强

        Args:
            features: 输入特征数组，形状 (N, 168)
            sigma_range: 噪声标准差范围

        Returns:
            添加噪声后的特征数组
        """
        sigma = self.rng.uniform(sigma_range[0], sigma_range[1])
        noise = self.rng.normal(0, sigma, features.shape)
        augmented = features + noise * np.std(features, axis=0)
        return augmented

    def amplitude_scaling(
        self,
        features: np.ndarray,
        scale_range: Tuple[float, float] = (0.9, 1.1)
    ) -> np.ndarray:
        """幅值缩放增强

        Args:
            features: 输入特征数组，形状 (N, 168)
            scale_range: 缩放因子范围

        Returns:
            缩放后的特征数组
        """
        scale = self.rng.uniform(scale_range[0], scale_range[1])
        return features * scale

    def time_shift(
        self,
        features: np.ndarray,
        shift_range: Tuple[int, int] = (-50, 50)
    ) -> np.ndarray:
        """时间偏移增强

        对特征向量进行循环移位操作

        Args:
            features: 输入特征数组，形状 (N, 168)
            shift_range: 偏移量范围

        Returns:
            偏移后的特征数组
        """
        shift = self.rng.integers(shift_range[0], shift_range[1] + 1)
        augmented = np.roll(features, shift, axis=1)
        return augmented

    def sliding_window(
        self,
        signal: np.ndarray,
        window_size: int = 1024,
        step: int = 512
    ) -> List[np.ndarray]:
        """滑动窗口切片

        将时序信号切分为固定大小的窗口

        Args:
            signal: 输入信号，形状 (N, 7)
            window_size: 窗口大小
            step: 步长

        Returns:
            窗口列表
        """
        windows = []
        num_samples = signal.shape[0]

        if num_samples < window_size:
            windows.append(signal)
            return windows

        start = 0
        while start + window_size <= num_samples:
            windows.append(signal[start:start + window_size])
            start += step

        # 如果最后一个窗口不完整，保留剩余部分
        if start < num_samples and (num_samples - start) >= window_size // 2:
            windows.append(signal[start:start + window_size])

        return windows

    def bin_wear_category(self, wear_label) -> str:
        """将磨损标签分类为低/中/高

        Args:
            wear_label: 磨损标签，可以是字典或数组

        Returns:
            磨损类别: 'low', 'medium', 'high'
        """
        # 处理字典格式
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

    def compute_class_distribution(
        self,
        labels: np.ndarray
    ) -> Dict[str, int]:
        """计算类别分布

        Args:
            labels: 标签数组，形状 (N, 3)

        Returns:
            类别计数字典 {'low': count, 'medium': count, 'high': count}
        """
        distribution = {'low': 0, 'medium': 0, 'high': 0}

        for i in range(labels.shape[0]):
            category = self.bin_wear_category(labels[i])
            distribution[category] += 1

        return distribution

    def augment_single(
        self,
        sample: Tuple[np.ndarray, np.ndarray],
        n_augmentations: int = 3
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """对单个样本进行多种增强

        Args:
            sample: (features, label) 元组
            n_augmentations: 每个样本的增强次数

        Returns:
            增强后的样本列表 [(aug_features, label), ...]
        """
        features, label = sample
        augmented_samples = []

        augmentation_methods = [
            lambda f: self.add_gaussian_noise(f, self.DEFAULT_SIGMA_RANGE),
            lambda f: self.amplitude_scaling(f, self.DEFAULT_SCALE_RANGE),
            lambda f: self.time_shift(f, self.DEFAULT_SHIFT_RANGE),
        ]

        for i in range(n_augmentations):
            method = augmentation_methods[i % len(augmentation_methods)]
            aug_features = method(features.copy())
            augmented_samples.append((aug_features, label))

        return augmented_samples

    def balance_classes(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """SMOTE风格类别平衡

        对少数类进行过采样以达到类别平衡

        Args:
            features: 特征数组，形状 (N, 168)
            labels: 标签数组，形状 (N, 3)

        Returns:
            平衡后的 (features, labels)
        """
        self.class_distribution_before = self.compute_class_distribution(labels)

        # 计算每个类别的样本数
        class_counts = self.class_distribution_before.copy()

        # 找到多数类的数量
        max_count = max(class_counts.values())

        # 将labels转换为统一格式（支持列表、数组和字典数组）
        if isinstance(labels, list):
            # 如果是字典列表，转换为数组
            if len(labels) > 0 and isinstance(labels[0], dict):
                labels_array = np.array([[l['flute_1'], l['flute_2'], l['flute_3']] for l in labels])
            else:
                labels_array = np.array(labels)
        elif isinstance(labels, np.ndarray):
            # 如果是numpy数组，检查元素是否为字典
            if labels.dtype == object and len(labels) > 0 and isinstance(labels[0], dict):
                labels_array = np.array([[l['flute_1'], l['flute_2'], l['flute_3']] for l in labels])
            else:
                labels_array = labels
        else:
            labels_array = labels
        
        # 使用转换后的标签数组
        labels = labels_array
        
        balanced_features_list = [features]
        balanced_labels_list = [labels]

        # 对每个类别进行过采样
        for category in ['low', 'medium', 'high']:
            count = class_counts[category]
            if count == 0:
                continue

            # 计算需要生成的合成样本数
            n_synthetic = max_count - count

            # 找到该类别的所有样本索引
            category_indices = []
            for i in range(len(labels)):
                if self.bin_wear_category(labels[i]) == category:
                    category_indices.append(i)

            if len(category_indices) == 0:
                continue

            category_features = features[category_indices]

            # 使用SMOTE风格生成合成样本
            n_iterations = n_synthetic // count + 1

            for _ in range(n_iterations):
                if len(balanced_features_list[-1]) >= max_count * 3:
                    break

                # 随机选择两个样本进行插值（使用category_features的本地索引）
                local_indices = self.rng.choice(
                    len(category_features),
                    size=min(len(category_features), n_synthetic),
                    replace=False
                )

                for local_idx in local_indices:
                    # 随机选择另一个样本（使用本地索引）
                    other_local_idx = self.rng.choice(len(category_features))
                    if other_local_idx == local_idx:
                        other_local_idx = (local_idx + 1) % len(category_features)

                    # 生成合成样本 (随机插值)
                    alpha = self.rng.uniform(0, 1)
                    synthetic_features = category_features[local_idx] * alpha + category_features[other_local_idx] * (1 - alpha)

                    # 随机选择增强方法
                    if self.rng.random() < 0.5:
                        synthetic_features = self.add_gaussian_noise(
                            synthetic_features.reshape(1, -1),
                            self.DEFAULT_SIGMA_RANGE
                        ).flatten()

                    balanced_features_list.append(synthetic_features.reshape(1, -1))
                    balanced_labels_list.append(labels[category_indices[local_idx]].reshape(1, -1))

        # 合并所有特征和标签
        balanced_features = np.vstack(balanced_features_list)
        balanced_labels = np.vstack(balanced_labels_list)

        self.class_distribution_after = self.compute_class_distribution(balanced_labels)

        self.logger.info(
            f"类别平衡完成: 原始分布={self.class_distribution_before}, "
            f"平衡后分布={self.class_distribution_after}"
        )

        return balanced_features, balanced_labels

    def validate_augmented(self, features: np.ndarray) -> bool:
        """验证增强后数据的物理有效性

        Args:
            features: 增强后的特征数组

        Returns:
            True表示数据有效，False表示存在异常
        """
        if features is None or len(features) == 0:
            return False

        # 检查是否存在NaN或Inf
        if not np.all(np.isfinite(features)):
            self.logger.warn("增强数据包含NaN或Inf值")
            return False

        # 检查是否存在全零向量
        zero_vector = np.zeros(features.shape[1])
        for i in range(features.shape[0]):
            if np.allclose(features[i], zero_vector, atol=1e-6):
                self.logger.warn(f"样本{i}为全零向量")
                return False

        # 检查特征方差（不应为0）
        feature_std = np.std(features, axis=0)
        if np.any(feature_std == 0):
            self.logger.warn("存在方差为零的特征")
            return False

        return True

    def augment_dataset(
        self,
        features_list: List[np.ndarray],
        labels_list: List[np.ndarray],
        tool_ids: List[str],
        cut_unique_ids: List[str],
        n_augmentations: int = 2,
        balance: bool = True
    ) -> Dict:
        """对整个数据集进行增强

        Args:
            features_list: 特征列表，每个元素形状 (168,)
            labels_list: 标签列表，每个元素形状 (3,)
            tool_ids: 刀具ID列表
            cut_unique_ids: 切割唯一ID列表
            n_augmentations: 每个样本的增强次数
            balance: 是否进行类别平衡

        Returns:
            增强后的数据集字典:
            {
                'features': np.ndarray (N_samples, 168),
                'labels': np.ndarray (N_samples, 3),
                'tool_ids': list,
                'cut_unique_ids': list,
                'source_indices': list
            }
        """
        self.original_sample_count = len(features_list)

        self.logger.info(
            f"开始数据增强: 原始样本数={self.original_sample_count}, "
            f"增强倍数={n_augmentations}, 平衡={balance}"
        )

        # 转换为数组
        features = np.array(features_list)
        labels = np.array(labels_list)

        # 记录原始类别分布
        self.class_distribution_before = self.compute_class_distribution(labels)
        self.logger.stat(
            f"增强前类别分布: 低磨损={self.class_distribution_before['low']}, "
            f"中磨损={self.class_distribution_before['medium']}, "
            f"高磨损={self.class_distribution_before['high']}"
        )

        # 存储增强样本
        aug_features_list = []
        aug_labels_list = []
        aug_tool_ids = []
        aug_cut_unique_ids = []
        aug_source_indices = []

        # 对每个样本进行增强
        for i in range(len(features_list)):
            sample = (features[i], labels[i])

            # 原始样本
            aug_features_list.append(features[i])
            aug_labels_list.append(labels[i])
            aug_tool_ids.append(tool_ids[i])
            aug_cut_unique_ids.append(cut_unique_ids[i])
            aug_source_indices.append(i)

            # 增强样本
            augmented_samples = self.augment_single(sample, n_augmentations)
            for aug_feat, aug_label in augmented_samples:
                aug_features_list.append(aug_feat)
                aug_labels_list.append(aug_label)
                aug_tool_ids.append(tool_ids[i])
                aug_cut_unique_ids.append(cut_unique_ids[i])
                aug_source_indices.append(i)

        # 转换为数组
        aug_features = np.array(aug_features_list)
        aug_labels = np.array(aug_labels_list)

        # 类别平衡
        if balance:
            aug_features, aug_labels = self.balance_classes(aug_features, aug_labels)

            # 扩展其他列表以匹配平衡后的数据
            n_original = len(aug_features_list)
            n_balanced = len(aug_features)

            if n_balanced > n_original:
                # 补充tool_ids和cut_unique_ids（复制原始数据）
                additional_tool_ids = []
                additional_cut_unique_ids = []
                additional_source_indices = []

                for _ in range(n_balanced - n_original):
                    # 从原始数据中随机选择一个
                    idx = self.rng.integers(0, len(tool_ids))
                    additional_tool_ids.append(tool_ids[idx])
                    additional_cut_unique_ids.append(cut_unique_ids[idx])
                    additional_source_indices.append(idx)

                aug_tool_ids.extend(additional_tool_ids)
                aug_cut_unique_ids.extend(additional_cut_unique_ids)
                aug_source_indices.extend(additional_source_indices)

        # 验证增强数据
        if not self.validate_augmented(aug_features):
            self.logger.error("增强数据验证失败")
            raise ValueError("Augmented data validation failed")

        self.augmented_sample_count = len(aug_features)
        self.expansion_ratio = self.augmented_sample_count / self.original_sample_count

        # 记录平衡后的类别分布
        self.class_distribution_after = self.compute_class_distribution(aug_labels)

        self.logger.stat(
            f"数据增强完成: 原始样本={self.original_sample_count}, "
            f"增强后样本={self.augmented_sample_count}, "
            f"扩展比例={self.expansion_ratio:.2f}x"
        )
        self.logger.stat(
            f"增强后类别分布: 低磨损={self.class_distribution_after['low']}, "
            f"中磨损={self.class_distribution_after['medium']}, "
            f"高磨损={self.class_distribution_after['high']}"
        )

        return {
            'features': aug_features,
            'labels': aug_labels,
            'tool_ids': aug_tool_ids,
            'cut_unique_ids': aug_cut_unique_ids,
            'source_indices': aug_source_indices
        }

    def get_statistics(self) -> Dict:
        """获取增强统计信息

        Returns:
            包含统计信息的字典
        """
        return {
            'original_sample_count': self.original_sample_count,
            'augmented_sample_count': self.augmented_sample_count,
            'expansion_ratio': self.expansion_ratio,
            'class_distribution_before': self.class_distribution_before,
            'class_distribution_after': self.class_distribution_after
        }
