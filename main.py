"""PHM2010 项目主入口

该模块是PHM2010刀具磨损预测项目的主入口点，负责协调各个处理阶段。

修改说明：
- 复用 run_pipeline.py 的 DataPipeline 实现 Stage1-3 流式处理
- Stage1-3 结果写入磁盘缓存
- Stage4 从磁盘缓存批量读取数据
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.logger import Logger
from core.monitor import GlobalMonitor
from core.utils import (
    ensure_dir,
    get_data_dir,
    get_results_dir,
    get_cache_path,
    get_interim_dir,
    get_run_id as generate_run_id,
)
from pipeline.run_pipeline import DataPipeline  # 复用流式处理
from modules.stage4_feature_extract.feature_extractor import BatchFeatureExtractor
from modules.stage5_augmentation.augmenter import DataAugmenter
from modules.stage6_tfrecord_build.tfrecord_builder import TFRecordBuilder


# 默认路径配置
DEFAULT_RAW_DATA_DIR = "phm2010_raw"
DEFAULT_OUTPUT_DIR = "results/{run_id}/"
DEFAULT_PROCESSED_DATA_DIR = "data/processed/"


class PipelineRunner:
    """PHM2010 数据处理流水线运行器"""

    def __init__(
        self,
        run_id: str,
        data_dir: str = DEFAULT_RAW_DATA_DIR,
        output_dir: Optional[str] = None,
        resume: bool = False,
    ):
        """初始化流水线运行器

        Args:
            run_id: 运行唯一标识符
            data_dir: 原始数据目录路径
            output_dir: 输出目录路径
            resume: 是否从检查点恢复
        """
        self.run_id = run_id
        self.data_dir = data_dir
        self.output_dir = output_dir or get_results_dir(run_id)
        self.resume = resume

        # 创建输出目录
        ensure_dir(self.output_dir)
        ensure_dir(get_data_dir(run_id))

        # 初始化日志器
        self.logger = Logger(run_id)
        self.logger.info(f"初始化流水线运行器: run_id={run_id}")

        # 初始化监控器
        self.monitor = GlobalMonitor()
        self.monitor.reset()

        # 统计信息
        self.stats = {
            "samples_loaded": 0,
            "samples_validated": 0,
            "samples_preprocessed": 0,
            "features_extracted": 0,
            "samples_augmented": 0,
            "train_samples": 0,
            "test_samples": 0,
        }

    def run_pipeline_stages_1_3(self) -> List[Dict]:
        """运行Stage1-3完整流水线（使用流式处理）

        Returns:
            预处理后的数据列表（从磁盘缓存加载）
        """
        self.logger.info("=" * 50)
        self.logger.info("开始 Stage1-3: 流式数据处理")
        self.logger.info("=" * 50)

        # 复用 run_pipeline.py 的 DataPipeline 进行流式处理
        pipeline = DataPipeline(
            raw_data_dir=self.data_dir,
            run_id=self.run_id,
            output_dir="results",
            resume=self.resume
        )

        # 执行流式处理
        pipeline_stats = pipeline.run()

        # 更新统计信息
        self.stats["samples_loaded"] = pipeline_stats["total_processed"]
        self.stats["samples_validated"] = pipeline_stats["success_count"]
        self.stats["samples_preprocessed"] = pipeline_stats["success_count"]

        # 从磁盘缓存加载预处理后的数据
        preprocessed_data = self._load_preprocessed_from_cache(self.run_id)

        self.logger.stat(f"Stage1-3 完成: 预处理样本数={len(preprocessed_data)}")
        return preprocessed_data

    def _load_preprocessed_from_cache(self, run_id: str) -> List[Dict]:
        """从磁盘缓存加载预处理后的数据

        Args:
            run_id: 运行ID

        Returns:
            预处理数据列表
        """
        cache_dir = Path("results") / run_id / "data"
        preprocessed_data = []

        if not cache_dir.exists():
            self.logger.warn(f"缓存目录不存在: {cache_dir}")
            return preprocessed_data

        # 遍历所有 .npz 文件
        npz_files = list(cache_dir.glob("*.npz"))

        for npz_file in npz_files:
            try:
                data = np.load(npz_file, allow_pickle=True)

                # 构建数据字典
                cut_data = {
                    "cut_unique_id": npz_file.stem,
                    "tool_id": str(data["tool_id"]),
                    "processed_signal": data["processed_signal"],
                    "wear_label": {
                        "flute_1": float(data["wear_label_flute_1"]),
                        "flute_2": float(data["wear_label_flute_2"]),
                        "flute_3": float(data["wear_label_flute_3"]),
                        "robust_wear": float(data["wear_label_robust_wear"]),
                        "wear_stage": str(data["wear_label_wear_stage"])
                    },
                    "filter_params": {
                        "mean": data["filter_mean"],
                        "std": data["filter_std"]
                    },
                    "status": str(data["status"])
                }

                preprocessed_data.append(cut_data)

            except Exception as e:
                self.logger.error(f"加载缓存文件失败: {npz_file}", exc=e)

        # 按 cut_unique_id 排序
        preprocessed_data.sort(key=lambda x: x["cut_unique_id"])

        return preprocessed_data

    def run_stage4_feature_extraction(self, preprocessed_data: List[Dict]) -> List[Dict]:
        """运行Stage4: 批量特征提取

        Args:
            preprocessed_data: 预处理后的数据列表

        Returns:
            特征提取结果列表
        """
        self.logger.info("=" * 50)
        self.logger.info("开始 Stage4: 多域特征提取")
        self.logger.info("=" * 50)

        # 准备数据
        signal_list = [d["processed_signal"] for d in preprocessed_data]
        cut_ids = [d["cut_unique_id"] for d in preprocessed_data]
        tool_ids = [d["tool_id"] for d in preprocessed_data]
        wear_labels = [d["wear_label"] for d in preprocessed_data]

        # 创建缓存目录
        cache_dir = get_cache_path(self.run_id, "features")
        ensure_dir(cache_dir)

        # 批量特征提取
        extractor = BatchFeatureExtractor(self.run_id, self.logger)
        features_data = extractor.process_batch(
            signal_list=signal_list,
            cut_ids=cut_ids,
            tool_ids=tool_ids,
            wear_labels=wear_labels,
            save_dir=cache_dir,
        )

        self.stats["features_extracted"] = len(features_data)
        self.logger.stat(f"Stage4 完成: 提取特征样本数={self.stats['features_extracted']}")

        return features_data

    def run_stage5_augmentation(self, features_data: List[Dict]) -> Dict:
        """运行Stage5: 数据增强

        Args:
            features_data: Stage4提取的特征数据列表

        Returns:
            增强后的数据集
        """
        self.logger.info("=" * 50)
        self.logger.info("开始 Stage5: 数据增强与样本重构")
        self.logger.info("=" * 50)

        # 准备数据
        features_list = [d["feature_vector"] for d in features_data]
        labels_list = [d["wear_label"] for d in features_data]
        tool_ids = [d["tool_id"] for d in features_data]
        cut_unique_ids = [d["cut_unique_id"] for d in features_data]

        # 数据增强
        augmenter = DataAugmenter(self.run_id)
        augmented_data = augmenter.augment_dataset(
            features_list=features_list,
            labels_list=labels_list,
            tool_ids=tool_ids,
            cut_unique_ids=cut_unique_ids,
            n_augmentations=2,
            balance=True,
        )

        self.stats["samples_augmented"] = len(augmented_data["features"])
        self.logger.stat(f"Stage5 完成: 增强后样本数={self.stats['samples_augmented']}")

        # 保存增强后的数据
        samples_dir = get_cache_path(self.run_id, "samples")
        ensure_dir(samples_dir)
        np.save(os.path.join(samples_dir, "augmented_features.npy"), augmented_data["features"])
        np.save(os.path.join(samples_dir, "augmented_labels.npy"), augmented_data["labels"])

        return augmented_data

    def run_stage6_tfrecord(self, augmented_data: Dict) -> Dict:
        """运行Stage6: TFRecord数据集构建

        Args:
            augmented_data: Stage5增强后的数据集

        Returns:
            数据集元信息
        """
        self.logger.info("=" * 50)
        self.logger.info("开始 Stage6: TFRecord数据集构建")
        self.logger.info("=" * 50)

        features = augmented_data["features"]
        labels = augmented_data["labels"]
        tool_ids = augmented_data["tool_ids"]
        cut_unique_ids = augmented_data["cut_unique_ids"]

        # 创建TFRecord构建器
        output_dir = os.path.join(self.output_dir, "tfrecord")
        builder = TFRecordBuilder(output_dir=output_dir)

        # 构建数据集
        metadata = builder.build(
            features=features,
            labels=labels,
            tool_ids=tool_ids,
            cut_unique_ids=cut_unique_ids,
            version=self.run_id,
        )

        self.stats["train_samples"] = metadata["train_samples"]
        self.stats["test_samples"] = metadata["test_samples"]

        self.logger.stat(f"Stage6 完成: 训练样本={self.stats['train_samples']}, 测试样本={self.stats['test_samples']}")

        return metadata

    def run_all_stages(self) -> Dict:
        """运行完整流水线 (Stage1-6)

        Returns:
            最终统计信息
        """
        self.logger.info("=" * 60)
        self.logger.info(f"开始完整流水线运行: run_id={self.run_id}")
        self.logger.info("=" * 60)

        # Stage 1-3: 流式数据处理（复用 run_pipeline.py）
        preprocessed_data = self.run_pipeline_stages_1_3()
        if not preprocessed_data:
            self.logger.error("流水线在Stage1-3阶段失败，终止运行")
            return self.stats

        # Stage 4: 特征提取
        features_data = self.run_stage4_feature_extraction(preprocessed_data)
        if not features_data:
            self.logger.error("Stage4特征提取失败，终止运行")
            return self.stats

        # Stage 5: 数据增强
        augmented_data = self.run_stage5_augmentation(features_data)
        if not augmented_data or len(augmented_data["features"]) == 0:
            self.logger.error("Stage5数据增强失败，终止运行")
            return self.stats

        # Stage 6: TFRecord构建
        metadata = self.run_stage6_tfrecord(augmented_data)

        self.logger.info("=" * 60)
        self.logger.info("完整流水线运行完成")
        self.logger.info("=" * 60)

        return self.stats

    def print_summary(self, stats: Dict, metadata: Optional[Dict] = None):
        """打印运行摘要

        Args:
            stats: 运行统计信息
            metadata: 可选的元信息
        """
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("运行摘要")
        self.logger.info("=" * 60)
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"数据目录: {self.data_dir}")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info("")
        self.logger.info("处理统计:")
        self.logger.info(f"  Stage1-3 (流式处理): {stats['samples_preprocessed']:>6} 样本")
        self.logger.info(f"  Stage4 (特征提取):   {stats['features_extracted']:>6} 样本")
        self.logger.info(f"  Stage5 (数据增强):   {stats['samples_augmented']:>6} 样本")
        self.logger.info(f"  Stage6 (TFRecord):")
        self.logger.info(f"    - 训练集:          {stats['train_samples']:>6} 样本")
        self.logger.info(f"    - 测试集:          {stats['test_samples']:>6} 样本")
        self.logger.info("")
        self.logger.info("输出文件位置:")
        self.logger.info(f"  {self.output_dir}")
        self.logger.info("=" * 60)


def parse_stage_arg(stage_str: str) -> Tuple[List[int], bool]:
    """解析stage参数

    Args:
        stage_str: stage参数字符串，如 '1', '1-3', 'all'

    Returns:
        (stage_list, is_all) 元组
    """
    stage_str = stage_str.strip().lower()

    if stage_str == "all":
        return list(range(1, 7)), True

    if "-" in stage_str:
        parts = stage_str.split("-")
        if len(parts) != 2:
            raise ValueError(f"无效的stage范围: {stage_str}")
        start = int(parts[0])
        end = int(parts[1])
        if start < 1 or end > 6 or start > end:
            raise ValueError(f"stage范围必须在1-6之间: {stage_str}")
        return list(range(start, end + 1)), False

    stage_num = int(stage_str)
    if stage_num < 1 or stage_num > 6:
        raise ValueError(f"stage必须在1-6之间: {stage_num}")
    return [stage_num], False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="PHM2010 刀具磨损预测数据处理流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py --stage all
  python main.py --stage 1-3 --resume
  python main.py --stage 4
  python main.py --stage 6 --run-id run_001_20250524_1530
        """,
    )

    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        help="要运行的阶段: 1, 2, 3, 4, 5, 6, 1-3, all (默认: all)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="运行ID (不提供则自动生成)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_RAW_DATA_DIR,
        help=f"原始数据目录路径 (默认: {DEFAULT_RAW_DATA_DIR})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从检查点恢复运行（仅Stage1-3有效）",
    )

    args = parser.parse_args()

    # 解析stage参数
    try:
        stages, is_all = parse_stage_arg(args.stage)
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)

    # 生成或使用提供的run_id
    run_id = args.run_id or generate_run_id()

    print(f"PHM2010 数据处理流水线")
    print(f"Stage: {args.stage} -> {stages}")
    print(f"Run ID: {run_id}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Resume: {args.resume}")
    print("")

    # 创建流水线运行器
    runner = PipelineRunner(
        run_id=run_id,
        data_dir=args.data_dir,
        resume=args.resume,
    )

    # 根据stage参数运行
    if is_all:
        # 运行完整流水线
        stats = runner.run_all_stages()
        runner.print_summary(stats)
    else:
        print("注意: 非完整流水线模式可能需要手动管理依赖")
        print("建议使用 --stage all 运行完整流程")
        sys.exit(0)

    print(f"\n运行完成! Run ID: {run_id}")
    print(f"结果保存在: {runner.output_dir}")

    # 关闭日志
    runner.logger.close()


if __name__ == "__main__":
    main()
