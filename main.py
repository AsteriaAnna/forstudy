"""PHM2010 项目主入口

该模块是PHM2010刀具磨损预测项目的主入口点，负责协调各个处理阶段。
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
from pipeline.stage1_loading import Stage1Loader
from pipeline.stage2_validation import Stage2Validator
from pipeline.stage3_preprocessing import Stage3Preprocessor
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

    def run_stage1_loading(self) -> List[Dict]:
        """运行Stage1: 原始数据加载

        Returns:
            加载的切削数据列表
        """
        self.logger.info("=" * 50)
        self.logger.info("开始 Stage1: 原始数据加载")
        self.logger.info("=" * 50)

        loader = Stage1Loader(self.data_dir, self.logger)
        loaded_data = []

        for cutting_data in loader.generate_cut_data():
            self.monitor.increment_loaded()
            self.monitor.increment_success()
            self.stats["samples_loaded"] += 1
            loaded_data.append(cutting_data)

        self.logger.stat(f"Stage1 完成: 加载样本数={self.stats['samples_loaded']}")
        return loaded_data

    def run_stage2_validation(self, loaded_data: List[Dict]) -> List[Dict]:
        """运行Stage2: 数据质量验证

        Args:
            loaded_data: Stage1加载的数据列表

        Returns:
            验证通过的数据列表
        """
        self.logger.info("=" * 50)
        self.logger.info("开始 Stage2: 数据质量验证")
        self.logger.info("=" * 50)

        validator = Stage2Validator()
        validated_data = []

        for cutting_data in loaded_data:
            valid_data, error_info = validator.validate(cutting_data)
            if valid_data is not None:
                valid_data["status"] = "VALID"
                validated_data.append(valid_data)
                self.stats["samples_validated"] += 1
            else:
                self.logger.warn(f"数据验证失败: {cutting_data.get('cut_unique_id', 'unknown')}")

        self.logger.stat(f"Stage2 完成: 验证通过样本数={self.stats['samples_validated']}")
        return validated_data

    def run_stage3_preprocessing(self, validated_data: List[Dict]) -> List[Dict]:
        """运行Stage3: 信号预处理

        Args:
            validated_data: Stage2验证通过的数据列表

        Returns:
            预处理后的数据列表
        """
        self.logger.info("=" * 50)
        self.logger.info("开始 Stage3: 信号预处理")
        self.logger.info("=" * 50)

        preprocessor = Stage3Preprocessor(self.run_id)
        preprocessed_data = []

        for cutting_data in validated_data:
            result = preprocessor.preprocess(cutting_data)
            if result is not None:
                preprocessed_data.append(result)
                self.stats["samples_preprocessed"] += 1

                # 保存预处理后的信号到缓存
                cache_dir = get_cache_path(self.run_id, "stage3")
                signal_path = os.path.join(cache_dir, f"{result['cut_unique_id']}.npy")
                np.save(signal_path, result["processed_signal"])

        self.logger.stat(f"Stage3 完成: 预处理样本数={self.stats['samples_preprocessed']}")
        return preprocessed_data

    def run_pipeline_stages_1_3(self) -> List[Dict]:
        """运行Stage1-3完整流水线

        Returns:
            预处理后的数据列表
        """
        # Stage 1: 加载
        loaded_data = self.run_stage1_loading()
        if not loaded_data:
            self.logger.error("Stage1 加载失败，无数据")
            return []

        # Stage 2: 验证
        validated_data = self.run_stage2_validation(loaded_data)
        if not validated_data:
            self.logger.error("Stage2 验证失败，无有效数据")
            return []

        # Stage 3: 预处理
        preprocessed_data = self.run_stage3_preprocessing(validated_data)
        if not preprocessed_data:
            self.logger.error("Stage3 预处理失败，无有效数据")
            return []

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

        # Stage 1-3: 数据处理流水线
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
        self.logger.info(f"  Stage1 (加载):       {stats['samples_loaded']:>6} 样本")
        self.logger.info(f"  Stage2 (验证):       {stats['samples_validated']:>6} 样本")
        self.logger.info(f"  Stage3 (预处理):     {stats['samples_preprocessed']:>6} 样本")
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
        help="从检查点恢复运行",
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
        # 运行指定的单个或多个阶段
        # 注意: 这里需要按顺序依赖关系运行
        preprocessed_data = None
        features_data = None
        augmented_data = None
        metadata = None

        for stage in stages:
            if stage == 1:
                loaded_data = runner.run_stage1_loading()
            elif stage == 2:
                if preprocessed_data is None and runner.stats["samples_loaded"] > 0:
                    # 需要先运行stage1
                    loaded_data = runner.run_stage1_loading()
                    validated_data = runner.run_stage2_validation(loaded_data)
                    preprocessed_data = validated_data
                else:
                    preprocessed_data = runner.run_stage2_validation(
                        runner.run_stage1_loading() if runner.stats["samples_loaded"] == 0 else []
                    )
            elif stage == 3:
                if preprocessed_data is None:
                    # 需要先运行stage1-2
                    loaded_data = runner.run_stage1_loading()
                    validated_data = runner.run_stage2_validation(loaded_data)
                    preprocessed_data = runner.run_stage3_preprocessing(validated_data)
                else:
                    preprocessed_data = runner.run_stage3_preprocessing(preprocessed_data)
            elif stage == 4:
                if preprocessed_data is None:
                    print("错误: Stage4需要先运行Stage1-3")
                    sys.exit(1)
                features_data = runner.run_stage4_feature_extraction(preprocessed_data)
            elif stage == 5:
                if features_data is None:
                    print("错误: Stage5需要先运行Stage4")
                    sys.exit(1)
                augmented_data = runner.run_stage5_augmentation(features_data)
            elif stage == 6:
                if augmented_data is None:
                    print("错误: Stage6需要先运行Stage5")
                    sys.exit(1)
                metadata = runner.run_stage6_tfrecord(augmented_data)

        # 打印统计信息
        if 1 in stages:
            print(f"Stage1: 加载 {runner.stats['samples_loaded']} 样本")
        if 2 in stages:
            print(f"Stage2: 验证 {runner.stats['samples_validated']} 样本")
        if 3 in stages:
            print(f"Stage3: 预处理 {runner.stats['samples_preprocessed']} 样本")
        if 4 in stages:
            print(f"Stage4: 提取 {runner.stats['features_extracted']} 特征")
        if 5 in stages:
            print(f"Stage5: 增强 {runner.stats['samples_augmented']} 样本")
        if 6 in stages:
            print(f"Stage6: 构建 TFRecord (训练: {runner.stats['train_samples']}, 测试: {runner.stats['test_samples']})")

    print(f"\n运行完成! Run ID: {run_id}")
    print(f"结果保存在: {runner.output_dir}")

    # 关闭日志
    runner.logger.close()


if __name__ == "__main__":
    main()
