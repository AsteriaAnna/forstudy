"""Pipeline集成模块: Stage1-4完整流程

该模块整合Stage1Loader、Stage2Validator、Stage3Preprocessor和Stage4FeatureExtractor，
实现完整的从原始数据到特征提取的pipeline流程。

特性:
- Stage1-3流式处理（低内存占用）
- Stage4批量处理（支持归一化和特征选择）
- 断点续传支持
- 错误处理与跳过
- 进度跟踪
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from core.logger import Logger
from core.monitor import GlobalMonitor
from core.utils import get_interim_dir
from pipeline.stage1_loading import Stage1Loader
from pipeline.stage2_validation import Stage2Validator
from pipeline.stage3_preprocessing import Stage3Preprocessor
from modules.stage4_feature_extract.feature_extractor import BatchFeatureExtractor


class DataPipeline:
    """PHM2010数据处理Pipeline整合器

    执行流程:
    - Stage1-3: 流式处理（低内存占用）
    - Stage4: 批量处理（特征提取→归一化→特征选择）
    支持断点续传、流式处理和错误恢复。
    """

    def __init__(
        self,
        raw_data_dir: str,
        run_id: str,
        output_dir: str = "results",
        tool_ids: Optional[list[str]] = None,
        resume: bool = True,
        normalization_method: Optional[str] = None,
        feature_selector=None
    ):
        """初始化Pipeline

        Args:
            raw_data_dir: 原始数据根目录
            run_id: 运行唯一标识符
            output_dir: 输出根目录
            tool_ids: 要处理的刀具ID列表，None表示全部
            resume: 是否支持断点续传
            normalization_method: 特征归一化方法，'standard'或'minmax'，None表示不进行归一化
            feature_selector: 特征选择器实例（如CVOCA），None表示不进行特征选择
        """
        self.raw_data_dir = raw_data_dir
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.tool_ids = tool_ids
        self.resume = resume
        self.normalization_method = normalization_method
        self.feature_selector = feature_selector

        # 创建运行目录
        self.run_dir = self.output_dir / run_id
        self.data_dir = self.run_dir / "data"
        self.log_dir = self.run_dir / "logs"
        self.checkpoint_file = self.run_dir / "checkpoint.json"

        # 特征输出目录
        self.features_dir = Path(get_interim_dir("features")) / run_id

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)

        # 初始化Logger
        self.logger = Logger(run_id, log_dir=str(self.log_dir))

        # 初始化全局监控器
        self.monitor = GlobalMonitor()
        self.monitor.reset()

        # 初始化四个Stage
        self.stage1 = Stage1Loader(raw_data_dir, self.logger)
        self.stage2 = Stage2Validator()
        self.stage3 = Stage3Preprocessor(run_id)
        self.stage4 = BatchFeatureExtractor(
            run_id, 
            self.logger,
            normalization_method=normalization_method,
            feature_selector=feature_selector
        )

        # 统计信息
        self.stats = {
            'total_processed': 0,
            'success_count': 0,
            'filtered_count': 0,
            'filtered_by_type': {},
            'skipped_count': 0,
            'processing_time': 0.0,
            'start_time': None,
            'end_time': None,
            'stage4_features_count': 0
        }

        # 加载checkpoint
        self.processed_ids: set = set()
        if self.resume:
            self._load_checkpoint()

    def _load_checkpoint(self) -> bool:
        """加载checkpoint文件，恢复已处理的cut_unique_id列表

        Returns:
            是否成功加载checkpoint
        """
        if not self.checkpoint_file.exists():
            self.logger.info(f"未找到checkpoint文件，将从头开始处理")
            return False

        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            self.processed_ids = set(checkpoint_data.get('processed_ids', []))
            stats_data = checkpoint_data.get('stats', {})

            # 恢复统计信息
            self.stats['success_count'] = stats_data.get('success_count', 0)
            self.stats['filtered_count'] = stats_data.get('filtered_count', 0)
            self.stats['filtered_by_type'] = stats_data.get('filtered_by_type', {})
            self.stats['skipped_count'] = stats_data.get('skipped_count', 0)

            self.logger.info(
                f"Checkpoint加载成功: 已处理 {len(self.processed_ids)} 个切削，"
                f"成功 {self.stats['success_count']}，过滤 {self.stats['filtered_count']}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Checkpoint加载失败: {e}")
            return False

    def _save_checkpoint(self):
        """保存checkpoint文件"""
        try:
            checkpoint_data = {
                'processed_ids': list(self.processed_ids),
                'stats': {
                    'success_count': self.stats['success_count'],
                    'filtered_count': self.stats['filtered_count'],
                    'filtered_by_type': self.stats['filtered_by_type'],
                    'skipped_count': self.stats['skipped_count']
                }
            }

            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Checkpoint已保存: {len(self.processed_ids)} 个已处理ID")

        except Exception as e:
            self.logger.error(f"Checkpoint保存失败: {e}")

    def _save_processed_data(self, preprocessed_data: dict) -> bool:
        """保存预处理后的数据到缓存

        Args:
            preprocessed_data: 预处理后的数据字典

        Returns:
            是否保存成功
        """
        cut_unique_id = preprocessed_data.get('cut_unique_id')
        if not cut_unique_id:
            return False

        output_path = self.data_dir / f"{cut_unique_id}.npz"

        try:
            np.savez(
                output_path,
                processed_signal=preprocessed_data['processed_signal'],
                tool_id=preprocessed_data['tool_id'],
                wear_label_flute_1=preprocessed_data['wear_label']['flute_1'],
                wear_label_flute_2=preprocessed_data['wear_label']['flute_2'],
                wear_label_flute_3=preprocessed_data['wear_label']['flute_3'],
                wear_label_robust_wear=preprocessed_data['wear_label']['robust_wear'],
                wear_label_wear_stage=preprocessed_data['wear_label']['wear_stage'],
                status=preprocessed_data.get('status', 'PROCESSED')
            )
            return True

        except Exception as e:
            self.logger.error(f"数据保存失败: {cut_unique_id}", exc=e)
            return False

    def _process_single_cutting(self, cut_data: dict) -> bool:
        """处理单个切削数据，执行Stage1-3完整流程

        Args:
            cut_data: Stage1加载的切削数据

        Returns:
            处理是否成功
        """
        cut_unique_id = cut_data.get('cut_unique_id', 'unknown')

        # Stage2: 验证
        valid_data, error_info = self.stage2.validate(cut_data)

        if error_info:
            self.stats['filtered_count'] += 1

            # 分类统计过滤原因
            for error_type in error_info.keys():
                if error_type not in self.stats['filtered_by_type']:
                    self.stats['filtered_by_type'][error_type] = 0
                self.stats['filtered_by_type'][error_type] += 1

            self.logger.warn(
                f"Stage2验证失败 [{error_type}]: {cut_unique_id}",
                cut_unique_id
            )
            return False

        # Stage3: 预处理
        preprocessed_data = self.stage3.preprocess(valid_data)

        if preprocessed_data is None:
            self.stats['filtered_count'] += 1
            if 'preprocess_error' not in self.stats['filtered_by_type']:
                self.stats['filtered_by_type']['preprocess_error'] = 0
            self.stats['filtered_by_type']['preprocess_error'] += 1
            self.logger.error(f"Stage3预处理失败: {cut_unique_id}", cut_unique_id)
            return False

        # 保存到缓存
        if self._save_processed_data(preprocessed_data):
            self.stats['success_count'] += 1
            self.processed_ids.add(cut_unique_id)
            self.logger.info(f"处理成功: {cut_unique_id}", cut_unique_id)
            return True
        else:
            return False

    def run(self) -> dict:
        """执行完整的Pipeline流程

        Returns:
            处理统计信息字典
        """
        self.stats['start_time'] = time.time()
        self.logger.info(f"=" * 60)
        self.logger.info(f"Pipeline启动: run_id={self.run_id}")
        self.logger.info(f"原始数据目录: {self.raw_data_dir}")
        self.logger.info(f"输出目录: {self.run_dir}")
        self.logger.info(f"断点续传: {'启用' if self.resume else '禁用'}")
        self.logger.info(f"=" * 60)

        # 获取总切削数量
        total_counts = self.stage1.get_total_counts(self.tool_ids)
        total_expected = sum(total_counts.values())
        self.logger.info(f"待处理总切削数: {total_expected} (各刀具: {total_counts})")

        processed_count = 0
        resume_skipped = 0

        # 流式处理每个切削
        for cut_data in self.stage1.generate_cut_data(self.tool_ids):
            cut_unique_id = cut_data.get('cut_unique_id')

            self.stats['total_processed'] += 1
            processed_count += 1

            # 断点续传: 跳过已处理的切削
            if cut_unique_id in self.processed_ids:
                self.stats['skipped_count'] += 1
                resume_skipped += 1
                self.logger.info(f"跳过已处理: {cut_unique_id}", cut_unique_id)
                continue

            # 执行Stage1-3处理
            try:
                self._process_single_cutting(cut_data)
            except Exception as e:
                self.logger.error(f"处理异常: {cut_unique_id}", cut_unique_id, e)
                self.stats['filtered_count'] += 1
                if 'process_exception' not in self.stats['filtered_by_type']:
                    self.stats['filtered_by_type']['process_exception'] = 0
                self.stats['filtered_by_type']['process_exception'] += 1

            # 每处理10个或每100个保存一次checkpoint
            if processed_count % 10 == 0:
                self._save_checkpoint()
                self._print_progress(processed_count, total_expected, resume_skipped)

        # 最终保存checkpoint
        self._save_checkpoint()

        # Stage4: 批量特征处理（特征提取→归一化→特征选择）
        self._run_stage4()

        self.stats['end_time'] = time.time()
        self.stats['processing_time'] = self.stats['end_time'] - self.stats['start_time']

        # 打印最终统计
        self._print_final_stats()

        return self.stats

    def _run_stage4(self):
        """执行Stage4批量特征处理"""
        self.logger.info(f"=" * 60)
        self.logger.info(f"Stage4: 批量特征处理开始")
        self.logger.info(f"=" * 60)

        if self.stats['success_count'] == 0:
            self.logger.warn("没有有效的预处理数据，跳过Stage4")
            return

        # 加载所有预处理数据
        signal_list, cut_ids, tool_ids, wear_labels = self._load_preprocessed_data()

        if len(signal_list) == 0:
            self.logger.warn("未找到预处理数据文件")
            return

        self.logger.info(f"加载预处理数据完成: {len(signal_list)} 个样本")

        # 执行批量特征处理
        try:
            results = self.stage4.process_batch(
                signal_list=signal_list,
                cut_ids=cut_ids,
                tool_ids=tool_ids,
                wear_labels=wear_labels,
                save_dir=str(self.features_dir)
            )
            self.stats['stage4_features_count'] = len(results)
            self.logger.info(f"Stage4处理完成: 成功提取 {len(results)} 个特征向量")
        except Exception as e:
            self.logger.error(f"Stage4处理失败", exc=e)

    def _load_preprocessed_data(self):
        """加载所有预处理数据"""
        signal_list = []
        cut_ids = []
        tool_ids = []
        wear_labels = []

        for cut_unique_id in self.processed_ids:
            file_path = self.data_dir / f"{cut_unique_id}.npz"
            if not file_path.exists():
                continue

            try:
                data = np.load(file_path, allow_pickle=True)
                signal = data['processed_signal']
                tool_id = str(data['tool_id'])
                
                # 重建磨损标签字典
                wear_label = {
                    'flute_1': float(data['wear_label_flute_1']),
                    'flute_2': float(data['wear_label_flute_2']),
                    'flute_3': float(data['wear_label_flute_3']),
                    'robust_wear': float(data['wear_label_robust_wear']),
                    'wear_stage': str(data['wear_label_wear_stage'])
                }

                signal_list.append(signal)
                cut_ids.append(cut_unique_id)
                tool_ids.append(tool_id)
                wear_labels.append(wear_label)
            except Exception as e:
                self.logger.warn(f"加载预处理数据失败: {cut_unique_id}", exc=e)

        return signal_list, cut_ids, tool_ids, wear_labels

    def _print_progress(self, current: int, total: int, skipped: int):
        """打印进度信息

        Args:
            current: 当前已处理数量
            total: 总数量
            skipped: 因断点续传跳过的数量
        """
        elapsed = time.time() - self.stats['start_time']
        rate = current / elapsed if elapsed > 0 else 0
        remaining = (total - current) / rate if rate > 0 else 0

        self.logger.info(
            f"进度: {current}/{total} ({100*current/total:.1f}%) | "
            f"成功: {self.stats['success_count']} | "
            f"过滤: {self.stats['filtered_count']} | "
            f"跳过: {skipped} | "
            f"速率: {rate:.1f}个/秒 | "
            f"剩余: {remaining:.0f}秒"
        )

    def _print_final_stats(self):
        """打印最终统计信息"""
        self.logger.info(f"=" * 60)
        self.logger.info(f"Pipeline执行完成")
        self.logger.info(f"=" * 60)
        self.logger.stat(f"总处理数: {self.stats['total_processed']}")
        self.logger.stat(f"成功处理: {self.stats['success_count']}")
        self.logger.stat(f"过滤数量: {self.stats['filtered_count']}")
        self.logger.stat(f"跳过数量: {self.stats['skipped_count']}")
        self.logger.stat(f"提取特征数: {self.stats['stage4_features_count']}")
        self.logger.stat(f"处理时间: {self.stats['processing_time']:.2f}秒")

        if self.stats['filtered_by_type']:
            self.logger.info(f"过滤原因分布:")
            for error_type, count in sorted(self.stats['filtered_by_type'].items()):
                self.logger.stat(f"  - {error_type}: {count}")

        success_rate = (
            self.stats['success_count'] / self.stats['total_processed'] * 100
            if self.stats['total_processed'] > 0 else 0
        )
        self.logger.stat(f"成功率: {success_rate:.2f}%")
        self.logger.info(f"=" * 60)
        self.logger.info(f"输出目录: {self.run_dir}")
        self.logger.info(f"数据目录: {self.data_dir}")
        self.logger.info(f"特征目录: {self.features_dir}")
        self.logger.info(f"日志目录: {self.log_dir}")
        self.logger.info(f"=" * 60)

    def get_stats(self) -> dict:
        """获取当前统计信息

        Returns:
            统计信息字典
        """
        return dict(self.stats)

    def get_processed_ids(self) -> list:
        """获取已处理的cut_unique_id列表

        Returns:
            已处理ID列表
        """
        return list(self.processed_ids)


def main():
    """Pipeline命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='PHM2010数据处理Pipeline')
    parser.add_argument(
        '--raw-data-dir',
        type=str,
        default='phm2010_raw',
        help='原始数据目录路径'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='运行ID，默认为自动生成'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='输出目录路径'
    )
    parser.add_argument(
        '--tool-ids',
        type=str,
        nargs='+',
        default=None,
        help='要处理的刀具ID列表，如 c1 c4 c6'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='禁用断点续传，从头开始处理'
    )
    parser.add_argument(
        '--normalization',
        type=str,
        default=None,
        choices=['standard', 'minmax'],
        help='特征归一化方法: standard(Z-score) 或 minmax(0-1归一化)'
    )

    args = parser.parse_args()

    # 生成run_id
    run_id = args.run_id or Logger.generate_run_id(1)

    # 创建并运行Pipeline
    pipeline = DataPipeline(
        raw_data_dir=args.raw_data_dir,
        run_id=run_id,
        output_dir=args.output_dir,
        tool_ids=args.tool_ids,
        resume=not args.no_resume,
        normalization_method=args.normalization
    )

    stats = pipeline.run()

    return stats


if __name__ == '__main__':
    main()
