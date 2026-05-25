"""Stage1: 逐切削流式原始数据加载模块

PHM2010数据集的原始数据加载器，以生成器方式逐切削加载信号数据和磨损标签。
"""

import os
import re
from pathlib import Path
from typing import Generator, Optional

import numpy as np

from core.logger import Logger


class Stage1Loader:
    """PHM2010原始数据流式加载器"""

    SUPPORTED_TOOL_IDS = ['c1', 'c4', 'c6']

    # 文件名模式: c_1_001.csv -> tool_id=c1, cut_num=1
    # 文件名模式: c_c1_1.csv -> tool_id=c1, cut_num=1
    PATTERN_UNDERSCORE = re.compile(r'^c_(\d+)_(\d+)\.csv$')
    PATTERN_PREFIX = re.compile(r'^c_c(\d+)_(\d+)\.csv$')

    def __init__(self, raw_data_dir: str, logger: Optional[Logger] = None):
        """初始化Stage1加载器

        Args:
            raw_data_dir: phm2010_raw文件夹路径
            logger: 日志记录器实例
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.logger = logger or Logger("stage1_default")
        self._wear_cache: dict[str, dict[int, dict[str, float]]] = {}

    def _get_tool_signal_dir(self, tool_id: str) -> Path:
        """获取刀具信号文件目录

        Args:
            tool_id: 刀具ID (c1/c4/c6)

        Returns:
            信号文件目录路径
        """
        return self.raw_data_dir / tool_id / tool_id

    def _get_wear_label_path(self, tool_id: str) -> Path:
        """获取磨损标签文件路径

        Args:
            tool_id: 刀具ID (c1/c4/c6)

        Returns:
            磨损标签文件路径
        """
        return self.raw_data_dir / tool_id / f"{tool_id}_wear.csv"

    def _calculate_robust_wear(self, flute_values: list) -> float:
        values = np.array(flute_values, dtype=np.float64)
        mean_val = np.mean(values)
        std_val = np.std(values)

        if std_val == 0:
            return mean_val

        confidence = 1 - np.abs(values - mean_val) / (3 * std_val)
        confidence = np.clip(confidence, 0.1, 1.0)
        weights = confidence / np.sum(confidence)
        robust_wear = np.sum(values * weights)

        return float(robust_wear)

    def _compute_wear_stages(self, tool_id: str, wear_data: dict) -> dict:
        from sklearn.mixture import GaussianMixture

        sorted_cuts = sorted(wear_data.keys())

        wear_values = np.array([
            wear_data[cut]['robust_wear']
            for cut in sorted_cuts
        ], dtype=np.float64).reshape(-1, 1)

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

        means = gmm.means_.flatten()
        stage_order = np.argsort(means)
        stage_names = ['initial', 'normal', 'severe']

        result = {}
        for cut_num, label in zip(sorted_cuts, labels):
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
        THRESHOLD_LOW = 50.0
        THRESHOLD_HIGH = 150.0

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

    def load_wear_labels(self, tool_id: str) -> dict[int, dict[str, float]]:
        """加载指定刀具的磨损标签

        Args:
            tool_id: 刀具ID (c1/c4/c6)

        Returns:
            {cut_num: {flute_1, flute_2, flute_3, robust_wear, wear_stage}}
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

                        robust_wear = self._calculate_robust_wear([flute_1, flute_2, flute_3])

                        wear_data[cut_num] = {
                            'flute_1': flute_1,
                            'flute_2': flute_2,
                            'flute_3': flute_3,
                            'robust_wear': robust_wear
                        }
                    except (ValueError, IndexError):
                        continue

            if len(wear_data) > 0:
                stage_mapping = self._compute_wear_stages(tool_id, wear_data)
                for cut_num, stage in stage_mapping.items():
                    if cut_num in wear_data:
                        wear_data[cut_num]['wear_stage'] = stage

            self._wear_cache[tool_id] = wear_data
            self.logger.info(f"Loaded {len(wear_data)} wear labels for {tool_id}", tool_id)

        except Exception as e:
            self.logger.error(f"Failed to load wear labels: {wear_path}", tool_id, e)
            return {}

        return wear_data

    def _parse_filename(self, filename: str) -> tuple[Optional[str], Optional[int]]:
        """解析信号文件名，提取tool_id和cut_num

        Args:
            filename: 文件名 (不含路径)

        Returns:
            (tool_id, cut_num) 或 (None, None)
        """
        # Pattern: c_1_001.csv -> tool_id=c1, cut_num=1
        match = self.PATTERN_UNDERSCORE.match(filename)
        if match:
            tool_num = match.group(1)
            cut_num = int(match.group(2))
            return f'c{tool_num}', cut_num

        # Pattern: c_c1_1.csv -> tool_id=c1, cut_num=1
        match = self.PATTERN_PREFIX.match(filename)
        if match:
            tool_num = match.group(1)
            cut_num = int(match.group(2))
            return f'c{tool_num}', cut_num

        return None, None

    def scan_signal_files(self, tool_id: str) -> list[tuple[str, int]]:
        """扫描指定刀具的所有信号文件

        Args:
            tool_id: 刀具ID (c1/c4/c6)

        Returns:
            [(filename, cut_num), ...] 按cut_num排序
        """
        signal_dir = self._get_tool_signal_dir(tool_id)

        if not signal_dir.exists():
            self.logger.error(f"Signal directory not found: {signal_dir}", tool_id)
            return []

        files: list[tuple[str, int]] = []

        try:
            for filename in os.listdir(signal_dir):
                if not filename.endswith('.csv'):
                    continue

                parsed_tool_id, cut_num = self._parse_filename(filename)
                if parsed_tool_id == tool_id and cut_num is not None:
                    files.append((filename, cut_num))

            files.sort(key=lambda x: x[1])
            self.logger.info(f"Found {len(files)} signal files for {tool_id}", tool_id)

        except Exception as e:
            self.logger.error(f"Failed to scan signal files: {signal_dir}", tool_id, e)

        return files

    def load_single_cutting(
        self,
        filepath: str | Path,
        tool_id: str,
        cut_num: int
    ) -> Optional[dict]:
        """加载单个切削的信号数据

        Args:
            filepath: 信号文件路径
            tool_id: 刀具ID
            cut_num: 切削编号

        Returns:
            切削数据字典或None(加载失败时)
        """
        filepath = Path(filepath)

        if not filepath.exists():
            self.logger.error(f"File not found: {filepath}", f"{tool_id}_cut{cut_num}")
            return None

        try:
            data = np.loadtxt(filepath, delimiter=',')
        except Exception as e:
            self.logger.error(f"Failed to load signal data: {filepath}", f"{tool_id}_cut{cut_num}", e)
            return None

        if data.ndim != 2 or data.shape[1] != 7:
            self.logger.error(
                f"Invalid signal dimension: {data.shape}, expected (N, 7)",
                f"{tool_id}_cut{cut_num}"
            )
            return None

        if data.size == 0:
            self.logger.error(f"Empty signal file: {filepath}", f"{tool_id}_cut{cut_num}")
            return None

        return {
            'signal': data,
            'tool_id': tool_id,
            'cut_num': cut_num,
            'cut_unique_id': f'{tool_id}_cut{cut_num}'
        }

    def generate_cut_data(
        self,
        tool_ids: Optional[list[str]] = None
    ) -> Generator[dict, None, None]:
        """生成器: 流式加载所有切削数据

        Args:
            tool_ids: 要加载的刀具ID列表，默认全部

        Yields:
            cutting_data字典，包含:
            - signal: np.ndarray (N, 7)
            - tool_id: str
            - cut_unique_id: str
            - wear_label: {flute_1, flute_2, flute_3}
            - status: 'LOADED'
        """
        if tool_ids is None:
            tool_ids = self.SUPPORTED_TOOL_IDS

        total_loaded = 0
        total_skipped = 0

        for tool_id in tool_ids:
            if tool_id not in self.SUPPORTED_TOOL_IDS:
                self.logger.warn(f"Unsupported tool_id: {tool_id}")
                continue

            wear_labels = self.load_wear_labels(tool_id)
            signal_files = self.scan_signal_files(tool_id)

            for filename, cut_num in signal_files:
                filepath = self._get_tool_signal_dir(tool_id) / filename

                cutting_data = self.load_single_cutting(filepath, tool_id, cut_num)

                if cutting_data is None:
                    total_skipped += 1
                    continue

                if cut_num not in wear_labels:
                    self.logger.warn(
                        f"Wear label missing for cut {cut_num}",
                        f"{tool_id}_cut{cut_num}"
                    )
                    total_skipped += 1
                    continue

                cutting_data['wear_label'] = wear_labels[cut_num]
                cutting_data['status'] = 'LOADED'

                total_loaded += 1
                yield cutting_data

        self.logger.stat(
            f"Stage1 Summary - Loaded: {total_loaded}, Skipped: {total_skipped}"
        )

    def get_total_counts(self, tool_ids: Optional[list[str]] = None) -> dict[str, int]:
        """获取各刀具的信号文件数量

        Args:
            tool_ids: 刀具ID列表，默认全部

        Returns:
            {tool_id: file_count}
        """
        if tool_ids is None:
            tool_ids = self.SUPPORTED_TOOL_IDS

        counts = {}
        for tool_id in tool_ids:
            files = self.scan_signal_files(tool_id)
            counts[tool_id] = len(files)

        return counts


if __name__ == '__main__':
    run_id = Logger.generate_run_id(1)
    logger = Logger(run_id)

    raw_dir = "phm2010_raw"
    loader = Stage1Loader(raw_dir, logger)

    logger.info(f"Total signal files: {loader.get_total_counts()}")

    for i, cutting_data in enumerate(loader.generate_cut_data()):
        logger.info(
            f"Loaded: {cutting_data['cut_unique_id']}, "
            f"signal shape: {cutting_data['signal'].shape}, "
            f"wear: {cutting_data['wear_label']}"
        )

        if i >= 4:
            break

    logger.info("Stage1 loading test completed")
    logger.close()
