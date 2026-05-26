"""Stage6: TFRecord dataset builder for PHM2010 project.

This module handles dataset splitting and TFRecord format conversion
for the PHM2010 tool wear prediction project.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import tensorflow as tf


class TFRecordBuilder:
    """Builder class for creating TFRecord datasets from augmented features.

    Handles dataset splitting by tool_id (c1+c4 for train, c6 for test),
    TFRecord serialization, and metadata generation.
    """

    TRAIN_TOOLS = ["c1", "c4"]
    TEST_TOOLS = ["c6"]
    FEATURE_DIM = 175
    WEAR_LABEL_DIM = 3

    def __init__(self, output_dir: str = None):
        """Initialize TFRecordBuilder.

        Args:
            output_dir: Directory for output files. Defaults to data/processed/
        """
        if output_dir is None:
            from core.utils import get_processed_data_dir
            output_dir = get_processed_data_dir()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def split_dataset(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        tool_ids: List[str]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Split dataset by tool_id to avoid data leakage.

        Args:
            features: Feature array of shape (n_samples, 168)
            labels: Wear labels array of shape (n_samples, 3)
            tool_ids: List of tool_id strings

        Returns:
            Tuple of (train_dict, test_dict), each containing:
                - features: np.ndarray
                - labels: np.ndarray
                - tool_ids: List[str]
                - cut_unique_ids: List[str]
        """
        train_indices = []
        test_indices = []

        for idx, tool_id in enumerate(tool_ids):
            if tool_id in self.TRAIN_TOOLS:
                train_indices.append(idx)
            elif tool_id in self.TEST_TOOLS:
                test_indices.append(idx)

        train_features = features[train_indices]
        train_labels = labels[train_indices]
        train_tool_ids = [tool_ids[i] for i in train_indices]

        test_features = features[test_indices]
        test_labels = labels[test_indices]
        test_tool_ids = [tool_ids[i] for i in test_indices]

        train_dict = {
            "features": train_features,
            "labels": train_labels,
            "tool_ids": train_tool_ids
        }

        test_dict = {
            "features": test_features,
            "labels": test_labels,
            "tool_ids": test_tool_ids
        }

        return train_dict, test_dict

    def _create_feature_spec(self) -> dict:
        """Create TFRecord feature specification.

        Returns:
            Dictionary mapping feature names to TFFeature types
        """
        return {
            "features": tf.io.FixedLenFeature(
                [self.FEATURE_DIM], dtype=tf.float32
            ),
            "tool_id": tf.io.FixedLenFeature([], dtype=tf.string),
            "wear_label": tf.io.FixedLenFeature(
                [self.WEAR_LABEL_DIM], dtype=tf.float32
            ),
            "cut_unique_id": tf.io.FixedLenFeature([], dtype=tf.string)
        }

    def _serialize_example(
        self,
        features: np.ndarray,
        tool_id: str,
        wear_label: np.ndarray,
        cut_unique_id: str
    ) -> tf.train.Example:
        """Serialize a single example to TFRecord format.

        Args:
            features: Feature vector (168 dims)
            tool_id: Tool identifier string
            wear_label: Wear label array (3 values for 3 flutes)
            cut_unique_id: Unique cut identifier string

        Returns:
            Serialized tf.train.Example object
        """
        feature = {
            "features": tf.train.Feature(
                float_list=tf.train.FloatList(value=features.tolist())
            ),
            "tool_id": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[tool_id.encode("utf-8")])
            ),
            "wear_label": tf.train.Feature(
                float_list=tf.train.FloatList(value=wear_label.tolist())
            ),
            "cut_unique_id": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[cut_unique_id.encode("utf-8")])
            )
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def create_tfrecord(
        self,
        samples: Dict[str, Any],
        output_path: str,
        cut_unique_ids: List[str] = None
    ) -> int:
        """Create TFRecord file from sample data.

        Args:
            samples: Dictionary containing:
                - features: np.ndarray of shape (n_samples, 168)
                - labels: np.ndarray of shape (n_samples, 3)
                - tool_ids: List of tool_id strings
            output_path: Path to output TFRecord file
            cut_unique_ids: Optional list of cut_unique_ids. If None,
                           will be generated from tool_ids

        Returns:
            Number of samples written to TFRecord
        """
        features = samples["features"]
        labels = samples["labels"]
        tool_ids = samples["tool_ids"]

        if cut_unique_ids is None:
            cut_unique_ids = [f"{tid}_cut{idx}" for idx, tid in enumerate(tool_ids)]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tf.io.TFRecordWriter(str(output_path)) as writer:
            for i in range(len(features)):
                example = self._serialize_example(
                    features=features[i],
                    tool_id=tool_ids[i],
                    wear_label=labels[i],
                    cut_unique_id=cut_unique_ids[i]
                )
                writer.write(example.SerializeToString())

        return len(features)

    def read_tfrecord(self, input_path: str) -> tf.data.Dataset:
        """Read TFRecord file and return as TensorFlow dataset.

        Args:
            input_path: Path to input TFRecord file

        Returns:
            tf.data.Dataset containing parsed examples
        """
        feature_spec = self._create_feature_spec()

        def parse_function(example_proto):
            return tf.io.parse_single_example(example_proto, feature_spec)

        dataset = tf.data.TFRecordDataset(input_path)
        dataset = dataset.map(parse_function)
        return dataset

    def read_tfrecord_as_arrays(
        self,
        input_path: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Read TFRecord file and return as numpy arrays.

        Args:
            input_path: Path to input TFRecord file

        Returns:
            Tuple of:
                - features: np.ndarray (n_samples, 168)
                - labels: np.ndarray (n_samples, 3)
                - tool_ids: List[str]
                - cut_unique_ids: List[str]
        """
        dataset = self.read_tfrecord(input_path)

        features_list = []
        labels_list = []
        tool_ids_list = []
        cut_unique_ids_list = []

        for example in dataset:
            features_list.append(example["features"].numpy())
            labels_list.append(example["wear_label"].numpy())
            tool_ids_list.append(example["tool_id"].numpy().decode("utf-8"))
            cut_unique_ids_list.append(
                example["cut_unique_id"].numpy().decode("utf-8")
            )

        features = np.array(features_list)
        labels = np.array(labels_list)

        return features, labels, tool_ids_list, cut_unique_ids_list

    def generate_metadata(
        self,
        train_count: int,
        test_count: int,
        version: str = None
    ) -> Dict[str, Any]:
        """Generate metadata dictionary for the dataset.

        Args:
            train_count: Number of training samples
            test_count: Number of test samples
            version: Version string. If None, generated from current time

        Returns:
            Metadata dictionary
        """
        if version is None:
            version = self._generate_version()

        now = datetime.now()
        created_at = now.strftime("%Y-%m-%d %H:%M:%S")

        total = train_count + test_count
        train_ratio = (train_count / total * 100) if total > 0 else 0
        test_ratio = (test_count / total * 100) if total > 0 else 0
        split_ratio = f"{int(train_ratio)}:{int(test_ratio)}"

        metadata = {
            "version": version,
            "created_at": created_at,
            "train_samples": train_count,
            "test_samples": test_count,
            "feature_dim": self.FEATURE_DIM,
            "split_ratio": split_ratio,
            "split_method": "by_tool_id",
            "train_tools": self.TRAIN_TOOLS,
            "test_tools": self.TEST_TOOLS
        }

        return metadata

    def _generate_version(self) -> str:
        """Generate version string from current time.

        Returns:
            Version string in format: run_001_20250524_1530
        """
        from core.utils import get_run_id
        return get_run_id()

    def save_metadata(self, metadata: Dict[str, Any], output_path: str = None) -> str:
        """Save metadata to JSON file.

        Args:
            metadata: Metadata dictionary to save
            output_path: Path to output JSON file. If None,
                        defaults to data/processed/dataset_meta.json

        Returns:
            Path to saved metadata file
        """
        if output_path is None:
            output_path = self.output_dir / "dataset_meta.json"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return str(output_path)

    def build(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        tool_ids: List[str],
        cut_unique_ids: List[str] = None,
        version: str = None
    ) -> Dict[str, Any]:
        """Build complete TFRecord dataset from features and labels.

        This is the main entry point that performs:
        1. Dataset splitting by tool_id
        2. TFRecord file creation
        3. Metadata generation and saving

        Args:
            features: Feature array (n_samples, 168)
            labels: Wear labels array (n_samples, 3)
            tool_ids: List of tool_id strings
            cut_unique_ids: Optional list of cut_unique_ids
            version: Optional version string

        Returns:
            Metadata dictionary with dataset statistics
        """
        if cut_unique_ids is None:
            cut_unique_ids = [f"{tid}_cut{idx}" for idx, tid in enumerate(tool_ids)]

        train_data, test_data = self.split_dataset(features, labels, tool_ids)

        train_output = self.output_dir / "train.tfrecord"
        test_output = self.output_dir / "test.tfrecord"

        train_count = self.create_tfrecord(train_data, str(train_output))
        test_count = self.create_tfrecord(test_data, str(test_output))

        metadata = self.generate_metadata(train_count, test_count, version)
        self.save_metadata(metadata)

        return metadata
