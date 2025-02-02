import glob
import os
import random
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfgt
import yaml
from tqdm import tqdm


class TFRecord2NumPy:
    """Convert TF record Dataset to NumPy array."""

    def __init__(self, dataset):
        self.dataset = dataset.map(lambda x: self._decode(x, 1024))

    def _decode(self, serialized_example, total_num_point: int):
        features = tf.io.parse_example(
            [serialized_example],
            features={
                "xyz": tf.io.FixedLenFeature([total_num_point, 3], tf.float32),
                "rgb": tf.io.FixedLenFeature([total_num_point, 3], tf.float32),
                "translation": tf.io.FixedLenFeature([3], tf.float32),
                "quaternion": tf.io.FixedLenFeature([4], tf.float32),
                "num_valid_points_in_segment": tf.io.FixedLenFeature([], tf.int64),
                "seq_id": tf.io.FixedLenFeature([], tf.int64),
                "frame_id": tf.io.FixedLenFeature([], tf.int64),
                "class_id": tf.io.FixedLenFeature([], tf.int64),
            },
        )
        return features

    def _set_rotation_matrix(self, element) -> np.ndarray:
        rot = tfgt.rotation_matrix_3d.from_quaternion(
            element["quaternion"],
        )
        assert rot.shape == (1, 3, 3)
        return rot

    def convert_to_seg_npy(self, element) -> np.ndarray:
        seg = tf.squeeze(element["xyz"]).numpy()
        assert seg.shape == (1024, 3)
        return seg

    def convert_to_pos_npy(self, element) -> np.array:
        rotation_matrix = self._set_rotation_matrix(element)
        pos = tf.concat(
            [
                tf.reshape(rotation_matrix, -1),
                tf.reshape(element.get("translation"), -1),
                tf.constant([0, 0, 0, 1], dtype=tf.float32),
                tf.reshape(tf.cast(element.get("class_id"), tf.float32), -1),
            ],
            axis=0,
        ).numpy()
        assert pos.shape == (17,)
        return pos

    def save_npy(
        self, seg: np.ndarray, pos: np.array, dir_path: str, file_basename: int
    ):
        np.save(os.path.join(dir_path, f"{file_basename:05d}_seg"), seg)
        np.save(os.path.join(dir_path, f"{file_basename:05d}_pos"), pos)


def prepare_file_paths(dir_name: str) -> List[str]:
    "Return a list of *.tfrecords files"
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DIR_NAME = "FPS1024"
    FILE_PATHS = list(glob.iglob(os.path.join(ROOT_DIR, DIR_NAME, "*.tfrecords")))
    return FILE_PATHS


def count_tfrecord_dataset(ds):
    """Iterate through TFRecord dataset to count the numer of data points"""
    ds_size = sum(1 for _ in ds)
    return ds_size


def get_dataset_partitions(
    ds_size: int,
    split: List[float],
) -> List[List]:
    assert sum(split) == 1
    train_split, val_split, _ = split

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    shuffled_list = list(range(ds_size))
    random.shuffle(shuffled_list)

    train_list = shuffled_list[0:train_size]
    val_list = shuffled_list[train_size : train_size + val_size]
    test_list = shuffled_list[train_size + val_size :]
    assert len(train_list) + len(val_list) + len(test_list) == ds_size

    return train_list, val_list, test_list


if __name__ == "__main__":
    # Env needs for CPU
    if len(tf.config.list_physical_devices("GPU")) == 0:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)

    TRAIN_FILES = prepare_file_paths(params["data"]["tfrecord_dir"])
    # Read all data and shuffle
    tr_dataset = tf.data.TFRecordDataset(
        TRAIN_FILES
    )  # .shuffle(params["data"]["shuffle"])
    data_converter = TFRecord2NumPy(tr_dataset)
    # Split into train, val and test
    train_list, val_list, _ = get_dataset_partitions(
        ds_size=params["data"]["total_num_items"],
        split=params["data"]["split"],
    )
    # Prepare path
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_DIR = os.path.join(ROOT_DIR, params["data"]["train_dir"])
    VAL_DIR = os.path.join(ROOT_DIR, params["data"]["val_dir"])
    TEST_DIR = os.path.join(ROOT_DIR, params["data"]["test_dir"])

    for count, element in tqdm(
        enumerate(data_converter.dataset),
        total=params["data"]["total_num_items"],
        desc="Saving seg and pos .npy files...",
    ):
        if count in train_list:
            DIR = TRAIN_DIR
        elif count in val_list:
            DIR = VAL_DIR
        else:
            DIR = TEST_DIR

        pos = data_converter.convert_to_pos_npy(element)
        seg = data_converter.convert_to_seg_npy(element)
        data_converter.save_npy(
            seg=seg,
            pos=pos,
            dir_path=DIR,
            file_basename=count,
        )
