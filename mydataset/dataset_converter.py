import glob
import os

import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfgt


class TFRecord2NumPy:
    """Convert TF record Dataset to NumPy array."""

    # @tf.function(experimental_follow_type_hints=True)
    def __init__(self, dataset: tf.data.TFRecordDataset):
        self.dataset = dataset.map(lambda x: self._decode(x, 1024))

    def _decode(self, serialized_example, total_num_point):
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

    def _set_rotation_matrix(self, element):
        return tfgt.rotation_matrix_3d.from_quaternion(
            element.get("quaternion"),
        )

    def convert_to_seg_npy(self, element):
        return tf.squeeze(element.get("xyz")).numpy()

    def convert_to_pos_npy(self, element):
        rotation_matrix = self._set_rotation_matrix()
        return tf.concat(
            [
                tf.reshape(rotation_matrix, -1),
                tf.reshape(element.get("translation"), -1),
                tf.constant([0, 0, 0, 1], dtype=tf.float32),
                tf.reshape(tf.cast(element.get("class_id"), tf.float32), -1),
            ],
            axis=0,
        ).numpy()

    def save_npy(self, seg: np.ndarray, pos: np.array, dir_path: str, file_name: str):
        np.save(os.path.join(dir_path, file_name, "_seg"), seg)
        np.save(os.path.join(dir_path, file_name, "_pos"), pos)


if __name__ == "__main__":
    # TODO: write condition
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    ROOT_DIR = os.path.dirname(os.getcwd())
    DIR_NAME = "mydataset/FPS1024"
    train_filenames = list(glob.iglob(os.path.join(ROOT_DIR, DIR_NAME, "*.tfrecords")))
    tr_dataset = tf.data.TFRecordDataset(train_filenames).shuffle(10000)
    data_converter = TFRecord2NumPy(tr_dataset)
    for i in data_converter.dataset:
        print(i)
        break
