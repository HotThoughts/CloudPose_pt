# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "utils"))


class myDatasetConfig(object):
    def __init__(self):
        self.num_class = 21

        # Class name and id map
        # see: https://github.com/GeeeG/CloudPose/blob/d6410dc4af9a58c00511e34fdc41c6cfd9f96cba/ycb_video_data_tfRecords/script/2_dataset_to_tfRecord_small.py
        self.type2class = {
            "master chef can": 0,
            "cracker box": 1,
            "suger box": 2,
            "tomato soup can": 3,
            "mustard bottle": 4,
            "tuna fish can": 5,
            "pudding box": 6,
            "gelatin box": 7,
            "potted meat can": 8,
            "banana": 9,
            "pitcher base": 10,
            "bleach cleanser": 11,
            "bowl": 12,
            "mug": 13,
            "drill": 14,
            "wood block": 15,
            "scissors": 16,
            "large marker": 17,
            "large clapm": 18,
            "extra large clamp": 19,
            "foam brick": 20,
        }

        self.class2type = {self.type2class[t]: t for t in self.type2class}

        # 2D array
        self.onehot_encoding = np.eye(self.num_class)[
            np.array([range(self.num_class)]).reshape(-1)
        ]

    def sem2class(self, cls):
        # Select ith row of the 2D array
        onehot = self.onehot_encoding[int(cls), :]
        return onehot

    def size2class(self, type_name):
        """Convert 3D box size (l,w,h) to size class and size residual"""
        size_class = self.type2class[type_name]  # 0
        # size_residual = size - self.type_mean_size[type_name]  # 尺寸
        return size_class

    def class2size(self, pred_cls):
        """Inverse function to size2class"""
        mean_size = self.type_mean_size[self.class2type[pred_cls]]
        return mean_size

    def class2sem(self, pred_cls):
        sem = self.class2type[pred_cls]
        return sem
