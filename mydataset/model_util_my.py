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

        self.type2class = {f"cl{i}": i for i in range(self.num_class)}

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
