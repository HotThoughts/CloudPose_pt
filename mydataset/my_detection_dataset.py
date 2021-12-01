import os
import sys

import cv2
import numpy as np
from torch.utils.data import Dataset

from mydataset.model_util_my import myDatasetConfig
from utils import pc_util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "utils"))

DC = myDatasetConfig()  # Dataset specific Config
MEAN_COLOR_RGB = np.array([0.5, 0.5, 0.5])  # sunrgbd color is in 0~1


class myDetectionDataset(Dataset):
    def __init__(
        self, split_set="train", num_points=1024, use_color=False, scan_idx_list=None
    ):
        assert num_points <= 4096

        self.data_path = os.path.join(BASE_DIR, "my_%s" % (split_set))

        self.scan_names = sorted(
            list(set([os.path.basename(x)[0:5] for x in os.listdir(self.data_path)]))
        )

        if scan_idx_list is not None:
            self.scan_names = [self.scan_names[i] for i in scan_idx_list]
        self.num_points = num_points
        self.use_color = use_color

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):

        scan_name = self.scan_names[idx]
        point_cloud = np.load(
            os.path.join(self.data_path, scan_name) + "_seg.npy"
        )  # Nx3
        bbox = np.load(os.path.join(self.data_path, scan_name) + "_pos.npy")  # K,20

        if not self.use_color:
            point_cloud = point_cloud[:, 0:3]
        else:
            point_cloud = point_cloud[:, 0:6]
            point_cloud[:, 3:] = point_cloud[:, 3:] - MEAN_COLOR_RGB

        # ------------------------------- LABELS ------------------------------
        ret_dict = {}
        point_cloud = pc_util.random_sampling(point_cloud, self.num_points)
        one_hot = DC.sem2class(bbox[16])
        one_hot_ex_rep = np.repeat(
            np.expand_dims(one_hot, axis=0), self.num_points, axis=0
        )
        point_cloud_with_cls = np.concatenate((point_cloud, one_hot_ex_rep), axis=1)

        ret_dict["point_clouds"] = point_cloud_with_cls.astype(np.float32)

        matrix44 = bbox[0:16].reshape((4, 4))
        matrix33 = matrix44[:3, :3]
        axag = cv2.Rodrigues(matrix33)[0].flatten()
        ret_dict["axag_label"] = axag.astype(np.float32)

        translate = matrix44[:3, 3]
        ret_dict["translate_label"] = translate.astype(np.float32)

        return ret_dict
