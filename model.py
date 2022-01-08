import torch.nn as nn
import torch.utils.data

from pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction
from transformer import TransformerBlock


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(
            k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True
        )

    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])

    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(
            xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)
        ).transpose(1, 2)
        return feats1 + feats2


class Backbone(nn.Module):
    def __init__(
        self, num_class=21, npoints=1024, nblocks=4, nneighbor=16, d_points=768
    ):
        super().__init__()
        transformer_dim = 512
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 1024), nn.ReLU(), nn.Linear(64, 64)
        )
        self.transformer1 = TransformerBlock(64, transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 64 * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(
                    npoints // 4 ** (i + 1),
                    nneighbor,
                    [channel // 2 + 3, channel, channel],
                )
            )
            self.transformers.append(
                TransformerBlock(channel, transformer_dim, nneighbor)
            )
        self.nblocks = nblocks

    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats


class CloudPose_trans(nn.Module):
    def __init__(
        self,
        channel=3,
        num_class=21,
        npoints=1024,
        nblocks=4,
        nneighbor=16,
        d_points=3,
    ):
        super(CloudPose_trans, self).__init__()
        self.backbone = Backbone()
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_class),
        )
        self.nblocks = nblocks

    def forward(self, x):
        batch_size = x.shape[0]
        points, _ = self.backbone(x)
        x = self.fc2(points)
        max_indices = torch.argmax(x, dim=1)
        return x, max_indices


class CloudPose_rot(nn.Module):
    def __init__(
        self,
        channel=3,
        num_class=21,
        npoints=1024,
        nblocks=4,
        nneighbor=16,
        d_points=1024,
    ):
        super(CloudPose_rot, self).__init__()
        self.backbone = Backbone()
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_class),
        )
        self.nblocks = nblocks

    def forward(self, x):
        batch_size = x.shape[0]
        points, _ = self.backbone(x)
        x = self.fc2(points)
        max_indices = torch.argmax(x, dim=1)
        return x, max_indices


class CloudPose_all(nn.Module):
    def __init__(self, channel=3, num_class=5):
        super(CloudPose_all, self).__init__()
        self.num_class = num_class
        self.channel = channel

        self.trans = CloudPose_trans(self.channel, self.num_class)
        self.rot = CloudPose_rot(self.channel, self.num_class)

    def forward(self, input):
        point_clouds = input["point_clouds"]
        point_clouds_tp = point_clouds.transpose(1, 2)  # b 8 256

        base_xyz = torch.mean(point_clouds_tp[:, : self.channel, :], dim=2)
        point_clouds_res = point_clouds_tp[:, : self.channel, :] - base_xyz.unsqueeze(
            -1
        )  # b 3 1
        point_clouds_res_with_cls = torch.cat(
            (point_clouds_res, point_clouds_tp[:, self.channel :, :]), dim=1
        )  # channel 在前 cls在后

        t, ind_t = self.trans(point_clouds_res_with_cls)
        r, ind_r = self.rot(point_clouds_res_with_cls)  # better than point_clouds_tp
        # r, ind_r = self.rot(point_clouds_tp)

        end_points = {}
        end_points["translate_pred"] = t + base_xyz
        end_points["axag_pred"] = r
        return end_points


if __name__ == "__main__":
    sim_data = torch.rand(32, 2500, 3 + 5)
    input = {}
    input["point_clouds"] = sim_data
    feat = CloudPose_all(3, 5)
    end_points = feat(input)
    print(end_points["translate_pred"], end_points["axag_pred"])
