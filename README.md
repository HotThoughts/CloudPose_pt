# CloudPose_pt
CloudPose pytorch version.
## Introduction

This repository contains the implementation of CloudPose in PyTorch.

From Paper: "Learning Object Pose Estimation with Point Cloud"

CloudPose is also available in [Tensorflow](https://github.com/GeeeG/CloudPose) 

## Requirements

Pytorch(tested with 1.1.0--1.5.0) on CUDA(tested with 10.0--10.2)

Tensorflow (tested with 1.14.0) for tensorboard

## Preparations

### Make your own data set

0000_seg.npy, 0001_seg.npy...respectively are two-dimensional numpy arrays containing 1024*channel, representing the points in the scene point cloud that have been segmented and downsampled  to 1024.

0000_pos.npy, 0001_pos.npy ...  are one-dimensional numpy arrays containing 17 elements, where [0:12] represents the first three rows and the first four columns of the rigid body transformation matrix, and the last element is the point cloud block Category (0, 1, 2...)

### Dataset Structure
example:

#### train:

./mydataset/my_train/0000_seg.npy

./mydataset/my_train/0000_pos.npy

./mydataset/my_train/0001_seg.npy

./mydataset/my_train/0001_pos.npy


#### val:

./mydataset/my_val/0000_seg.npy

./mydataset/my_val/0000_pos.npy

./mydataset/my_val/0001_seg.npy

./mydataset/my_val/0001_pos.npy


## Train a network

```python
python train.py --batch_size 128 --log_dir log --num_point 256
```



## Visualize loss

```python
python -m tensorboard.main --logdir log --port=3111 --host=127.0.0.1
```



###### References

[CloudPose](https://github.com/GeeeG/CloudPose)

Code Structure from [votenet](https://github.com/facebookresearch/votenet)

[PointNet](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) pytorch implementation
