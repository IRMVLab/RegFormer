# -*- coding:UTF-8 -*-

import torch
import numpy as np
from scipy.spatial.transform import Rotation
# author:Zhiheng Feng
# contact: fzhsjtu@foxmail.com
# datetime:2021/10/21 19:46
# software: PyCharm

"""
文件说明：
    用于点云处理的一些函数
"""

def generate_rand_rotm(x_lim=5.0, y_lim=5.0, z_lim=180.0):
    '''
    Input:
        x_lim
        y_lim
        z_lim
    return:
        rotm: [3,3]
    '''
    rand_z = np.random.uniform(low=-z_lim, high=z_lim)
    rand_y = np.random.uniform(low=-y_lim, high=y_lim)
    rand_x = np.random.uniform(low=-x_lim, high=x_lim)

    rand_eul = np.array([rand_z, rand_y, rand_x])
    r = Rotation.from_euler('zyx', rand_eul, degrees=True)
    rotm = r.as_matrix()
    return rotm

def generate_rand_trans(x_lim=10.0, y_lim=1.0, z_lim=0.1):
    '''
    Input:
        x_lim
        y_lim
        z_lim
    return:
        trans [3]
    '''
    rand_x = np.random.uniform(low=-x_lim, high=x_lim)
    rand_y = np.random.uniform(low=-y_lim, high=y_lim)
    rand_z = np.random.uniform(low=-z_lim, high=z_lim)

    rand_trans = np.array([rand_x, rand_y, rand_z])

    return rand_trans

def apply_transform(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

def limited_points(points: np.float, npoints: int = 8192, fov_filter: bool = False) -> np.float:
    """

    :param points: 原点云
    :param npoints: 处理后的点数目
    :param fov_filter:  是否进行前方90度的视野限制
    :return:
    """
    x_range = np.array([-30, 30])
    y_range = np.array([-2, 1.1])
    z_range = np.array([-30, 30])
    is_ground = np.logical_or(points[:, 1] > y_range[1], points[:, 1] > y_range[1])
    not_ground = np.logical_not(is_ground)

    near_mask_x = np.logical_and(points[:, 0] < x_range[1], points[:, 0] > x_range[0])
    near_mask_z = np.logical_and(points[:, 2] < z_range[1], points[:, 2] > z_range[0])
    near_mask = np.logical_and(near_mask_x, near_mask_z)
    near_mask = np.logical_and(not_ground, near_mask)

    if fov_filter:  # 限制点云视野，限制90度
        near_mask_fov = np.logical_and(points[:, 2] > points[:, 0], points[:, 2] > -points[:, 0])
        near_mask = np.logical_and(near_mask_fov, near_mask)

    indices = np.where(near_mask)[0]
    if len(indices) >= npoints:
        sample_idx = np.random.choice(indices, npoints, replace=False)
    else:
        repeat_times = int(npoints / len(indices))
        sample_num = npoints % len(indices)
        sample_idx = np.concatenate(
            [np.repeat(indices, repeat_times), np.random.choice(indices, sample_num, replace=False)],
            axis=-1)

    return points[sample_idx]


def filter_points(points: np.float, npoints: int = 8192, fov_filter=True, furthest: int = 40) -> np.float:
    """

    :param points: 相机坐标系下的点云，np.array [n,3] or [n,4]
    :param fov_filter: 使用前方90度视野的点云
    :param furthest: z方向的临界值，该临界值以外的点云保留，临界值以内的点云随机采样
    :return: 处理后的点云，数据类型和输入点云一致,与limited_points不同的是原处保留
    """

    # 初步进行点云范围限制
    x_range = np.array([-30, 30])
    y_range = np.array([-2, 1.1])
    z_range = np.array([-30, 50])
    is_ground = np.logical_or(points[:, 1] > y_range[1], points[:, 1] > y_range[1])
    not_ground = np.logical_not(is_ground)

    near_mask_x = np.logical_and(points[:, 0] < x_range[1], points[:, 0] > x_range[0])
    near_mask_z = np.logical_and(points[:, 2] < z_range[1], points[:, 2] > z_range[0])
    near_mask = np.logical_and(near_mask_x, near_mask_z)
    near_mask = np.logical_and(not_ground, near_mask)

    if fov_filter:  # 限制点云视野，限制90度
        near_mask_fov = np.logical_and(points[:, 2] > points[:, 0], points[:, 2] > -points[:, 0])
        near_mask = np.logical_and(near_mask_fov, near_mask)

    limited_points = points[near_mask]  # 这个是进行初步范围限制后的点云，下一步进行采样

    is_far = np.logical_and(limited_points[:, 2] > furthest, limited_points[:, 2] > furthest)
    far_indices = np.where(is_far)[0]

    if len(far_indices) > npoints:  # 如果最外面的点云超过了数目要求
        sample_far = np.random.choice(far_indices, npoints, replace=False)
        return limited_points[sample_far]
    else:
        is_near = np.logical_not(is_far)
        near_indices = np.where(is_near)[0]
        if len(near_indices) == 0:
            repeat_times = int(npoints / len(far_indices))
            sample_num = npoints % len(far_indices)
            sample_all = np.concatenate(
                [np.repeat(far_indices, repeat_times), np.random.choice(far_indices, sample_num, replace=False)],
                axis=-1)
            return limited_points[sample_all]
        else:
            repeat_times = int((npoints - len(far_indices)) / len(near_indices))
            sample_num = (npoints - len(far_indices)) % len(near_indices)
            sample_all = np.concatenate(
                [np.repeat(near_indices, repeat_times), np.random.choice(near_indices, sample_num, replace=False),
                 far_indices],
                axis=-1)
            return limited_points[sample_all]


def aug_matrix():
    
    anglex = np.clip(0.01 * np.random.randn(), -0.02, 0.02).astype(np.float32) * np.pi / 4.0
    angley = np.clip(0.01 * np.random.randn(), -0.02, 0.02).astype(np.float32) * np.pi / 4.0
    anglez = np.clip(0.05 * np.random.randn(), -0.1, 0.1).astype(np.float32) * np.pi / 4.0

    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    
    Rx = np.array([[1, 0, 0],
                    [0, cosx, -sinx],
                    [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny],
                    [0, 1, 0],
                    [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0],
                    [sinz, cosz, 0],
                    [0, 0, 1]])

    scale = np.diag(np.random.uniform(1.00, 1.00, 3).astype(np.float32))
    R_trans = Rx.dot(Ry).dot(Rz).dot(scale.T)
    # R_trans = Rx.dot(Ry).dot(Rz)

    xx = np.clip(0.5 * np.random.randn(), -1.0, 1.0).astype(np.float32)
    yy = np.clip(0.1 * np.random.randn(), -0.2, 0.2).astype(np.float32)
    zz = np.clip(0.05 * np.random.randn(), -0.15, 0.15).astype(np.float32)

    add_xyz = np.array([[xx], [yy], [zz]])

    T_trans = np.concatenate([R_trans, add_xyz], axis=-1)
    filler = np.array([0.0, 0.0, 0.0, 1.0])
    filler = np.expand_dims(filler, axis=0)  ##1*4
    T_trans = np.concatenate([T_trans, filler], axis=0)  # 4*4

    return T_trans



def point_aug(cloud: torch.Tensor) -> torch.Tensor:
    """

    :param cloud:[n,3]或者[n,4]的点云
    :return:增强后[n,4]的点云
    """

    if cloud.shape[1] == 4:
        density = cloud[:, 3:4]
    elif cloud.shape[1] == 3:
        N = cloud.shape[0]
        density = torch.ones([N, 1], device=cloud.device)
    else:
        print('[Attenton]: the input points shape is {} which is wrong'.format(cloud.shape))

    cloud = torch.cat([cloud[:, :3], density], dim=-1)
    T_trans = aug_matrix()
    T_trans = torch.from_numpy(T_trans)
    T_trans = T_trans.cuda(device=cloud.device)

    points_trans = torch.matmul(T_trans, cloud.t())
    points_trans = points_trans.t()

    points_trans = torch.cat([points_trans[:, :3], density], dim=-1)
    return points_trans
