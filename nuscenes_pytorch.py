# -*- coding:UTF-8 -*-

import os
import yaml
import argparse
import torch
import numpy as np
import torch.utils.data as data
from tools.points_process import aug_matrix, generate_rand_rotm, generate_rand_trans, apply_transform

"""
     Read data from Nuscenes

"""


class points_dataset(data.Dataset):

    def __init__(self, is_training: int = 1, num_point: int = 24000,  data_dir_list: list = 'test',
                 config: argparse.Namespace = None, data_keep: list = 'nuscenes_list'):
        """

        :param train
        :param data_dir_list
        :param config
        """
        self.args = config
        self.is_training = is_training
        self.seqs = data_dir_list
        self.data_keep = data_keep
        self.lidar_root = config.lidar_root
        self.dataset = self.make_dataset()


    def make_dataset(self):
        last_row = np.zeros((1, 4), dtype=np.float32)
        last_row[:, 3] = 1.0
        dataset = []
        for seq in self.seqs:
            if seq == 'test':
                data_root = os.path.join(self.lidar_root, 'v1.0-test')
            else:
                data_root = os.path.join(self.lidar_root, 'v1.0-trainval')
            fn_pair_poses = os.path.join(self.data_keep, seq + '.txt')
            with open(fn_pair_poses, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data_dict = {}
                    line = line.strip(' \n').split(' ')
                    src_fn = os.path.join(data_root, line[0])
                    dst_fn = os.path.join(data_root, line[1])
                    values = []
                    for i in range(2, len(line)):
                        values.append(float(line[i]))
                    values = np.array(values).astype(np.float32)
                    rela_pose = values.reshape(3, 4)
                    rela_pose = np.concatenate([rela_pose, last_row], axis=0)
                    data_dict['points1'] = src_fn
                    data_dict['points2'] = dst_fn
                    data_dict['Tr'] = rela_pose
                    dataset.append(data_dict)

        return dataset


    def __getitem__(self, index):

        data_dict = self.dataset[index]
        fn1_dir = data_dict['points1']
        fn2_dir = data_dict['points2']
        pose = data_dict['Tr']



        point1 = np.fromfile(fn1_dir, dtype=np.float32).reshape(-1, 5)
        point2 = np.fromfile(fn2_dir, dtype=np.float32).reshape(-1, 5)
        pos1 = point1[:, :3].astype(np.float32)
        pos2 = point2[:, :3].astype(np.float32)

        T_gt = np.linalg.inv(pose)



        # Augment matrix#
        if self.is_training:
            T_trans = aug_matrix()
        else:
            T_trans = np.eye(4).astype(np.float32)

        T_trans_inv = np.linalg.inv(T_trans)

        if self.is_training:
            aug_T = np.zeros((4, 4), dtype=np.float32)
            aug_T[3, 3] = 1.0
            rand_rotm = generate_rand_rotm(0.0, 0.0, 30.0)
            aug_T[:3, :3] = rand_rotm
            pos2 = apply_transform(pos2, aug_T)
            T_gt = T_gt.dot(np.linalg.inv(aug_T))



        return  torch.from_numpy(pos2).float(), torch.from_numpy(pos1).float(), T_gt, T_trans, T_trans_inv

    def get_index(self, value, mylist):
        mylist.sort()
        for i, num in enumerate(mylist):
            if num > value:
                return i - 1, mylist[i - 1], num

    def __len__(self):
        return len(self.dataset)




