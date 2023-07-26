# -*- coding:UTF-8 -*-

import os
import yaml
import argparse
import torch
import numpy as np
import torch.utils.data as data
from tools.points_process import aug_matrix, generate_rand_rotm, generate_rand_trans, apply_transform
"""
     Read data from KITTI

"""


class points_dataset(data.Dataset):

    def __init__(self, is_training: int = 1, num_point: int = 120000,  data_dir_list: list = [0, 1, 2, 3, 4, 5, 6],
                 config: argparse.Namespace = None, data_keep: list = 'kitti_list'):
        """

        :param train
        :param data_dir_list
        :param config
        :param data_keep
        """
        self.args = config
        data_dir_list.sort()
        self.is_training = is_training
        self.data_list = data_dir_list
        self.data_keep = data_keep
        self.lidar_root = config.lidar_root
        self.pose_root = config.pose_root
        self.data_len_sequence = [4530, 1090, 4650, 790, 260, 2750, 1090, 1090, 4060, 1580, 1190]


        Tr_tmp = []
        data_sum = [0]
        vel_to_cam_Tr = []

        with open('./tools/calib.yaml', "r") as f:
            con = yaml.load(f, Loader=yaml.FullLoader)
        for i in range(11):
            vel_to_cam_Tr.append(np.array(con['Tr{}'.format(i)]))
        for i in self.data_list:
            data_sum.append(data_sum[-1] + self.data_len_sequence[i] + 1)
            # [0, 4531, 5622, 10273, 11064, 11325, 14076]

            Tr_tmp.append(vel_to_cam_Tr[i])

        self.Tr_list = Tr_tmp
        self.data_sum = data_sum
        self.lidar_path = self.lidar_root
        self.dataset = self.make_dataset()



    def __len__(self):
        return self.data_sum[-1]

    def make_dataset(self):
        last_row = np.zeros((1, 4), dtype=np.float32)
        last_row[:, 3] = 1.0
        dataset = []
        sequence_str_list = []
        for item in self.data_list:
            sequence_str_list.append('{:02d}'.format(item))
        for seq in sequence_str_list:
            fn_pair_poses = os.path.join(self.data_keep, seq + '.txt')
            with open(fn_pair_poses, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data_dict = {}
                    line = line.strip(' \n').split(' ')
                    p1 = line[0]
                    p2 = str('{:06d}'.format(int(line[0])+10))
                    src_fn = os.path.join(self.lidar_path, seq, 'velodyne', p1 + '.bin')
                    dst_fn = os.path.join(self.lidar_path, seq, 'velodyne', p2 + '.bin')

                    values = []

                    for i in range(2, len(line)):
                        values.append(float(line[i]))
                    values = np.array(values).astype(np.float32)
                    rela_pose = values.reshape(3, 4)
                    rela_pose = np.concatenate([rela_pose, last_row], axis=0)
                    data_dict['points1'] = src_fn
                    data_dict['points2'] = dst_fn
                    data_dict['pose'] = rela_pose
                    dataset.append(data_dict)

        return dataset

    def __getitem__(self, index):

        data_dict = self.dataset[index]
        fn1_dir = data_dict['points1']
        fn2_dir = data_dict['points2']
        pose = data_dict['pose']

        # source PC and target PC
        point1 = np.fromfile(fn1_dir, dtype=np.float32).reshape(-1, 4)
        point2 = np.fromfile(fn2_dir, dtype=np.float32).reshape(-1, 4)
        pos1 = point1[:, :3].astype(np.float32)
        pos2 = point2[:, :3].astype(np.float32)

        # ground truth pose
        T_gt = np.linalg.inv(pose)

        # Augment
        aug_T = np.zeros((4, 4), dtype=np.float32)
        aug_T[3, 3] = 1.0
        rand_rotm = generate_rand_rotm(1.0, 1.0, 45.0)
        aug_T[:3, :3] = rand_rotm
        pos2 = apply_transform(pos2, aug_T)
        T_gt = T_gt.dot(np.linalg.inv(aug_T))



        return  torch.from_numpy(pos2).float(), torch.from_numpy(pos1).float(), T_gt

    def get_index(self, value, mylist):
        mylist.sort()
        for i, num in enumerate(mylist):
            if num > value:
                return i - 1, mylist[i - 1], num



