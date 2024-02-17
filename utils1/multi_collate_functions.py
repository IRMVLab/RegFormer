import torch
import numpy as np


def collate_pair(list_data):
    """Collates data using a list, for tensors which are of different sizes
    (e.g. different number of points). Otherwise, stacks them as per normal.
    """
    # print(np.asarray(list_data).shape)
    #
    # batch_sz = len(list_data)
    # print(batch_sz)
    # print(len(list_data[0]))
    # print(len(list_data[2]))
    point2 = torch.tensor([(item[0]).astype(np.float32) for item in list_data])
    # print(point2.type)
    # point2=torch.from_numpy(point2)
    # print(len(point2))
    point1 = torch.tensor([(item[1]).astype(np.float32) for item in list_data])
    # point1 = ([torch.from_numpy(item[1]) for item in list_data])

    # sample_id = torch.from_numpy(np.asarray([item[2] for item in list_data]))
    T_gt = torch.from_numpy(np.asarray([item[2] for item in list_data]))
    T_trans = torch.from_numpy(np.asarray([item[3] for item in list_data]))
    T_trans_inv = torch.from_numpy(np.asarray([item[4] for item in list_data]))
    # print(T_gt.shape)

    return point2, point1, T_gt, T_trans, T_trans_inv
