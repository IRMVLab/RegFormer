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

    point2 = [item[0] for item in list_data]
    # print(np.asarray(point2[0]).shape)
    point1 = [item[1] for item in list_data]

    # sample_id = torch.from_numpy(np.asarray([item[2] for item in list_data]))
    T_gt = torch.from_numpy(np.asarray([item[2] for item in list_data]))
    T_trans = torch.from_numpy(np.asarray([item[3] for item in list_data]))
    T_trans_inv = torch.from_numpy(np.asarray([item[4] for item in list_data]))



    # Collate as normal, other than fields that cannot be collated due to differing sizes,
    # we retain it as a python list
    # to_retain_as_list = ['point2', 'point1', 'imback2', 'imback1', 'pts_origin_xy2', 'pts_origin_xy1', 'sample_id'
    #                      ]
    # data = {k: [list_data[b][k] for b in range(batch_sz)] for k in to_retain_as_list if k in list_data[0]}
    # data['T_gt'] = torch.stack([list_data[b]['T_gt'] for b in range(batch_sz)], dim=0)  # (B, 4, 4)
    # data['T_trans'] = torch.stack([list_data[b]['T_trans'] for b in range(batch_sz)], dim=0) # (B, 4, 4)
    # data['T_trans_inv'] = torch.stack([list_data[b]['T_trans_inv'] for b in range(batch_sz)], dim=0)  # (B, 4, 4)
    # data['Tr'] = torch.stack([list_data[b]['Tr'] for b in range(batch_sz)], dim=0)  # (B, 4, 4)

    return point2, point1, T_gt, T_trans, T_trans_inv
