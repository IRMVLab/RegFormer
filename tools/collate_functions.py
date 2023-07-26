import torch
import numpy as np


def collate_pair(list_data):
    """Collates data using a list, for tensors which are of different sizes
    (e.g. different number of points). Otherwise, stacks them as per normal.
    """
    # print(np.asarray(list_data).shape)
    # batch_sz = len(list_data)
    # print(batch_sz)

    point2 = [item[0] for item in list_data]
    point1 = [item[1] for item in list_data]
    T_gt = torch.from_numpy(np.asarray([item[2] for item in list_data]))



    return point2, point1, T_gt
