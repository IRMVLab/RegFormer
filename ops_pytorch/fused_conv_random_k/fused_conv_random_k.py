import torch
import sys
import os
import numpy as np
import fused_conv_random_k_cuda as fused_conv_random_k_module


def fused_conv_random_k(xyz1, xyz2, idx_n2, random_hw, H, W, npoints, kernel_size_H, kernel_size_W, K, flag_copy, distance, stride_h, stride_w, select_b_idx, select_h_idx, select_w_idx, valid_idx, valid_in_dis_idx, select_mask,small_h,small_w):
    '''
    Input:
        xyz1:(b, h, w, 3) float, projected xyz1 points 
        xyz2_feature:(b, h, w, c+3) float, projected xyz2 points with features
        idx_n2: (b, n, 2) int array, query idx of central points
        H, W : Input shape
        kernel_size_H, kernel_size_W: (size, size) int32 array, size
        k: the number of selected points (knn)
        distance: ( distance ) float  distance
        flag_copy  (bool)  whether copy or not for the output points
    
    Output:
        space_weight:(batch_size, npoint,  size*size , c)
    '''
    fused_conv_random_k_module.fused_conv_random_k(xyz1, xyz2, idx_n2, random_hw, H, W, npoints, kernel_size_H, kernel_size_W, K, flag_copy, distance, stride_h, stride_w, select_b_idx, select_h_idx, select_w_idx, valid_idx, valid_in_dis_idx, select_mask,small_h,small_w)
    return select_b_idx, select_h_idx, select_w_idx, valid_idx, valid_in_dis_idx, select_mask


if  __name__=='__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

    import time
    import numpy as np

    stride_h = 1 
    stride_w = 2

    batch_size = 1
    
    H = 4
    W = 9
    C = 3

    SMALL_H = 4
    SMALL_W = 5

    npoints = 2
    kernel_size_H = 1
    kernel_size_W = 5
    distance = 200

    K = 8
    flag_copy = 0

    point_cloud_pj_1 = np.arange(0,H * W).astype('float32')
    point_cloud_pj_1 = np.tile(np.reshape(point_cloud_pj_1, [1, H, W, 1]), [1, 1, 1, 3])

    print("point_cloud_pj_1:",point_cloud_pj_1.shape)
    print(point_cloud_pj_1)

    point_cloud_pj_2 = np.arange(0,SMALL_H * SMALL_W).astype('float32')
    point_cloud_pj_2 = np.tile(np.reshape(point_cloud_pj_2, [1, SMALL_H, SMALL_W, 1]), [1, 1, 1, 3])

    print("point_cloud_pj_2:",point_cloud_pj_2.shape)
    print(point_cloud_pj_2)

    idx_n2 = np.array([[[0, 0], [0, 1]]]).astype('int32')

    xyz1 = torch.from_numpy(point_cloud_pj_1)
    xyz1 = xyz1.cuda()
    
    xyz2 = torch.from_numpy(point_cloud_pj_2)
    xyz2 = xyz2.cuda()


    idx_n2_tmp1 = torch.from_numpy(idx_n2)
    idx_n2_tmp2 = idx_n2_tmp1.int()
    idx_n2 = idx_n2_tmp2.cuda()

    # random_H = tf.random_shuffle(tf.range(kernel_size_H) - kernel_size_H//2)
    # random_W = tf.random_shuffle(tf.range(kernel_size_W) - kernel_size_W//2)
    random_hw_tmp1 = torch.randperm(kernel_size_H * kernel_size_W)
    #print(sys.getsizeof(int))
    random_hw_tmp2 = random_hw_tmp1.int()
    random_hw = random_hw_tmp2.cuda()

    select_b_idx = torch.cuda.LongTensor(batch_size, npoints, K, 1)
    select_h_idx = torch.cuda.LongTensor(batch_size, npoints, K, 1)
    select_w_idx = torch.cuda.LongTensor(batch_size, npoints, K, 1)


    valid_idx = torch.cuda.FloatTensor(batch_size, npoints, H*W, 1)
    valid_in_dis_idx = torch.cuda.FloatTensor(batch_size, npoints, H*W, 1)
    select_mask = torch.cuda.FloatTensor(batch_size, npoints, K, 1)
    # print(xyz1.dtype, xyz2.dtype, idx_n2.dtype, random_hw.dtype, type(H), type(W), type(npoints), type(kernel_size_H), type(kernel_size_W), type(K), type(bool(flag_copy)), type(float(distance)), select_bhw_idx.dtype, valid_idx.dtype, valid_in_dis_idx.dtype ,select_mask.dtype)

    CUDA_before = time.time()
    select_b_idx, select_h_idx, select_w_idx, valid_idx, valid_in_dis_idx, select_mask = fused_conv_random_k(xyz1, xyz2, idx_n2, random_hw, H, W, npoints, kernel_size_H, kernel_size_W, K, bool(flag_copy), float(distance), stride_h, stride_w, select_b_idx, select_h_idx, select_w_idx, valid_idx, valid_in_dis_idx, select_mask,SMALL_H,SMALL_W)
    CUDA_after = time.time()
    time1 = CUDA_after - CUDA_before
    print(time1)
 
    Select_before = time.time()
    select_b_idx = select_b_idx.cpu()
    select_h_idx = select_h_idx.cpu()
    select_w_idx = select_w_idx.cpu()
    
    print("select_b_idx_after:",select_b_idx.shape)
    print(select_b_idx)
    print("select_h_idx_after:",select_h_idx.shape)
    print(select_h_idx)
    print("select_w_idx_after:",select_w_idx.shape)
    print(select_w_idx)

    select_xyz_feature = point_cloud_pj_2[select_b_idx, select_h_idx, select_w_idx, : ]
    print("select_xyz_feature_before_reshaped",select_xyz_feature.shape)
    select_xyz_feature = select_xyz_feature.reshape(batch_size, npoints, K, 3)
    print("select_xyz_feature_after_reshaped",select_xyz_feature.shape)
    select_xyz_feature = torch.from_numpy(select_xyz_feature)

    Select_after = time.time()
    time2 = Select_after - Select_before
    print(time2)

    select_xyz_feature = select_xyz_feature.cuda()
    select_xyz_feature = select_xyz_feature * select_mask

    print(' conv 2d ok ')
    print(select_b_idx,select_h_idx,select_w_idx)
    print("select_b_idx:",select_b_idx.shape)
    print("select_h_idx:",select_h_idx.shape)
    print("select_w_idx:",select_w_idx.shape)

    print('selected__xyz: ', select_xyz_feature[:, :, :, :3])
    print("selected__xyz:",select_xyz_feature.shape)
    
