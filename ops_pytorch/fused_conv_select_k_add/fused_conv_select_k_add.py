import torch
import sys
import os
import numpy as np
import fused_conv_select_k_add_cuda as fused_conv_select_k_add_module

def fused_conv_select_add_k(xyz1, xyz2, idx_n2, random_hw, H, W, npoints, kernel_size_H, kernel_size_W, K, flag_copy, distance, select_b_idx, select_h_idx, select_w_idx, add_b_idx, add_h_idx, add_w_idx, valid_idx, valid_in_dis_idx,select_mask, add_mask):
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
    fused_conv_select_k_add_module.fused_conv_select_add_k(xyz1, xyz2, idx_n2, random_hw, H, W, npoints, kernel_size_H, kernel_size_W, K, flag_copy, distance, select_b_idx, select_h_idx, select_w_idx, add_b_idx, add_h_idx,add_w_idx, valid_idx, valid_in_dis_idx,select_mask, add_mask)
    return select_b_idx, select_h_idx, select_w_idx, add_b_idx, add_h_idx, add_w_idx, valid_idx, valid_in_dis_idx, select_mask, add_mask

    
    
if  __name__=='__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

    import time
    import numpy as np

    

    batch_size = 1
    H = 4
    W = 7
    C = 3
    npoints = 2
    kernel_size_H = 1
    kernel_size_W = 5
    distance = 200
    
    K = 8
    flag_copy = 0

    point_cloud_pj = np.arange(H * W).astype('float32')
    point_cloud_pj = np.tile(np.reshape(point_cloud_pj, [1, H, W, 1]), [1, 1, 1, 3])

    idx_n2 = np.array([[[0, 0], [0, 1]]]).astype('int32')

    xyz = torch.from_numpy(point_cloud_pj)
    xyz1 = xyz.cuda()
    xyz2 = xyz1
    
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

    print("select_b_idx:",select_b_idx)

    add_b_idx = torch.cuda.LongTensor(batch_size, H, W, 1)                           # (B, H, W, 1)
    add_h_idx = torch.cuda.LongTensor(batch_size, H, W, 1)
    add_w_idx = torch.cuda.LongTensor(batch_size, H, W, 1)

    print("add_b_idx:",add_b_idx)

    valid_idx = torch.cuda.FloatTensor(batch_size, npoints, H*W, 1)
    valid_in_dis_idx = torch.cuda.FloatTensor(batch_size, npoints, H*W, 1)
    select_mask = torch.cuda.FloatTensor(batch_size, npoints, K, 1)
    add_mask = torch.cuda.FloatTensor(batch_size, H, W, 1)                           # (B, H, W, 1)
    # print(xyz1.dtype, xyz2.dtype, idx_n2.dtype, random_hw.dtype, type(H), type(W), type(npoints), type(kernel_size_H), type(kernel_size_W), type(K), type(bool(flag_copy)), type(float(distance)), select_bhw_idx.dtype, valid_idx.dtype, valid_in_dis_idx.dtype ,select_mask.dtype)

    CUDA_before = time.time()
    select_b_idx, select_h_idx, select_w_idx, add_b_idx, add_h_idx, add_w_idx, valid_idx, valid_in_dis_idx, select_mask, add_mask = fused_conv_select_add_k(xyz1, xyz2, idx_n2, random_hw, H, W, npoints, kernel_size_H, kernel_size_W, K, bool(flag_copy), float(distance), select_b_idx, select_h_idx, select_w_idx, add_b_idx,add_h_idx, add_w_idx, valid_idx, valid_in_dis_idx, select_mask, add_mask)
    CUDA_after = time.time()
    time1 = CUDA_after - CUDA_before
    print(time1)
    print(' fused_conv_add_k done ')

    print("select_b_idx_after:",select_b_idx)
    print("add_b_idx_after:",add_b_idx)

    Select_before = time.time()
    
    select_b_idx = select_b_idx.cpu()
    select_h_idx = select_h_idx.cpu()
    select_w_idx = select_w_idx.cpu()

    add_b_idx = add_b_idx.cpu()
    add_h_idx = add_h_idx.cpu()
    add_w_idx = add_w_idx.cpu()

    select_xyz_feature = point_cloud_pj[select_b_idx, select_h_idx, select_w_idx, : ]
    
    print("select_xyz_feature_before_reshaped:",select_xyz_feature.shape)
    select_xyz_feature = select_xyz_feature.reshape(batch_size, npoints, K, 3)
    print("select_xyz_feature_after_reshaped:",select_xyz_feature.shape)
    print("select_xyz_feature_after_reshaped:",select_xyz_feature)
    select_xyz_feature = torch.from_numpy(select_xyz_feature)
    select_xyz_feature = select_xyz_feature.cuda()
    select_xyz_feature = select_xyz_feature * select_mask

    add_xyz_feature = point_cloud_pj[add_b_idx, add_h_idx, add_w_idx, : ]
    print("add_xyz_feature_before_reshaped:",add_xyz_feature.shape)
    add_xyz_feature = add_xyz_feature.reshape(batch_size, H, W, 3)
    print("add_xyz_feature_after_reshaped:",add_xyz_feature.shape)
    print("add_xyz_feature_after_reshaped:",add_xyz_feature)
    add_xyz_feature = torch.from_numpy(add_xyz_feature)
    add_xyz_feature = add_xyz_feature.cuda()
    add_xyz_feature = add_xyz_feature * add_mask

    point_cloud_pj = torch.from_numpy(point_cloud_pj)
    point_cloud_pj = point_cloud_pj.cuda()
    point_cloud_pj = point_cloud_pj + add_xyz_feature
    print("added_point_cloud_pj:",point_cloud_pj.shape)

    new_point_cloud_pj = point_cloud_pj[select_b_idx, select_h_idx, select_w_idx, : ]
    print("final_point_cloud_before_reshaped:",new_point_cloud_pj.shape)
    new_point_cloud_pj = new_point_cloud_pj.reshape(batch_size, npoints, K, 3)
    print("final_point_cloud_after_reshaped:",new_point_cloud_pj.shape)
    new_point_cloud_expand = new_point_cloud_pj
    # new_point_cloud = new_point_cloud_pj.reshape(batch_size, -1, 3)
    # print("final_point_cloud:",new_point_cloud.shape)

    # new_point_cloud_expand = (torch.unsqueeze(new_point_cloud, 2)).repeat(1,1,K,1)
    # print("final_point_cloud_expand:",new_point_cloud_expand.shape)

    point_cloud_difference = new_point_cloud_expand - select_xyz_feature
    print("final_point_cloud_difference:",point_cloud_difference.shape)

    point_cloud_final = torch.cat([point_cloud_difference, select_xyz_feature], axis = -1)
    print("final_point_cloud:",point_cloud_final.shape)

    Select_after = time.time()
    time2 = Select_after - Select_before
    print(time2)

    print(' ---end--- ')
    
    
