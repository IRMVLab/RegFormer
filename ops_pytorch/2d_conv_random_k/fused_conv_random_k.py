import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


fused_conv_random_k_module = tf.load_op_library(os.path.join(BASE_DIR, 'fused_conv_so.so'))


def fused_conv_random_k(xyz1, xyz2, idx_n2, random_hw, H, W, npoints, kernel_size_H, kernel_size_W, K, flag_copy, distance, stride_h, stride_w):
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
    return fused_conv_random_k_module.fused_conv_random_k(xyz1, xyz2, idx_n2, random_hw, H, W, npoints, kernel_size_H, kernel_size_W, K, flag_copy, distance, stride_h, stride_w)



# class FusedConvBetweenTest(tf.test.TestCase):
#     def test(self):
#         pass
#     def test_grad(self):
        
#         np.random.seed(100)
        
#         idx_n2 = np.array([[[0, 0], [1, 1], [2, 3]]]).astype('int32')
        
#         point_cloud_pj = np.random.random((1, 4, 4, 3)).astype('float32')  #  B H W 3  input
#         point_cloud_pj_feature = np.random.random((1, 4, 4, 3)).astype('float32')
        
#         with tf.device('/gpu:1'):

#             xyz = tf.constant(point_cloud_pj_feature[:, :, :, :3])

#             idx_n2 = tf.constant(idx_n2)
            
#             xyz_feature = tf.constant(point_cloud_pj_feature)
            
#             H = 4
#             W = 4
#             npoints = 3
#             kernel_size_H = 3
#             kernel_size_W = 3
#             distance = 0.7

#             K = 8
#             flag_copy = 1

#             # select_xyz_feature, valid_idx, valid_in_dis_idx = fused_conv(xyz_feature, idx_n2, H, W, npoints, kernel_size_H, kernel_size_W, distance)
#             select_xyz_feature, valid_idx, valid_in_dis_idx = fused_conv_between(xyz, xyz_feature, idx_n2, H, W, npoints, kernel_size_H, kernel_size_W, K, flag_copy, distance)

#             # B H' W' M 3
            
#             # print(select_feature.eval())
             
#         with self.test_session() as sess:
#             print("---- Going to compute gradient error")

#             err_thero, err_numer = tf.test.compute_gradient(xyz_feature, (1, 4, 4, 3), select_xyz_feature, (1, 3, 8, 3))

#             np.set_printoptions(threshold=np.inf)

#             error_1 = np.array(err_thero)
#             error_2 = np.array(err_numer)

#             print("err_thero: ",error_1)
#             print("err_numer: ",error_2)
            
#             ret = sess.run(select_xyz_feature)
#             print("origin_xyz: ", xyz.eval())
#             print("output_xyz: ", ret)
            
#             err = tf.test.compute_gradient_error(xyz_feature, (1, 4, 4, 3), select_xyz_feature, (1, 3, 8, 3))
#             print(err)
#             self.assertLess(err, 1e-4) 



if  __name__=='__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

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

    xyz = tf.constant(point_cloud_pj)
    xyz2 = xyz
    
    idx_n2 = tf.constant(idx_n2)

    # random_H = tf.random_shuffle(tf.range(kernel_size_H) - kernel_size_H//2)
    # random_W = tf.random_shuffle(tf.range(kernel_size_W) - kernel_size_W//2)
    random_hw = tf.random_shuffle(tf.range(kernel_size_H * kernel_size_W))


    select_bhw_idx, valid_idx, valid_in_dis_idx, select_mask = fused_conv_random_k(xyz, xyz2, idx_n2, random_hw, H, W, npoints, kernel_size_H, kernel_size_W, K, flag_copy, distance, 1, 1)
    select_bhw_idx = select_bhw_idx[:, :, :, :3]

    select_xyz_feature = tf.gather_nd(point_cloud_pj, select_bhw_idx)
    select_xyz_feature = select_xyz_feature * select_mask

    print(' conv 2d ok ')
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    with tf.Session() as sess:

        now = time.time()
        
        ret = sess.run(select_xyz_feature)
            
        print(time.time() - now)
        print(ret.shape, ret.dtype)

        print ('xyz: ', xyz.eval())
        print ('selected__xyz: ', ret[:, :, :, :3])
