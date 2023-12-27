// input: kernel_size(h,w), stride_size(h,w), distance(float), flag_padding, xyz(b,H,W,3), bhw_idx(b,H,W,3)
// output: selected_xyz(b, npoints, h*w, 3), selected_feature(b, npoints, h*w, 3)
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <cstdlib>        // Header file needed to use rand
#include "fused_conv_gpu.h"


__global__ void fused_conv_select_k_gpu(int batch_size, int H, int W, int npoints, int kernel_size_H,
                                        int kernel_size_W, int K, int flag, float distance, int stride_h,
                                        int stride_w, const float *xyz1,
                                        const float *xyz2, const int *idx_n2, const int *random_hw,
                                        long *selected_b_idx, long *selected_h_idx, long *selected_w_idx,
                                        float *valid_idx,
                                        float *valid_in_dis_idx, float *selected_mask, int small_h, int small_w) {

    int batch_index = blockIdx.x; //当前线程块索引
    int index_thread = threadIdx.x;
    int stride_thread = blockDim.x;

    int kernel_total = kernel_size_H * kernel_size_W;        // 一个kernel的大小
    int selected_W_idx = 0, selected_H_idx = 0;

    float dist_square = distance * distance;

    int kernel_half_H = kernel_size_H / 2;
    int kernel_half_W = kernel_size_W / 2;

    xyz1 += batch_index * H * W * 3;                //point cloud of current image
    xyz2 += batch_index * small_h * small_w * 3;
    idx_n2 += batch_index * npoints * 2;                                            // 2d coordinates of central points
    selected_b_idx += batch_index * npoints * K *
                      1; //(b, npoints, k, 1)，			// batch index of K selected points around central points
    selected_h_idx += batch_index * npoints * K * 1; //(b, npoints, k, 1)，
    selected_w_idx += batch_index * npoints * K * 1; //(b, npoints, k, 1)，

    valid_idx += batch_index * npoints * kernel_total *
                 1; //(b, npoints, h*w, 1)，	// coordinate-valid kernel points around central points
    valid_in_dis_idx += batch_index * npoints * kernel_total *
                        1; //(b, npoints, h*w, 1)， // distance-and-corrdinate-valid kernel points around central points

    selected_mask += batch_index * npoints * K * 1; //(b, npoints, h*w, 1)，坐标有效且距离有效的点，含复制的点（重复使用最近邻的点）


    //////////////      Fused  Conv  Between

    for (int current_n = index_thread; current_n < npoints; current_n += stride_thread)  //  output_W circle
    {

        int idx_w[500], idx_h[500];
        float Dist[500];

        for (int ii = 0; ii < 500; ++ii) {
            idx_w[ii] = 0;
            idx_h[ii] = 0;
            Dist[ii] = 1e10f;
        }

        int m_idx = 0;   //  mth point in each kernel
        int num_select = 0; // the number of selected points in each kernel
        int num_valid_idx = 0; // the number of valid points in each kernel

        selected_H_idx = idx_n2[current_n * 2 + 0];    // the  central points H idx of input 2d frame
        selected_W_idx = idx_n2[current_n * 2 + 1];    // the  central points W idx of input 2d frame

        float x_c = xyz1[selected_H_idx * W * 3 + selected_W_idx * 3 + 0];
        float y_c = xyz1[selected_H_idx * W * 3 + selected_W_idx * 3 + 1];
        float z_c = xyz1[selected_H_idx * W * 3 + selected_W_idx * 3 + 2];

        float Dist_c = max((x_c - 0) * (x_c - 0) + (y_c - 0) * (y_c - 0) + (z_c - 0) * (z_c - 0), 1e-10f);

        if (Dist_c <= 1e-10f)    //   not  valid  central  points of xyz1
        {
            continue;

        }

        //  valid  central  points of xyz2

        for (int current_HW_idx = 0;
             current_HW_idx < kernel_total; ++current_HW_idx) //select points in every kernel element
        {

            int kernel_HW_idx = random_hw[current_HW_idx]; //random_hw is random selected elements of kernel coordinations

            // stride is due to the points are in xyz2 which may be a down-sampled point cloud, the scale isn't the same as xyz1
            int kernel_select_H_idx =
                    selected_H_idx / stride_h + kernel_HW_idx / kernel_size_W - kernel_half_H; // random select ？？？
            int kernel_select_W_idx =
                    selected_W_idx / stride_w + kernel_HW_idx % kernel_size_W - kernel_half_W; // random select ？？？


            //TODO: change1
            if (flag & 0x2)  // bit boolean 0x2 == 0b0010
            {
                // only skip the point which is over the boundary along the H axis
                if ((kernel_select_H_idx < 0) ||
                    (kernel_select_H_idx >= small_h)) //  the region of padding points (not valid)
                {
                    ++m_idx;
                    continue;
                }
                // along the width axis, they should be treated as a circle shift
                if (kernel_select_W_idx < 0) {
                    kernel_select_W_idx = small_w + kernel_select_W_idx;   // circle shift
                }

                if (kernel_select_W_idx >= small_w) {
                    kernel_select_W_idx = kernel_select_W_idx - small_w;  // circle shift
                }
            }
            else{
                if ((kernel_select_H_idx < 0) || (kernel_select_H_idx >= small_h) || (kernel_select_W_idx < 0) ||
                    (kernel_select_W_idx >= small_w)) //  the region of padding points (not valid)
                {
                    ++m_idx;
                    continue;
                }

                //if (kernel_select_W_idx < 0)
                //{
                //	kernel_select_W_idx = small_w + kernel_select_W_idx;   ////    cylindrical project???
                //}

                //if (kernel_select_W_idx >= small_w)
                //{
                //	kernel_select_W_idx = kernel_select_W_idx - small_w;  ////    cylindrical project???
                //}
            }


            //  not the padding points

            float x_q = xyz2[kernel_select_H_idx * small_w * 3 + kernel_select_W_idx * 3 + 0];
            float y_q = xyz2[kernel_select_H_idx * small_w * 3 + kernel_select_W_idx * 3 + 1];
            float z_q = xyz2[kernel_select_H_idx * small_w * 3 + kernel_select_W_idx * 3 + 2];

            float Dist_q_0 = x_q * x_q + y_q * y_q + z_q * z_q;

            if (Dist_q_0 <= 1e-10f)  //  not valid xyz2 points
            {
                ++m_idx;
                continue;
            }

            // valid xyz2 points, calculate the distance

            //valid_idx[current_n * kernel_total * 1 + num_valid_idx * 1 + 0] = 1.0;//与有效点编号不一定对的上？
            //++num_valid_idx;

            float Dist_q = max((x_c - x_q) * (x_c - x_q) + (y_c - y_q) * (y_c - y_q) + (z_c - z_q) * (z_c - z_q),
                               1e-10f);

            if (Dist_q > dist_square)  // too far from the central points, regarding as not valid
            {
                ++m_idx;
                continue;
            }

            // selected_bhw_idx[current_n * K * 3 + m_idx * 3 + 0 ] = batch_index;
            // selected_bhw_idx[current_n * K * 3 + m_idx * 3 + 1 ] = kernel_select_H_idx;
            // selected_bhw_idx[current_n * K * 3 + m_idx * 3 + 2 ] = kernel_select_W_idx;
            // selected_mask[current_n * K * 1 + m_idx * 1 + 0 ] = 1.0;

            //valid_in_dis_idx[current_n * kernel_total * 1 + num_select * 1 + 0] = 1.0;

            Dist[m_idx] = Dist_q;
            idx_h[m_idx] = kernel_select_H_idx;
            idx_w[m_idx] = kernel_select_W_idx;

            ++m_idx;
            ++num_select;

            if (num_select >= kernel_total)  //  search all position
                break;

        }

        //?int sort_num = 0;

        for (int s_idx = 0; s_idx < K; ++s_idx)  // knn
        {
            int min_idx = s_idx;  // min_idx idx

            // find the min_idx
            for (int t = s_idx + 1; t < kernel_total; ++t) {
                if (Dist[t] < Dist[min_idx]) {
                    min_idx = t;
                }
            }

            // swap min_idx-th and i-th element
            if (min_idx != s_idx) {
                float tmp_dist = Dist[min_idx];
                int tmp_idx_w = idx_w[min_idx];
                int tmp_idx_h = idx_h[min_idx];

                Dist[min_idx] = Dist[s_idx];
                idx_w[min_idx] = idx_w[s_idx];
                idx_h[min_idx] = idx_h[s_idx];

                Dist[s_idx] = tmp_dist;
                idx_w[s_idx] = tmp_idx_w;
                idx_h[s_idx] = tmp_idx_h;

            }

            // TODO:change2, also a bit boolean 0b0001
            if ((flag & 0x1) && (s_idx == 0)) // copy the first selected point in xyz2 for K times
            {
                for (int k_idx = 0; k_idx < K; ++k_idx) {

                    selected_b_idx[current_n * K + k_idx] = batch_index;
                    selected_h_idx[current_n * K + k_idx] = idx_h[s_idx];
                    selected_w_idx[current_n * K + k_idx] = idx_w[s_idx];
                    selected_mask[current_n * K * 1 + k_idx * 1 + 0] = 1.0;

                }

            }     //  copy done


            if (Dist[s_idx] < 1e10f)  //  whether this is a valid points or not
            {

                selected_b_idx[current_n * K + s_idx] = batch_index;
                selected_h_idx[current_n * K + s_idx] = idx_h[s_idx];
                selected_w_idx[current_n * K + s_idx] = idx_w[s_idx];
                selected_mask[current_n * K * 1 + s_idx * 1 + 0] = 1.0;

            }

        }


    }

}


void FusedConvSelectKLauncher(int batch_size, int H, int W, int npoints, int kernel_size_H,
                              int kernel_size_W, int K, int flag, float distance, int stride_h, int stride_w,
                              const float *xyz1, const float *xyz2, const int *idx_n2, const int *random_hw,
                              long *selected_b_idx, long *selected_h_idx, long *selected_w_idx, float *valid_idx,
                              float *valid_in_dis_idx,
                              float *selected_mask, int small_h, int small_w, cudaStream_t stream) {

    cudaError_t err;

    fused_conv_select_k_gpu<<<batch_size, 512, 0, stream>>>(batch_size, H, W, npoints, kernel_size_H, kernel_size_W, K,
                                                            flag, distance, stride_h, stride_w, xyz1, xyz2, idx_n2,
                                                            random_hw, selected_b_idx, selected_h_idx, selected_w_idx,
                                                            valid_idx, valid_in_dis_idx, selected_mask, small_h,
                                                            small_w);

    //cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
