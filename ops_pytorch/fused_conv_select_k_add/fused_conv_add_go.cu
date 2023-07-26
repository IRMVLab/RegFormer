// input: kernel_size(h,w), stride_size(h,w), distance(float), flag_padding, xyz(b,H,W,3), bhw_idx(b,H,W,3)
// output: selected_xyz(b, npoints, kernel_total, 3), selected_feature(b, npoints, kernel_total, 3)
#include <algorithm>
#include <stdio.h> 
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <cstdlib>		// Header file needed to use rand
#include "fused_conv_add_gpu.h"


__global__ void fused_conv_select_k_add_gpu(int batch_size, int H, int W, int npoints, int kernel_size_H, 
	int kernel_size_W, int K, int flag_copy, float distance, const float *xyz1, const float *xyz2, 	
	const int *idx_n2, const int *random_hw, long *selected_b_idx, long *selected_h_idx, 
	long *selected_w_idx, long *add_b_idx, long *add_h_idx, long *add_w_idx, float *valid_idx, 
	float *valid_in_dis_idx, float *selected_mask, float *add_mask){

	int batch_index = blockIdx.x; //当前线程块索引
    int index_thread = threadIdx.x;
	int stride_thread = blockDim.x;

	int kernel_total = kernel_size_H * kernel_size_W;
	int selected_W_idx = 0, selected_H_idx = 0;

	float dist_square = distance * distance;

	int kernel_half_H = kernel_size_H / 2;
	int kernel_half_W = kernel_size_W / 2;	

	xyz1 += batch_index * H * W * 3;
    xyz2 += batch_index * H * W * 3;
	idx_n2 += batch_index * npoints * 2;	
	selected_b_idx += batch_index * npoints * K * 1 ; //(b, npoints, h*w, 3)，选择点的坐标
	selected_h_idx += batch_index * npoints * K * 1 ; //(b, npoints, h*w, 3)，选择点的坐标
	selected_w_idx += batch_index * npoints * K * 1 ; //(b, npoints, h*w, 3)，选择点的坐标
	add_b_idx += batch_index * H * W * 1 ;
	add_h_idx += batch_index * H * W * 1 ;
	add_w_idx += batch_index * H * W * 1 ;

	valid_idx += batch_index * npoints * kernel_total * 1 ; //(b, npoints, h*w, 1)，坐标有效的点
	valid_in_dis_idx += batch_index * npoints * kernel_total * 1 ; //(b, npoints, h*w, 1)，坐标有效且距离有效的点
	
	selected_mask += batch_index * npoints * K * 1 ; //(b, npoints, h*w, 1)，坐标有效且距离有效的点，含复制的点（重复使用最近邻的点）
	add_mask += batch_index * H * W * 1 ; //(b, H, W, 1)


	//////////////      Fused  Conv  Between

	for (int current_n = index_thread; current_n < npoints; current_n += stride_thread)  //  output_W circle 
	{		

		int idx_w[200], idx_h[200], idx_add_w[200], idx_add_h[200];
		float Dist[200], Dist_2d[200];
	
		for(int ii = 0; ii<200; ++ii)
		{
			idx_w[ii] = 0;
			idx_h[ii] = 0;
			idx_add_w[ii] = 0;
			idx_add_h[ii] = 0;
			Dist[ii] = 1e10f;
			Dist_2d[ii] = 1e10f;
		}
		
		int num_select = 0; // the number of selected points in each kernel
		int num_valid_idx = 0; // the number of valid points in each kernel
		int flag_valid = 0;

		selected_H_idx = idx_n2[current_n * 2 + 0];  // the  central points H idx of input 2d frame
		selected_W_idx = idx_n2[current_n * 2 + 1];  // the  central points W idx of input 2d frame

		float x_c = xyz1[selected_H_idx * W * 3 + selected_W_idx * 3 + 0];
		float y_c = xyz1[selected_H_idx * W * 3 + selected_W_idx * 3 + 1];
		float z_c = xyz1[selected_H_idx * W * 3 + selected_W_idx * 3 + 2];

		float Dist_c = max((x_c-0)*(x_c-0)+(y_c-0)*(y_c-0)+(z_c-0)*(z_c-0), 1e-10f);
		
		if (Dist_c <= 1e-10f)    //   not  valid  central  points of xyz1
		{
			
			for (int current_HW_idx = 0; current_HW_idx < kernel_total; ++current_HW_idx)  // search for the nearest points
			{
				int kernel_HW_idx = random_hw[current_HW_idx];

				int kernel_select_H_idx = selected_H_idx + kernel_HW_idx / kernel_size_W - kernel_half_H; //  select kernel points
				int kernel_select_W_idx = selected_W_idx + kernel_HW_idx % kernel_size_W - kernel_half_W; //  select kernel points 
				

				if ( (kernel_select_H_idx < 0) || (kernel_select_H_idx >= H) || (kernel_select_W_idx < 0) || (kernel_select_W_idx >= W) ) //  the region of padding points (not valid)
				{
					continue;
				}
	

				int H_dist_2d = kernel_select_H_idx - selected_H_idx; // calculate the 2d dist between central points & query points
				int W_dist_2d = kernel_select_W_idx - selected_W_idx;


				// if (kernel_select_W_idx < 0)
				// {
				// 	kernel_select_W_idx = W + kernel_select_W_idx;   ////    cylindrical project
				// }
	
				// if (kernel_select_W_idx >= W)
				// {
				// 	kernel_select_W_idx = kernel_select_W_idx - W;  ////    cylindrical project
				// }
					
	
				//  not the padding points 			
				float x_q = xyz1[kernel_select_H_idx * W * 3 + kernel_select_W_idx * 3 + 0];
				float y_q = xyz1[kernel_select_H_idx * W * 3 + kernel_select_W_idx * 3 + 1];
				float z_q = xyz1[kernel_select_H_idx * W * 3 + kernel_select_W_idx * 3 + 2];
				
				float Dist_q_0 = x_q*x_q + y_q*y_q + z_q*z_q;
	
				if (Dist_q_0 <= 1e-10f)  //  not valid xyz1 points 
				{
					continue;
				}

				Dist_2d[current_HW_idx] = H_dist_2d * H_dist_2d + W_dist_2d * W_dist_2d;  // 2D distance
				idx_add_h[current_HW_idx] = kernel_select_H_idx;
				idx_add_w[current_HW_idx] = kernel_select_W_idx;
				flag_valid = 1;

			}


			if (flag_valid > 0)  //  conduct the copy process
			{

				int min_idx = 0;  // min_idx 
	
				// find the min_idx
				for (int t = 1; t < kernel_total; ++t) 
				{
					if (Dist_2d[t] < Dist_2d[min_idx]) 
					{
						min_idx = t;
					}
				}
	
				// swap min_idx-th and i-th element
				if (min_idx!=0) 
				{
					float tmp_dist = Dist_2d[min_idx];
					int tmp_idx_w = idx_add_w[min_idx];
					int tmp_idx_h = idx_add_h[min_idx];
					
					Dist_2d[min_idx] = Dist_2d[0];
					idx_add_w[min_idx] = idx_add_w[0];
					idx_add_h[min_idx] = idx_add_h[0];
	
					Dist_2d[0] = tmp_dist;
					idx_add_w[0] = tmp_idx_w;
					idx_add_h[0] = tmp_idx_h;
	
				}
				

				add_b_idx[selected_H_idx * W  + selected_W_idx] = batch_index;
				add_h_idx[selected_H_idx * W + selected_W_idx] = idx_add_h[0];
				add_w_idx[selected_H_idx * W  + selected_W_idx] = idx_add_w[0];
				add_mask[selected_H_idx * W * 1 + selected_W_idx * 1 + 0 ] = 1.0;  // get the idx of add points and add mask

				x_c = xyz1[idx_add_h[0] * W * 3 + idx_add_w[0] * 3 + 0];
				y_c = xyz1[idx_add_h[0] * W * 3 + idx_add_w[0] * 3 + 1];
				z_c = xyz1[idx_add_h[0] * W * 3 + idx_add_w[0] * 3 + 2];   // copy the central cordinate
			}

		}



		 //  valid  central  points of xyz1

		

		 //  valid  central  points of xyz2
		
		for (int current_HW_idx = 0; current_HW_idx < kernel_total; ++current_HW_idx) //select points in every kernel element
		{
			
			int kernel_HW_idx = random_hw[current_HW_idx];


			int kernel_select_H_idx = selected_H_idx + kernel_HW_idx / kernel_size_W - kernel_half_H; // random select 
			int kernel_select_W_idx = selected_W_idx + kernel_HW_idx % kernel_size_W - kernel_half_W; // random select 
			
			if ( (kernel_select_H_idx < 0) || (kernel_select_H_idx >= H) || (kernel_select_W_idx < 0) || (kernel_select_W_idx >= W) ) //  the region of padding points (not valid)
			{
				continue;
			}
		
		
			// if (kernel_select_W_idx < 0)
			// {
			// 	kernel_select_W_idx = W + kernel_select_W_idx;   ////    cylindrical project???
			// }

			// if (kernel_select_W_idx >= W)
			// {
			// 	kernel_select_W_idx = kernel_select_W_idx - W;  ////    cylindrical project???
			// }
				
			
				//  not the padding points 
			
			float x_q = xyz2[kernel_select_H_idx * W * 3 + kernel_select_W_idx * 3 + 0];
			float y_q = xyz2[kernel_select_H_idx * W * 3 + kernel_select_W_idx * 3 + 1];
			float z_q = xyz2[kernel_select_H_idx * W * 3 + kernel_select_W_idx * 3 + 2];
			
			float Dist_q_0 = x_q*x_q + y_q*y_q + z_q*z_q;

			if (Dist_q_0 <= 1e-10f)  //  not valid xyz2 points 
			{
				continue;
			}

				// valid xyz2 points, calculate the distance
				
			valid_idx[current_n * kernel_total * 1 + num_valid_idx * 1 + 0 ] = 1.0;//与有效点编号不一定对的上？
			++num_valid_idx;

			float Dist_q = max((x_c-x_q)*(x_c-x_q)+(y_c-y_q)*(y_c-y_q)+(z_c-z_q)*(z_c-z_q), 1e-10f);
			
			if (Dist_q > dist_square)  // too far from the central points, regarding as not valid
			{
				continue;
			}

			// selected_bhw_idx[current_n * K * 3 + m_idx * 3 + 0 ] = batch_index;
			// selected_bhw_idx[current_n * K * 3 + m_idx * 3 + 1 ] = kernel_select_H_idx;
			// selected_bhw_idx[current_n * K * 3 + m_idx * 3 + 2 ] = kernel_select_W_idx;
			// selected_mask[current_n * K * 1 + m_idx * 1 + 0 ] = 1.0;

			valid_in_dis_idx[current_n * kernel_total * 1 + num_select * 1 + 0 ] = 1.0;

			Dist[current_HW_idx] = Dist_q;
			idx_h[current_HW_idx] = kernel_select_H_idx;
			idx_w[current_HW_idx] = kernel_select_W_idx;

			++num_select;
			
			if(num_select >= kernel_total)  //  search all position
				break; 

		}

		//?int sort_num = 0;

		for (int s_idx = 0; s_idx < K; ++s_idx)  // knn  
		{
			int min_idx = s_idx;  // min_idx idx

			// find the min_idx
			for (int t=s_idx + 1; t < kernel_total; ++t) 
			{
				if (Dist[t] < Dist[min_idx]) 
				{
					min_idx = t;
				}
			}

			// swap min_idx-th and i-th element
			if (min_idx!=s_idx) 
			{
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
			
			
			if ((flag_copy == 1) && (s_idx == 0)) // copy the first selected point in xyz2 for K times
			{
				for (int k_idx = 0; k_idx < K; ++ k_idx)   
				{	

					selected_b_idx[current_n * K + k_idx] = batch_index;
					selected_h_idx[current_n * K + k_idx] = idx_h[s_idx];
					selected_w_idx[current_n * K + k_idx] = idx_w[s_idx];
					selected_mask[current_n * K * 1 + k_idx * 1 + 0 ] = 1.0;

				}
			
			}	 //  copy done


			if (Dist[s_idx] < 1e10f)  //  whether this is a valid points or not
			{	

				selected_b_idx[current_n * K + s_idx] = batch_index;
				selected_h_idx[current_n * K + s_idx] = idx_h[s_idx];
				selected_w_idx[current_n * K + s_idx] = idx_w[s_idx];
				selected_mask[current_n * K * 1 + s_idx * 1 + 0 ] = 1.0;

			}
		
		}


	}
		
}





void FusedConvSelectKAddLauncher(int batch_size, int H, int W, int npoints, int kernel_size_H, 
	int kernel_size_W, int K, int flag_copy, float distance, const float *xyz1, const float *xyz2, 	
	const int *idx_n2, const int *random_hw, long *selected_b_idx, long *selected_h_idx, 
	long *selected_w_idx, long *add_b_idx, long *add_h_idx, long *add_w_idx, float *valid_idx, 
	float *valid_in_dis_idx, float *selected_mask, float *add_mask, cudaStream_t stream){
	
	cudaError_t err;

	fused_conv_select_k_add_gpu<<<batch_size,256,0,stream>>>(batch_size, H, W, npoints, kernel_size_H, kernel_size_W, K, flag_copy, distance, xyz1, xyz2, idx_n2, random_hw, selected_b_idx, selected_h_idx, selected_w_idx, add_b_idx, add_h_idx, add_w_idx, valid_idx, valid_in_dis_idx, selected_mask, add_mask);

	//cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }	
}
