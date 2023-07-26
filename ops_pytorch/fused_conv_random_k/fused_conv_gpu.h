#ifndef _FUSE_CONV_GPU_H_
#define _FUSE_CONV_GPU_H_

#include <torch/extension.h>
#include <THC/THC.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

void torch_FusedConvRandomKLauncher(
	torch::Tensor xyz_tensor, 
	torch::Tensor xyz2_tensor,
	torch::Tensor idx_n2_tensor, 
	torch::Tensor random_hw_tensor, 
	int H, 
	int W, 
	int npoints, 
	int kernel_size_H, 
	int kernel_size_W, 
	int K, 
	bool flag_copy, 
	float distance, 
	int stride_h,
	int stride_w, 
	torch::Tensor select_b_idx_tensor,
	torch::Tensor select_h_idx_tensor,
	torch::Tensor select_w_idx_tensor, 
	torch::Tensor valid_idx_tensor, 
	torch::Tensor valid_in_dis_idx_tensor, 
	torch::Tensor select_mask_tensor,
	int small_h,
	int small_w);

void FusedConvRandomKLauncher( 
	int batch_size, 
	int H, 
	int W, 
	int npoints, 
	int kernel_size_H, 
	int kernel_size_W, 
	int K, 
	int flag_copy, 
	float distance, 
	int stride_h,
	int stride_w,
	const float *xyz1, 
	const float *xyz2, 
	const int *idx_n2, 
	const int *random_hw, 
	long *selected_b_idx, 
	long *selected_h_idx, 
	long *selected_w_idx, 
	float *valid_idx, 
	float *valid_in_dis_idx, 
	float *selected_mask,
	int small_h,
	int small_w,
	cudaStream_t stream);

#endif