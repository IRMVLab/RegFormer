#ifndef _FUSE_CONV_ADD_GPU_H_
#define _FUSE_CONV_ADD_GPU_H_

#include <torch/extension.h>
#include <THC/THC.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

void torch_FusedConvSelectKAddLauncher(
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
	torch::Tensor selected_b_idx_tensor, 
	torch::Tensor selected_h_idx_tensor,
	torch::Tensor selected_w_idx_tensor,
	torch::Tensor add_b_idx_tensor, 
	torch::Tensor add_h_idx_tensor,
	torch::Tensor add_w_idx_tensor,
	torch::Tensor valid_idx_tensor, 
	torch::Tensor valid_in_dis_idx_tensor,
	torch::Tensor selected_mask_tensor,
	torch::Tensor add_mask_tensor);

void FusedConvSelectKAddLauncher( 
	int batch_size, 
	int H, 
	int W, 
	int npoints, 
	int kernel_size_H, 
	int kernel_size_W, 
	int K, 
	int flag_copy, 
	float distance, 
	const float *xyz1, 
	const float *xyz2, 
	const int *idx_n2, 
	const int *random_hw, 
	long *selected_b_idx, 
	long *selected_h_idx, 
	long *selected_w_idx, 
	long *add_b_idx, 
	long *add_h_idx, 
	long *add_w_idx,
	float *valid_idx, 
	float *valid_in_dis_idx, 
	float *selected_mask,
	float *add_mask,
	cudaStream_t stream);

#endif