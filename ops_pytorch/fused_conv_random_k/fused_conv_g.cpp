#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "fused_conv_gpu.h"

extern THCState *state;

//#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
//#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

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
	torch::Tensor selected_b_idx_tensor, 
	torch::Tensor selected_h_idx_tensor,
	torch::Tensor selected_w_idx_tensor,
	torch::Tensor valid_idx_tensor, 
	torch::Tensor valid_in_dis_idx_tensor,
	torch::Tensor selected_mask_tensor,
	int small_h,
	int small_w ){

    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(xyz2_tensor);
	CHECK_INPUT(idx_n2_tensor);
    CHECK_INPUT(random_hw_tensor);
	CHECK_INPUT(selected_b_idx_tensor);
	CHECK_INPUT(selected_h_idx_tensor);
	CHECK_INPUT(selected_w_idx_tensor);
    CHECK_INPUT(valid_idx_tensor);
	CHECK_INPUT(valid_in_dis_idx_tensor);
    CHECK_INPUT(selected_mask_tensor);
	
	const auto batch_size = xyz_tensor.size(0);
	const float *xyz1 = xyz_tensor.data<float>();
	const float *xyz2 = xyz2_tensor.data<float>();
	const int *idx_n2 = idx_n2_tensor.data<int>();
	const int *random_hw = random_hw_tensor.data<int>();
	long *selected_b_idx = selected_b_idx_tensor.data<long>();
	long *selected_h_idx = selected_h_idx_tensor.data<long>();
	long *selected_w_idx = selected_w_idx_tensor.data<long>();
	float *valid_idx = valid_idx_tensor.data<float>();
	float *valid_in_dis_idx = valid_in_dis_idx_tensor.data<float>();
	float *selected_mask = selected_mask_tensor.data<float>();

	//cudaStream_t stream = THCState_getCurrentStream(state);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	
    FusedConvRandomKLauncher(batch_size, H, W, npoints, kernel_size_H, kernel_size_W, K, flag_copy, distance, 
	stride_h, stride_w, xyz1, xyz2, idx_n2, random_hw, selected_b_idx, selected_h_idx, selected_w_idx, valid_idx, valid_in_dis_idx, selected_mask, small_h, small_w, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_random_k",
          &torch_FusedConvRandomKLauncher,
          "torch_FusedConvRandomKLauncher kernel warpper");
}