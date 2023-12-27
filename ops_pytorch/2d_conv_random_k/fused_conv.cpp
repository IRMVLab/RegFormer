#include <cstdio>
#include <ctime>
#include <cstring> // memset
#include <cstdlib> // rand, RAND_MAX
#include <cmath> // sqrtf
#include <math.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>
using namespace tensorflow;


REGISTER_OP("FusedConvRandomK")
	.Attr("H: int")
	.Attr("W: int")
	.Attr("npoints: int")	
	.Attr("kernel_size_H: int")
	.Attr("kernel_size_W: int")
	.Attr("K: int")
	.Attr("flag_copy: int")
	.Attr("distance: float")
	.Attr("stride_h: int")
	.Attr("stride_w: int")
	.Input("xyz1: float32")//(batch_size,h,w,3) central points
	.Input("xyz2: float32")//(batch_size,h,w,3) queried points
	.Input("idx_n2: int32")//(batch_size, n, 2)
	.Input("random_hw: int32")//(kernel_h * kernel_w)  ##################################################  1 dim
	.Output("selected_bhw_idx: int32")//(batch_size, npoints, K, 3)
	.Output("selected_valid_idx: float32")//(batch_size, npoints, K, 1)
	.Output("selected_valid_in_dis_idx: float32")//(batch_size, npoints, K, 1)
	.Output("selected_mask: float32")//(batch_size, npoints, K, 1)
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
		::tensorflow::shape_inference::ShapeHandle dims1; // (batch_size, H, W, 3)
		c->WithRank(c->input(1), 4, &dims1);
		
		int H, W, npoints, kernel_size_H, kernel_size_W, K, flag_copy;
		float distance;
		int stride_h, stride_w;

		TF_RETURN_IF_ERROR(c->GetAttr("H", &H));
		TF_RETURN_IF_ERROR(c->GetAttr("W", &W));
		TF_RETURN_IF_ERROR(c->GetAttr("npoints", &npoints));
		TF_RETURN_IF_ERROR(c->GetAttr("kernel_size_H", &kernel_size_H));
		TF_RETURN_IF_ERROR(c->GetAttr("kernel_size_W", &kernel_size_W));
		TF_RETURN_IF_ERROR(c->GetAttr("K", &K));
		TF_RETURN_IF_ERROR(c->GetAttr("flag_copy", &flag_copy));
		TF_RETURN_IF_ERROR(c->GetAttr("distance", &distance));
		TF_RETURN_IF_ERROR(c->GetAttr("stride_h", &stride_h));
		TF_RETURN_IF_ERROR(c->GetAttr("stride_w", &stride_w));
 

		::tensorflow::shape_inference::ShapeHandle output_bhw_idx = c->MakeShape({ c->Dim(dims1, 0), npoints, K, 3 }); // b n k c+3
		::tensorflow::shape_inference::ShapeHandle output_valid_idx = c->MakeShape({ c->Dim(dims1, 0), npoints, kernel_size_H * kernel_size_W, 1 });
		::tensorflow::shape_inference::ShapeHandle output_valid_in_dis_idx = c->MakeShape({ c->Dim(dims1, 0), npoints, kernel_size_H * kernel_size_W, 1 });
		::tensorflow::shape_inference::ShapeHandle output_mask = c->MakeShape({ c->Dim(dims1, 0), npoints, K, 1 });
		c->set_output(0, output_bhw_idx);
		c->set_output(1, output_valid_idx);
		c->set_output(2, output_valid_in_dis_idx);
		c->set_output(3, output_mask);
		return Status::OK();
	});





//////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void FusedConvRandomKLauncher(int batch_size, int H, int W, int npoints, int kernel_size_H, int kernel_size_W, int K, int flag_copy, float distance, int stride_h, int stride_w, const float *xyz1, const float *xyz2, const int *idx_n2, const int *random_hw, int *selected_bhw_idx, float *valid_idx, float *valid_in_dis_idx, float *selected_mask, int small_h, int small_w);

class FusedConvRandomKGpuOp : public OpKernel {
public:
	explicit FusedConvRandomKGpuOp(OpKernelConstruction* context) : OpKernel(context) {
		OP_REQUIRES_OK(context, context->GetAttr("npoints", &npoints_));
	        OP_REQUIRES(context, npoints_ > 0, errors::InvalidArgument("FusedConv expects positive npoints"));
		
		OP_REQUIRES_OK(context, context->GetAttr("kernel_size_H", &kernel_size_H_));
	        OP_REQUIRES(context, kernel_size_H_ > 0, errors::InvalidArgument("FusedConv expects positive kernel_size_H"));
		
		OP_REQUIRES_OK(context, context->GetAttr("kernel_size_W", &kernel_size_W_));
	        OP_REQUIRES(context, kernel_size_W_ > 0, errors::InvalidArgument("FusedConv expects positive kernel_size_W"));
		
		OP_REQUIRES_OK(context, context->GetAttr("K", &K_));
	        OP_REQUIRES(context, K_ > 0, errors::InvalidArgument("FusedConv expects positive K"));

		OP_REQUIRES_OK(context, context->GetAttr("flag_copy", &flag_copy_));
	        OP_REQUIRES(context, flag_copy_ > -1, errors::InvalidArgument("FusedConv expects 0 OR 1 flag_copy"));
		
		OP_REQUIRES_OK(context, context->GetAttr("distance", &distance_));
	        OP_REQUIRES(context, distance_ > 0, errors::InvalidArgument("FusedConv expects positive distance"));

		OP_REQUIRES_OK(context, context->GetAttr("stride_h", &stride_h_));
	        OP_REQUIRES(context, stride_h_ > 0, errors::InvalidArgument("FusedConv expects positive stride_h"));

		OP_REQUIRES_OK(context, context->GetAttr("stride_w", &stride_w_));
	        OP_REQUIRES(context, stride_w_ > 0, errors::InvalidArgument("FusedConv expects positive stride_w"));
		
	}

	void Compute(OpKernelContext* context) override {
			
		const Tensor& xyz1_tensor = context->input(0);
		OP_REQUIRES(context, xyz1_tensor.dims() == 4 && xyz1_tensor.shape().dim_size(3) == 3, errors::InvalidArgument("FusedConvRandomK expects (batch_size, H, W, 3) xyz1 shape."));
		int batch_size = xyz1_tensor.shape().dim_size(0);
		int H = xyz1_tensor.shape().dim_size(1);
		int W = xyz1_tensor.shape().dim_size(2);

		int H2 = ceil(H / double(stride_h_));
		int W2 = ceil(W / double(stride_w_));
		// std::cout << H << " " << H2 << std::endl;
		// std::cout << W << " " << W2 << std::endl;
		const Tensor& xyz2_tensor = context->input(1);
		OP_REQUIRES(context, xyz2_tensor.dims() == 4 && xyz2_tensor.shape().dim_size(1) == H2, errors::InvalidArgument("FusedConvRandomK expects (batch_size, H/stride_h, W/stride_w, 3) xyz2 shape."));

		const Tensor& idx_n2_tensor = context->input(2);
		OP_REQUIRES(context, idx_n2_tensor.shape().dim_size(2) == 2 && idx_n2_tensor.shape().dim_size(1)==npoints_, errors::InvalidArgument("FusedConv expects (batch_size, npoints, 2) idx_n2 shape."));

		const Tensor& random_hw_tensor = context->input(3);
		OP_REQUIRES(context, random_hw_tensor.shape().dim_size(0) == kernel_size_H_ * kernel_size_W_, errors::InvalidArgument("FusedConv expects (kernel_size_h * kernel_size_w) random_hw shape."));


		Tensor *selected_bhw_idx_tensor = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{batch_size, npoints_, K_, 3}, &selected_bhw_idx_tensor));

		Tensor *valid_idx_tensor = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{batch_size, npoints_, kernel_size_H_ * kernel_size_W_, 1}, &valid_idx_tensor));

		Tensor *valid_in_dis_idx_tensor = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{batch_size, npoints_, kernel_size_H_ * kernel_size_W_, 1}, &valid_in_dis_idx_tensor));

		Tensor *selected_mask_tensor = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape{batch_size, npoints_, K_, 1}, &selected_mask_tensor));


		auto xyz1_flat = xyz1_tensor.flat<float>();
		const float *xyz1 = &(xyz1_flat(0));

		auto xyz2_flat = xyz2_tensor.flat<float>();
		const float *xyz2 = &(xyz2_flat(0));

		auto idx_n2_flat = idx_n2_tensor.flat<int>();
		const int *idx_n2 = &(idx_n2_flat(0));

		auto random_hw_flat = random_hw_tensor.flat<int>();
		const int *random_hw = &(random_hw_flat(0));
		
		
		auto selected_bhw_idx_flat = selected_bhw_idx_tensor->flat<int>();
		int *selected_bhw_idx = &(selected_bhw_idx_flat(0));
        cudaMemset(selected_bhw_idx, 0, sizeof(int) * batch_size * npoints_ * K_ * 3);

		auto valid_idx_flat = valid_idx_tensor->flat<float>();
		float *valid_idx = &(valid_idx_flat(0));
        cudaMemset(valid_idx, 0, sizeof(float) * batch_size * npoints_ * kernel_size_H_ * kernel_size_W_ * 1);

		auto valid_in_dis_idx_flat = valid_in_dis_idx_tensor->flat<float>();
		float *valid_in_dis_idx = &(valid_in_dis_idx_flat(0));
        cudaMemset(valid_in_dis_idx, 0, sizeof(float) * batch_size * npoints_ * kernel_size_H_ * kernel_size_W_ * 1);

		auto selected_mask_flat = selected_mask_tensor->flat<float>();
		float *selected_mask = &(selected_mask_flat(0));
        cudaMemset(selected_mask, 0, sizeof(float) * batch_size * npoints_ * K_ * 1);


		FusedConvRandomKLauncher(batch_size, H, W, npoints_, kernel_size_H_, kernel_size_W_, K_, flag_copy_, distance_, stride_h_, stride_w_, xyz1, xyz2, idx_n2, random_hw, selected_bhw_idx, valid_idx, valid_in_dis_idx, selected_mask, H2, W2);
	}
	private:
        int kernel_size_H_, kernel_size_W_, K_, flag_copy_, npoints_;
		float distance_;
		int stride_h_, stride_w_;
};
REGISTER_KERNEL_BUILDER(Name("FusedConvRandomK").Device(DEVICE_GPU), FusedConvRandomKGpuOp);

