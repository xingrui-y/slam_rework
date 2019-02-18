#include "image_ops.h"
#include "device_array.h"
#include "vector_math.h"
#include <cuda_runtime_api.h>

__global__ void compute_vmap_kernel(const PtrStepSz<float> depth, PtrStep<float4> vmap, float cx, float cy, float inv_fx, float inv_fy)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if(x >= depth.cols || y >= depth.rows)
        return;
        
    float4 v = make_float4(0);
    v.z = depth.ptr(y)[x];
    
    if(v.z > 0.1f)
    {
        v.x = v.z * (x - cx) * inv_fx;
        v.y = v.z * (y - cy) * inv_fy;
        v.w = 1;
    }

    vmap.ptr(y)[x] = v;
}

template<bool synchronize>
void compute_vmap(const DeviceArray2D<float> &depth, DeviceArray2D<float4> &vmap, float cx, float cy, float fx, float fy)
{
    dim3 thread(8, 8);
    dim3 block(div_up(depth.cols, thread.x), div_up(depth.rows, thread.y));

    compute_vmap_kernel<<<block, thread>>>(depth, vmap, cx, cy, 1.0 / fx, 1.0 / fy);

    if(synchronize)
    {
        safe_call(cudaDeviceSynchronize());
        safe_call(cudaGetLastError());
    }
}

__global__ void compute_nmap_kernel(const PtrStepSz<float4> vmap, PtrStep<float4> nmap)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= vmap.cols || y >= vmap.rows)
        return;
        
    nmap.ptr(y)[x] = make_float4(__int_as_float(0x7fffffff));
    
	if (x < 1 || y < 1 || x >= vmap.cols - 1 || y >= vmap.rows - 1)
		return;

	float4 left = vmap.ptr(y)[x - 1];
	float4 right = vmap.ptr(y)[x + 1];
	float4 up = vmap.ptr(y + 1)[x];
	float4 down = vmap.ptr(y - 1)[x];

	if(left == left && right == right && up == up && down == down)
	{
		nmap.ptr(y)[x] = make_float4(normalised(cross(left - right , up - down)), 1.f);
	}
}

void compute_nmap(const DeviceArray2D<float4> &vmap, DeviceArray2D<float4> &nmap)
{
	dim3 block(8, 8);
	dim3 grid(div_up(vmap.cols, block.x), div_up(vmap.rows, block.y));

	compute_nmap_kernel<<<grid, block>>>(vmap, nmap);

	safe_call(cudaDeviceSynchronize());
	safe_call(cudaGetLastError());
}

__global__ void subsample_kernel()
{

}

void subsample()
{

}

__global__ void subsample_mean_kernel()
{

}

void subsample_mean()
{
    
}