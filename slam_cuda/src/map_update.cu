#include "device_map_ops.h"
#include "vector_math.h"
#include "cuda_utils.h"
#include <opencv2/cudaarithm.hpp>

namespace slam
{
namespace map
{

template <int threadBlock>
__device__ inline int ComputeOffset(uint element, uint *sum)
{

    __shared__ uint buffer[threadBlock];
    __shared__ uint blockOffset;

    if (threadIdx.x == 0)
        memset(buffer, 0, sizeof(uint) * 16 * 16);
    __syncthreads();

    buffer[threadIdx.x] = element;
    __syncthreads();

    int s1, s2;

    for (s1 = 1, s2 = 1; s1 < threadBlock; s1 <<= 1)
    {
        s2 |= s1;
        if ((threadIdx.x & s2) == s2)
            buffer[threadIdx.x] += buffer[threadIdx.x - s1];
        __syncthreads();
    }

    for (s1 >>= 2, s2 >>= 1; s1 >= 1; s1 >>= 1, s2 >>= 1)
    {
        if (threadIdx.x != threadBlock - 1 && (threadIdx.x & s2) == s2)
            buffer[threadIdx.x + s1] += buffer[threadIdx.x];
        __syncthreads();
    }

    if (threadIdx.x == 0 && buffer[threadBlock - 1] > 0)
        blockOffset = atomicAdd(sum, buffer[threadBlock - 1]);
    __syncthreads();

    int offset;
    if (threadIdx.x == 0)
    {
        if (buffer[threadIdx.x] == 0)
            offset = -1;
        else
            offset = blockOffset;
    }
    else
    {
        if (buffer[threadIdx.x] == buffer[threadIdx.x - 1])
            offset = -1;
        else
            offset = blockOffset + buffer[threadIdx.x - 1];
    }

    return offset;
}

struct MapUpdateDelegate
{
    int width, height;
    MapStruct map_struct;
    float invfx, invfy;
    float fx, fy, cx, cy;
    DeviceMatrix3x4 pose;
    DeviceMatrix3x4 inv_pose;
    cv::cuda::PtrStep<float4> nmap;
    cv::cuda::PtrStep<uchar3> rgb;
    cv::cuda::PtrStep<float> depth;

    __device__ __forceinline__ float2 project_to_image(const float3 &pt) const
    {
        return make_float2(fx * pt.x / pt.z + cx, fy * pt.y / pt.z + cy);
    }

    __device__ __forceinline__ float3 unproject(const int &x, const int &y, const float &z) const
    {
        return make_float3(z * (x - cx) * invfx, z * (y - cy) * invfy, z);
    }

    __device__ __forceinline__ float3 unproject_world(const int &x, const int &y, const float &z) const
    {
        return pose(unproject(x, y, z));
    }

    __device__ __forceinline__ bool is_vertex_visible(float3 pt) const
    {
        pt = inv_pose(pt);
        float2 pt2d = project_to_image(pt);
        return pt2d.x >= 0 && pt2d.y >= 0 &&
               pt2d.x < width && pt2d.y < height &&
               pt.z >= param.zmin_update_ &&
               pt.z <= param.zmax_update_;
    }

    __device__ __forceinline__ bool is_block_visible(const int3 &pos) const
    {
        float scale = param.block_size_metric();
        float3 corner = pos * scale;

        if (is_vertex_visible(corner))
            return true;

        corner.z += scale;
        if (is_vertex_visible(corner))
            return true;

        corner.y += scale;
        if (is_vertex_visible(corner))
            return true;

        corner.x += scale;
        if (is_vertex_visible(corner))
            return true;

        corner.z -= scale;
        if (is_vertex_visible(corner))
            return true;

        corner.y -= scale;
        if (is_vertex_visible(corner))
            return true;

        corner.x -= scale;
        corner.y += scale;
        if (is_vertex_visible(corner))
            return true;

        corner.x += scale;
        corner.y -= scale;
        corner.z += scale;
        if (is_vertex_visible(corner))
            return true;

        return false;
    }

    __device__ __forceinline__ void create_visible_blocks()
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        if (x >= width && y >= height)
            return;

        const float z = depth.ptr(y)[x];
        if (isnan(z) || z < param.zmin_update_ || z > param.zmax_update_)
            return;

        float z_thresh = param.truncation_dist() * 0.5;
        float z_near = min(param.zmax_update_, z - z_thresh);
        float z_far = min(param.zmax_update_, z + z_thresh);
        if (z_near >= z_far)
            return;

        float3 pt_near = unproject_world(x, y, z_near) * param.inverse_voxel_size();
        float3 pt_far = unproject_world(x, y, z_far) * param.inverse_voxel_size();
        float3 dir = pt_far - pt_near;

        float len_dir = norm(dir);
        int num_steps = (int)ceil(2.0 * len_dir);
        dir = dir / (float)(num_steps - 1);

        for (int step = 0; step < num_steps; ++step, pt_near += dir)
            map_struct.create_block(map_struct.voxel_pos_to_block_pos(make_int3(pt_near)));
    }

    template <bool reverse = false>
    __device__ __forceinline__ void update_map_with_image()
    {
        if (blockIdx.x >= param.num_total_hash_entries_ || blockIdx.x >= *map_struct.visible_block_count_)
            return;

        HashEntry &entry = map_struct.visible_block_pos_[blockIdx.x];
        if (entry.ptr_ == -1)
            return;

        int3 block_pos = map_struct.block_pos_to_voxel_pos(entry.pos_);
        float truncation_dist = param.truncation_dist();
        float inv_trunc_dist = 1.0 / truncation_dist;

#pragma unroll
        for (int block_idx_z = 0; block_idx_z < BLOCK_SIZE; ++block_idx_z)
        {
            int3 local_pos = make_int3(threadIdx.x, threadIdx.y, block_idx_z);
            float3 pt = map_struct.voxel_pos_to_world_pt(block_pos + local_pos);
            int2 uv = make_int2(project_to_image(inv_pose(pt)) + make_float2(0.5, 0.5));
            if (uv.x < 0 || uv.y < 0 || uv.x > width - 1 || uv.y > height - 1)
                continue;

            const float z = depth.ptr(uv.y)[uv.x];
            if (isnan(z) || z > param.zmax_update_ || z < param.zmin_update_)
                continue;

            float sdf = z - pt.z;
            if (sdf >= -truncation_dist)
            {
                sdf = fmin(1.0f, sdf * inv_trunc_dist);
                const int local_idx = map_struct.local_pos_to_local_idx(local_pos);
                float3 new_rgb = make_float3(rgb.ptr(uv.y)[uv.x]);
                Voxel &prev = map_struct.voxels_[entry.ptr_ + local_idx];

                if (!reverse)
                {
                    prev.sdf_ = (prev.sdf_ * prev.weight_ + sdf) / (prev.weight_ + 1);
                    prev.rgb_ = make_uchar3((0.2f * new_rgb + 0.8f * make_float3(prev.rgb_)));
                    prev.weight_++;
                }
                else
                {
                    if ((prev.weight_ - 1) != 0)
                    {
                        prev.sdf_ = (prev.sdf_ * prev.weight_ - sdf) / (prev.weight_ - 1);
                        prev.rgb_ = make_uchar3((make_float3(prev.rgb_) - 0.2 * new_rgb) * 1.25f);
                        prev.weight_--;
                    }
                    else
                    {
                        prev.weight_ = 0;
                    }
                }
            }
        }
    }

    __device__ __forceinline__ void check_hash_entry_visibility()
    {
        __shared__ bool bScan;
        if (threadIdx.x == 0)
            bScan = false;
        __syncthreads();
        uint val = 0;

        int x = blockDim.x * blockIdx.x + threadIdx.x;
        if (x < param.num_total_hash_entries_)
        {
            HashEntry &e = map_struct.hash_table_[x];
            if (e.ptr_ != -1)
            {
                if (is_block_visible(e.pos_))
                {
                    bScan = true;
                    val = 1;
                }
            }
        }

        __syncthreads();
        if (bScan)
        {
            int offset = ComputeOffset<1024>(val, map_struct.visible_block_count_);
            if (offset != -1 && x < param.num_total_hash_entries_)
            {
                map_struct.visible_block_pos_[offset] = map_struct.hash_table_[x];
            }
        }
    }
};

__global__ void create_visible_blocks_kernel(MapUpdateDelegate delegate)
{
    delegate.create_visible_blocks();
}

__global__ void check_hash_entry_visibility_kernel(MapUpdateDelegate delegate)
{
    delegate.check_hash_entry_visibility();
}

template <bool reverse = false>
__global__ void update_map_with_image_kernel(MapUpdateDelegate delegate)
{
    delegate.update_map_with_image<reverse>();
}

void update(const cv::cuda::GpuMat depth,
            const cv::cuda::GpuMat image,
            MapStruct &map_struct,
            const Sophus::SE3d &frame_pose,
            const IntrinsicMatrixPtr intrinsic_matrix,
            uint &visible_block_count)
{
    const int cols = depth.cols;
    const int rows = depth.rows;

    map_struct.reset_visible_block_count();
    MapUpdateDelegate delegate;
    delegate.map_struct = map_struct;
    delegate.pose = frame_pose;
    delegate.inv_pose = frame_pose.inverse();
    delegate.fx = intrinsic_matrix->fx;
    delegate.fy = intrinsic_matrix->fy;
    delegate.cx = intrinsic_matrix->cx;
    delegate.cy = intrinsic_matrix->cy;
    delegate.invfx = 1.0 / intrinsic_matrix->fx;
    delegate.invfy = 1.0 / intrinsic_matrix->fy;
    delegate.depth = depth;
    delegate.rgb = image;
    delegate.height = rows;
    delegate.width = cols;

    dim3 thread(8, 8);
    dim3 block(div_up(cols, thread.x), div_up(rows, thread.y));

    create_visible_blocks_kernel<<<block, thread>>>(delegate);
    safe_call(cudaDeviceSynchronize());
    safe_call(cudaGetLastError());

    thread = dim3(MAX_THREAD);
    block = dim3(div_up(state.num_total_hash_entries_, thread.x));

    check_hash_entry_visibility_kernel<<<block, thread>>>(delegate);
    safe_call(cudaDeviceSynchronize());
    safe_call(cudaGetLastError());

    map_struct.get_visible_block_count(visible_block_count);
    if (visible_block_count == 0)
        return;

    thread = dim3(8, 8);
    block = dim3(visible_block_count);

    update_map_with_image_kernel<false><<<block, thread>>>(delegate);

    safe_call(cudaDeviceSynchronize());
    safe_call(cudaGetLastError());
}

} // namespace map
} // namespace slam