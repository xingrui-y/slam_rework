#include "device_map_ops.h"
#include "vector_math.h"
#include "cuda_utils.h"
#include <opencv2/opencv.hpp>

#define RENDERING_BLOCK_SIZE_X 16
#define RENDERING_BLOCK_SIZE_Y 16
#define RENDERING_BLOCK_SUBSAMPLE 8

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

struct RenderingBlockDelegate
{
    int width, height;
    DeviceMatrix3x4 inv_pose;
    float fx, fy, cx, cy;

    uint *rendering_block_count;
    uint *visible_block_count;

    HashEntry *visible_block_pos;
    mutable cv::cuda::PtrStepSz<float> zrange_x;
    mutable cv::cuda::PtrStep<float> zrange_y;
    RenderingBlock *rendering_blocks;

    __device__ __forceinline__ float2 project(const float3 &pt) const
    {
        return make_float2(fx * pt.x / pt.z + cx, fy * pt.y / pt.z + cy);
    }

    __device__ __forceinline__ void atomic_max(float *add, float val) const
    {
        int *address_as_i = (int *)add;
        int old = *address_as_i, assumed;
        do
        {
            assumed = old;
            old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
        } while (assumed != old);
    }

    __device__ __forceinline__ void atomic_min(float *add, float val) const
    {
        int *address_as_i = (int *)add;
        int old = *address_as_i, assumed;
        do
        {
            assumed = old;
            old = atomicCAS(address_as_i, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
        } while (assumed != old);
    }

    __device__ __forceinline__ bool create_rendering_block(const int3 &block_pos, RenderingBlock &block) const
    {
        block.upper_left = make_short2(zrange_x.cols, zrange_x.rows);
        block.lower_right = make_short2(-1, -1);
        block.zrange = make_float2(param.zmax_raycast_, param.zmin_raycast_);

#pragma unroll
        for (int corner = 0; corner < 8; ++corner)
        {
            int3 tmp = block_pos;
            tmp.x += (corner & 1) ? 1 : 0;
            tmp.y += (corner & 2) ? 1 : 0;
            tmp.z += (corner & 4) ? 1 : 0;

            float3 pt3d = tmp * param.block_size_metric();
            pt3d = inv_pose(pt3d);

            float2 pt2d = project(pt3d) / RENDERING_BLOCK_SUBSAMPLE;

            if (block.upper_left.x > floor(pt2d.x))
                block.upper_left.x = (int)floor(pt2d.x);

            if (block.lower_right.x < ceil(pt2d.x))
                block.lower_right.x = (int)ceil(pt2d.x);

            if (block.upper_left.y > floor(pt2d.y))
                block.upper_left.y = (int)floor(pt2d.y);

            if (block.lower_right.y < ceil(pt2d.y))
                block.lower_right.y = (int)ceil(pt2d.y);

            if (block.zrange.x > pt3d.z)
                block.zrange.x = pt3d.z;

            if (block.zrange.y < pt3d.z)
                block.zrange.y = pt3d.z;
        }

        if (block.upper_left.x < 0)
            block.upper_left.x = 0;

        if (block.upper_left.y < 0)
            block.upper_left.y = 0;

        if (block.lower_right.x >= zrange_x.cols)
            block.lower_right.x = zrange_x.cols - 1;

        if (block.lower_right.y >= zrange_x.rows)
            block.lower_right.y = zrange_x.rows - 1;

        if (block.upper_left.x > block.lower_right.x)
            return false;

        if (block.upper_left.y > block.lower_right.y)
            return false;

        if (block.zrange.x < param.zmin_raycast_)
            block.zrange.x = param.zmin_raycast_;

        if (block.zrange.y < param.zmin_raycast_)
            return false;

        return true;
    }

    __device__ __forceinline__ void create_rendering_block_list(int offset, const RenderingBlock &block, int &nx, int &ny) const
    {
        for (int y = 0; y < ny; ++y)
        {
            for (int x = 0; x < nx; ++x)
            {
                if (offset < param.num_max_rendering_blocks_)
                {
                    RenderingBlock &b(rendering_blocks[offset++]);
                    b.upper_left.x = block.upper_left.x + x * RENDERING_BLOCK_SIZE_X;
                    b.upper_left.y = block.upper_left.y + y * RENDERING_BLOCK_SIZE_Y;
                    b.lower_right.x = block.upper_left.x + (x + 1) * RENDERING_BLOCK_SIZE_X;
                    b.lower_right.y = block.upper_left.y + (y + 1) * RENDERING_BLOCK_SIZE_Y;

                    if (b.lower_right.x > block.lower_right.x)
                        b.lower_right.x = block.lower_right.x;

                    if (b.lower_right.y > block.lower_right.y)
                        b.lower_right.y = block.lower_right.y;

                    b.zrange = block.zrange;
                }
            }
        }
    }

    __device__ __forceinline__ void operator()() const
    {
        int x = threadIdx.x + blockDim.x * blockIdx.x;

        bool valid = false;
        uint requiredNoBlocks = 0;
        RenderingBlock block;
        int nx, ny;

        if (x < *visible_block_count && visible_block_pos[x].ptr_ != -1)
        {
            valid = create_rendering_block(visible_block_pos[x].pos_, block);
            float dx = (float)block.lower_right.x - block.upper_left.x + 1;
            float dy = (float)block.lower_right.y - block.upper_left.y + 1;
            nx = __float2int_ru(dx / RENDERING_BLOCK_SIZE_X);
            ny = __float2int_ru(dy / RENDERING_BLOCK_SIZE_Y);

            if (valid)
            {
                requiredNoBlocks = nx * ny;
                uint totalNoBlocks = *rendering_block_count + requiredNoBlocks;
                if (totalNoBlocks >= param.num_max_rendering_blocks_)
                {
                    requiredNoBlocks = 0;
                }
            }
        }

        int offset = ComputeOffset<1024>(requiredNoBlocks, rendering_block_count);
        if (valid && offset != -1 && (offset + requiredNoBlocks) < param.num_max_rendering_blocks_)
            create_rendering_block_list(offset, block, nx, ny);
    }

    __device__ __forceinline__ void fill_rendering_blocks() const
    {
        int x = threadIdx.x;
        int y = threadIdx.y;

        int block = blockIdx.x * 4 + blockIdx.y;
        if (block >= param.num_max_rendering_blocks_)
            return;

        RenderingBlock &b(rendering_blocks[block]);

        int xpos = b.upper_left.x + x;
        if (xpos > b.lower_right.x || xpos >= zrange_x.cols)
            return;

        int ypos = b.upper_left.y + y;
        if (ypos > b.lower_right.y || ypos >= zrange_x.rows)
            return;

        atomic_min(&zrange_x.ptr(ypos)[xpos], b.zrange.x);
        atomic_max(&zrange_y.ptr(ypos)[xpos], b.zrange.y);

        return;
    }
};

__global__ void create_rendering_blocks_kernel(const RenderingBlockDelegate delegate)
{
    delegate();
}

__global__ void split_and_fill_rendering_blocks_kernel(const RenderingBlockDelegate delegate)
{
    delegate.fill_rendering_blocks();
}

void create_rendering_blocks(MapStruct map_struct,
                             cv::cuda::GpuMat &zrange_x,
                             cv::cuda::GpuMat &zrange_y,
                             const Sophus::SE3d &frame_pose,
                             const IntrinsicMatrix intrinsic_matrix)
{
    uint visible_block_count;
    map_struct.get_visible_block_count(visible_block_count);
    if (visible_block_count == 0)
        return;

    const int cols = zrange_x.cols;
    const int rows = zrange_y.rows;

    zrange_x.setTo(cv::Scalar(100.f));
    zrange_y.setTo(cv::Scalar(0));
    map_struct.reset_rendering_block_count();

    RenderingBlockDelegate delegate;

    delegate.width = cols;
    delegate.height = rows;
    delegate.inv_pose = frame_pose.inverse();
    delegate.zrange_x = zrange_x;
    delegate.zrange_y = zrange_y;
    delegate.fx = intrinsic_matrix.fx;
    delegate.fy = intrinsic_matrix.fy;
    delegate.cx = intrinsic_matrix.cx;
    delegate.cy = intrinsic_matrix.cy;
    delegate.visible_block_pos = map_struct.visible_block_pos_;
    delegate.visible_block_count = map_struct.visible_block_count_;
    delegate.rendering_block_count = map_struct.rendering_block_count;
    delegate.rendering_blocks = map_struct.rendering_blocks;

    dim3 thread = dim3(MAX_THREAD);
    dim3 block = dim3(div_up(visible_block_count, thread.x));

    create_rendering_blocks_kernel<<<block, thread>>>(delegate);

    uint rendering_block_count;
    map_struct.get_rendering_block_count(rendering_block_count);
    if (rendering_block_count == 0)
        return;

    thread = dim3(RENDERING_BLOCK_SIZE_X, RENDERING_BLOCK_SIZE_Y);
    block = dim3((uint)ceil((float)rendering_block_count / 4), 4);

    split_and_fill_rendering_blocks_kernel<<<block, thread>>>(delegate);
}

struct MapRenderingDelegate
{
    int width, height;
    MapStruct map_struct;
    mutable cv::cuda::PtrStep<float4> vmap;
    mutable cv::cuda::PtrStep<float4> nmap;
    cv::cuda::PtrStepSz<float> zrange_x;
    cv::cuda::PtrStepSz<float> zrange_y;
    float invfx, invfy, cx, cy;
    DeviceMatrix3x4 pose, inv_pose;

    __device__ __forceinline__ float read_sdf(const float3 &pt3d, bool &valid)
    {
        Voxel *voxel = nullptr;
        map_struct.find_voxel(make_int3(pt3d), voxel);
        if (voxel != nullptr && voxel->get_weight() != 0)
        {
            valid = true;
            return voxel->get_sdf();
        }
        else
        {
            valid = false;
            return nanf("0x7fffff");
        }
    }

    __device__ __forceinline__ float read_sdf_interped(const float3 &pt, bool &valid)
    {
        float3 xyz = pt - floor(pt);
        float sdf[2], result[4];
        bool valid_pt;

        sdf[0] = read_sdf(pt, valid_pt);
        sdf[1] = read_sdf(pt + make_float3(1, 0, 0), valid);
        valid_pt &= valid;
        result[0] = (1.0f - xyz.x) * sdf[0] + xyz.x * sdf[1];

        sdf[0] = read_sdf(pt + make_float3(0, 1, 0), valid);
        valid_pt &= valid;
        sdf[1] = read_sdf(pt + make_float3(1, 1, 0), valid);
        valid_pt &= valid;
        result[1] = (1.0f - xyz.x) * sdf[0] + xyz.x * sdf[1];
        result[2] = (1.0f - xyz.y) * result[0] + xyz.y * result[1];

        sdf[0] = read_sdf(pt + make_float3(0, 0, 1), valid);
        valid_pt &= valid;
        sdf[1] = read_sdf(pt + make_float3(1, 0, 1), valid);
        valid_pt &= valid;
        result[0] = (1.0f - xyz.x) * sdf[0] + xyz.x * sdf[1];

        sdf[0] = read_sdf(pt + make_float3(0, 1, 1), valid);
        valid_pt &= valid;
        sdf[1] = read_sdf(pt + make_float3(1, 1, 1), valid);
        valid_pt &= valid;
        result[1] = (1.0f - xyz.x) * sdf[0] + xyz.x * sdf[1];
        result[3] = (1.0f - xyz.y) * result[0] + xyz.y * result[1];
        valid = valid_pt;
        return (1.0f - xyz.z) * result[2] + xyz.z * result[3];
    }

    // __device__ __forceinline__ bool read_normal_approximate(const float3 &pt, float3 &n)
    // {

    //     bool valid;
    //     float sdf[6];
    //     sdf[0] = read_sdf_interped(pt + make_float3(1, 0, 0), valid);
    //     if (isnan(sdf[0]) || sdf[0] == 1.0f || !valid)
    //         return false;

    //     sdf[1] = read_sdf_interped(pt + make_float3(-1, 0, 0), valid);
    //     if (isnan(sdf[1]) || sdf[1] == 1.0f || !valid)
    //         return false;

    //     sdf[2] = read_sdf_interped(pt + make_float3(0, 1, 0), valid);
    //     if (isnan(sdf[2]) || sdf[2] == 1.0f || !valid)
    //         return false;

    //     sdf[3] = read_sdf_interped(pt + make_float3(0, -1, 0), valid);
    //     if (isnan(sdf[3]) || sdf[3] == 1.0f || !valid)
    //         return false;

    //     sdf[4] = read_sdf_interped(pt + make_float3(0, 0, 1), valid);
    //     if (isnan(sdf[4]) || sdf[4] == 1.0f || !valid)
    //         return false;

    //     sdf[5] = read_sdf_interped(pt + make_float3(0, 0, -1), valid);
    //     if (isnan(sdf[5]) || sdf[5] == 1.0f || !valid)
    //         return false;

    //     n = make_float3(sdf[0] - sdf[1], sdf[2] - sdf[3], sdf[4] - sdf[5]);
    //     n = normalised(inv_pose.rotate(n));
    //     return true;
    // }

    __device__ __forceinline__ float3 unproject(const int &x, const int &y, const float &z) const
    {
        return make_float3((x - cx) * invfx * z, (y - cy) * invfy * z, z);
    }

    __device__ __forceinline__ void operator()()
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        if (x >= width || y >= height)
            return;

        vmap.ptr(y)[x] = make_float4(__int_as_float(0x7fffffff));
        // nmap.ptr(y)[x] = make_float4(__int_as_float(0x7fffffff));

        // int2 local_id;
        // local_id.x = __float2int_rd((float)x / 8);
        // local_id.y = __float2int_rd((float)y / 8);

        // float2 zrange;
        // zrange.x = zrange_x.ptr(local_id.y)[local_id.x];
        // zrange.y = zrange_y.ptr(local_id.y)[local_id.x];
        // if (zrange.y < 1e-3 || zrange.x < 1e-3 || isnan(zrange.x) || isnan(zrange.y))
        //     return;

        float sdf = 1.0f;
        float last_sdf;

        float3 pt = unproject(x, y, 0.3);
        float dist_s = norm(pt) * param.inverse_voxel_size();
        float3 block_s = pose(pt) * param.inverse_voxel_size();

        pt = unproject(x, y, 1.5);
        float dist_e = norm(pt) * param.inverse_voxel_size();
        float3 block_e = pose(pt) * param.inverse_voxel_size();

        float3 dir = normalised(block_e - block_s);
        float3 result = block_s;

        bool valid_sdf = false;
        bool found_pt = false;
        float step;

        while (dist_s < dist_e)
        {
            last_sdf = sdf;
            sdf = read_sdf(result, valid_sdf);

            if (sdf <= 0.5f && sdf >= -0.5f)
                sdf = read_sdf_interped(result, valid_sdf);

            if (sdf <= 0.0f)
                break;

            if (sdf >= 0.f && last_sdf < 0.f)
                return;

            if (valid_sdf)
                step = max(sdf * param.raycast_step_scale(), 1.0f);
            else
                step = 2;

            result += step * dir;
            dist_s += step;
        }

        if (sdf <= 0.0f)
        {
            step = sdf * param.raycast_step_scale();
            result += step * dir;

            sdf = read_sdf_interped(result, valid_sdf);

            step = sdf * param.raycast_step_scale();
            result += step * dir;

            // sdf = read_sdf_interped(result, valid_sdf);
            // if (valid_sdf && sdf < 0.05f && sdf > -0.05f)
            if (valid_sdf)
                found_pt = true;
        }

        if (found_pt)
        {
            // float3 normal;
            // if (read_normal_approximate(result, normal))
            // {
            result = inv_pose(result * param.voxel_size_);
            vmap.ptr(y)[x] = make_float4(result, 1.0);
            //     nmap.ptr(y)[x] = make_float4(normal, 1.0);
            // }
        }
    }
    __device__ __forceinline__ uchar3 read_colour(float3 pt3d, bool &valid)
    {
        Voxel *voxel = NULL;
        map_struct.find_voxel(make_int3(pt3d), voxel);
        if (voxel && voxel->get_weight() != 0)
        {
            valid = true;
            return voxel->rgb_;
        }
        else
        {
            valid = false;
            return make_uchar3(0);
        }
    }

    __device__ __forceinline__ uchar3 read_colour_interpolated(float3 pt, bool &valid)
    {
        float3 xyz = pt - floor(pt);
        uchar3 sdf[2];
        float3 result[4];
        bool valid_pt;

        sdf[0] = read_colour(pt, valid_pt);
        sdf[1] = read_colour(pt + make_float3(1, 0, 0), valid);
        valid_pt &= valid;
        result[0] = (1.0f - xyz.x) * sdf[0] + xyz.x * sdf[1];

        sdf[0] = read_colour(pt + make_float3(0, 1, 0), valid);
        valid_pt &= valid;
        sdf[1] = read_colour(pt + make_float3(1, 1, 0), valid);
        valid_pt &= valid;
        result[1] = (1.0f - xyz.x) * sdf[0] + xyz.x * sdf[1];
        result[2] = (1.0f - xyz.y) * result[0] + xyz.y * result[1];

        sdf[0] = read_colour(pt + make_float3(0, 0, 1), valid);
        valid_pt &= valid;
        sdf[1] = read_colour(pt + make_float3(1, 0, 1), valid);
        valid_pt &= valid;
        result[0] = (1.0f - xyz.x) * sdf[0] + xyz.x * sdf[1];

        sdf[0] = read_colour(pt + make_float3(0, 1, 1), valid);
        valid_pt &= valid;
        sdf[1] = read_colour(pt + make_float3(1, 1, 1), valid);
        valid_pt &= valid;
        result[1] = (1.0f - xyz.x) * sdf[0] + xyz.x * sdf[1];
        result[3] = (1.0f - xyz.y) * result[0] + xyz.y * result[1];
        valid = valid_pt;
        return make_uchar3((1.0f - xyz.z) * result[2] + xyz.z * result[3]);
    }
    cv::cuda::PtrStep<uchar3> image;
    __device__ __forceinline__ void raycast_with_colour()
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        if (x >= width || y >= height)
            return;

        vmap.ptr(y)[x] = make_float4(__int_as_float(0x7fffffff));
        image.ptr(y)[x] = make_uchar3(255);
        // nmap.ptr(y)[x] = make_float4(__int_as_float(0x7fffffff));

        // int2 local_id;
        // local_id.x = __float2int_rd((float)x / 8);
        // local_id.y = __float2int_rd((float)y / 8);

        // float2 zrange;
        // zrange.x = zrange_x.ptr(local_id.y)[local_id.x];
        // zrange.y = zrange_y.ptr(local_id.y)[local_id.x];
        // if (zrange.y < 1e-3 || zrange.x < 1e-3 || isnan(zrange.x) || isnan(zrange.y))
        //     return;

        float sdf = 1.0f;
        float last_sdf;

        float3 pt = unproject(x, y, 0.3);
        float dist_s = norm(pt) * param.inverse_voxel_size();
        float3 block_s = pose(pt) * param.inverse_voxel_size();

        pt = unproject(x, y, 1.5);
        float dist_e = norm(pt) * param.inverse_voxel_size();
        float3 block_e = pose(pt) * param.inverse_voxel_size();

        float3 dir = normalised(block_e - block_s);
        float3 result = block_s;

        bool valid_sdf = false;
        bool found_pt = false;
        float step;

        while (dist_s < dist_e)
        {
            last_sdf = sdf;
            sdf = read_sdf(result, valid_sdf);

            if (sdf <= 0.5f && sdf >= -0.5f)
                sdf = read_sdf_interped(result, valid_sdf);

            if (sdf <= 0.0f)
                break;

            if (sdf >= 0.f && last_sdf < 0.f)
                return;

            if (valid_sdf)
                step = max(sdf * param.raycast_step_scale(), 1.0f);
            else
                step = 2;

            result += step * dir;
            dist_s += step;
        }

        if (sdf <= 0.0f)
        {
            step = sdf * param.raycast_step_scale();
            result += step * dir;

            sdf = read_sdf_interped(result, valid_sdf);

            step = sdf * param.raycast_step_scale();
            result += step * dir;

            // sdf = read_sdf_interped(result, valid_sdf);
            // if (valid_sdf && sdf < 0.05f && sdf > -0.05f)
            if (valid_sdf)
                found_pt = true;
        }

        if (found_pt)
        {
            // float3 normal;
            // if (read_normal_approximate(result, normal))
            // {

            // auto rgb = read_colour_interpolated(result, valid_sdf);
            auto rgb = read_colour(result, valid_sdf);
            if (!valid_sdf)
                return;

            result = inv_pose(result * param.voxel_size_);
            vmap.ptr(y)[x] = make_float4(result, 1.0);
            image.ptr(y)[x] = rgb;
            //     nmap.ptr(y)[x] = make_float4(normal, 1.0);
            // }
        }
    }
};

__global__ void __launch_bounds__(32, 16) raycast_kernel(MapRenderingDelegate delegate)
{
    delegate();
}

__global__ void __launch_bounds__(32, 16) raycast_with_colour_kernel(MapRenderingDelegate delegate)
{
    delegate.raycast_with_colour();
}

void raycast(MapStruct map_struct,
             cv::cuda::GpuMat vmap,
             cv::cuda::GpuMat nmap,
             cv::cuda::GpuMat zrange_x,
             cv::cuda::GpuMat zrange_y,
             const Sophus::SE3d &pose,
             const IntrinsicMatrix intrinsic_matrix)
{
    const int cols = vmap.cols;
    const int rows = vmap.rows;

    MapRenderingDelegate delegate;

    delegate.width = cols;
    delegate.height = rows;
    delegate.map_struct = map_struct;
    delegate.vmap = vmap;
    delegate.nmap = nmap;
    delegate.zrange_x = zrange_x;
    delegate.zrange_y = zrange_y;
    delegate.invfx = intrinsic_matrix.invfx;
    delegate.invfy = intrinsic_matrix.invfy;
    delegate.cx = intrinsic_matrix.cx;
    delegate.cy = intrinsic_matrix.cy;
    delegate.pose = pose;
    delegate.inv_pose = pose.inverse();

    dim3 thread(4, 8);
    dim3 block(div_up(cols, thread.x), div_up(rows, thread.y));

    raycast_kernel<<<block, thread>>>(delegate);
}

void raycast_with_colour(MapStruct map_struct,
                         cv::cuda::GpuMat vmap,
                         cv::cuda::GpuMat nmap,
                         cv::cuda::GpuMat image,
                         cv::cuda::GpuMat zrange_x,
                         cv::cuda::GpuMat zrange_y,
                         const Sophus::SE3d &pose,
                         const IntrinsicMatrix intrinsic_matrix)
{
    const int cols = vmap.cols;
    const int rows = vmap.rows;

    MapRenderingDelegate delegate;

    delegate.width = cols;
    delegate.height = rows;
    delegate.map_struct = map_struct;
    delegate.vmap = vmap;
    delegate.nmap = nmap;
    delegate.image = image;
    delegate.zrange_x = zrange_x;
    delegate.zrange_y = zrange_y;
    delegate.invfx = intrinsic_matrix.invfx;
    delegate.invfy = intrinsic_matrix.invfy;
    delegate.cx = intrinsic_matrix.cx;
    delegate.cy = intrinsic_matrix.cy;
    delegate.pose = pose;
    delegate.inv_pose = pose.inverse();

    dim3 thread(4, 8);
    dim3 block(div_up(cols, thread.x), div_up(rows, thread.y));

    raycast_with_colour_kernel<<<block, thread>>>(delegate);
}

} // namespace map
} // namespace slam
