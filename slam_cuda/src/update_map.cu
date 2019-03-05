#include "device_map.h"
#include "device_scan.h"
#include "safe_call.h"
#include "vector_math.h"

__constant__ float2 depth_range;

class MapUpdateFunctor
{
  public:
    MapUpdateFunctor(const int &cols, const int &rows);

    __device__ __forceinline__ float fx() const;
    __device__ __forceinline__ float fy() const;
    __device__ __forceinline__ float cx() const;
    __device__ __forceinline__ float cy() const;
    __device__ __forceinline__ float invfx() const;
    __device__ __forceinline__ float invfy() const;
    __device__ __forceinline__ float2 project(const float3 &pt3d) const;
    __device__ __forceinline__ float3 unproject(const int &x, const int &y, const float &z) const;
    __device__ __forceinline__ float3 unproject_world(const int &x, const int &y, const float &z) const;
    __device__ __forceinline__ bool check_vertex_visibility(const float3 &pt3d) const;
    __device__ __forceinline__ bool check_block_visibility(const int3 &pos) const;

    template <bool fuse = true>
    __device__ void operator()() const;

    int width, height;
    float *intrinsics;
    Matrix3x4 frame_pose;
};

template <bool fuse = true>
__global__ void register_depth_map_kernel(const MapUpdateFunctor func)
{
    func.operator()<fuse>();
}

template <bool fuse = true>
void register_depth_map(const cv::cuda::GpuMat &depth_map,
                        const cv::cuda::GpuMat &image,
                        const Sophus::SE3d &frame_pose)
{
    MapUpdateFunctor func(640, 480);
    register_depth_map_kernel<fuse><<<1, 1>>>(func);
}

//==================================================================
// Implementations
//==================================================================
MapUpdateFunctor::MapUpdateFunctor(const int &cols, const int &rows) : width(cols), height(rows)
{
}

template <bool fuse>
__device__ void MapUpdateFunctor::operator()() const
{
}

__device__ __forceinline__ float MapUpdateFunctor::fx() const
{
    return intrinsics[0];
}

__device__ __forceinline__ float MapUpdateFunctor::fy() const
{
    return intrinsics[1];
}

__device__ __forceinline__ float MapUpdateFunctor::cx() const
{
    return intrinsics[2];
}

__device__ __forceinline__ float MapUpdateFunctor::cy() const
{
    return intrinsics[3];
}

__device__ __forceinline__ float MapUpdateFunctor::invfx() const
{
    return intrinsics[4];
}

__device__ __forceinline__ float MapUpdateFunctor::invfy() const
{
    return intrinsics[5];
}

__device__ __forceinline__ float2 MapUpdateFunctor::project(const float3 &pt3d) const
{
    return make_float2(fx() * pt3d.x / pt3d.z + cx(), fy() * pt3d.y / pt3d.z + cy());
}

__device__ __forceinline__ float3 MapUpdateFunctor::unproject(const int &x, const int &y, const float &z) const
{
    return make_float3(z * invfx() * (x - cx()), z * invfy() * (y - cy()), z);
}

__device__ __forceinline__ float3 MapUpdateFunctor::unproject_world(const int &x, const int &y, const float &z) const
{
    return frame_pose(unproject(x, y, z));
}

__device__ __forceinline__ bool MapUpdateFunctor::check_vertex_visibility(const float3 &pt3d) const
{
    float3 pt = frame_pose(pt3d);
    if (pt.z < 1e-3f)
        return false;

    float2 pt2d = project(pt);

    return pt2d.x >= 0 &&
           pt2d.y >= 0 &&
           pt2d.x < width &&
           pt2d.y < height &&
           pt.z >= depth_range.x &&
           pt.z <= depth_range.y;
}

__device__ __forceinline__ bool MapUpdateFunctor::check_block_visibility(const int3 &pos) const
{
    float scale = mapState.blockWidth();
    float3 corner = pos * scale;

    if (check_vertex_visibility(corner))
        return true;

    corner.z += scale;

    if (check_vertex_visibility(corner))
        return true;

    corner.y += scale;

    if (check_vertex_visibility(corner))
        return true;

    corner.x += scale;

    if (check_vertex_visibility(corner))
        return true;

    corner.z -= scale;

    if (check_vertex_visibility(corner))
        return true;
    corner.y -= scale;

    if (check_vertex_visibility(corner))
        return true;

    corner.x -= scale;
    corner.y += scale;

    if (check_vertex_visibility(corner))
        return true;

    corner.x += scale;
    corner.y -= scale;
    corner.z += scale;

    if (check_vertex_visibility(corner))
        return true;

    return false;
}

struct Fusion
{
    __device__ __forceinline__ float2 project(const float3 &pt3d) const
    {
        return make_float2(fx * pt3d.x / pt3d.z + cx, fy * pt3d.y / pt3d.z + cy);
    }

    __device__ __forceinline__ float3 unproject(const int &x, const int &y, const float &z) const
    {
        return make_float3(z * invfx * (x - cx), z * invfy * (y - cy), z);
    }

    __device__ __forceinline__ float3 unproject_world(const int &x, const int &y, const float &z) const
    {
        return Rview * unproject(x, y, z) + tview;
    }

    __device__ __forceinline__ bool check_vertex_visibility(const float3 &pt3d) const
    {
        float3 pt = RviewInv * (pt3d - tview);
        if (pt.z < 1e-3f)
            return false;

        float2 pt2d = project(pt);

        return pt2d.x >= 0 && pt2d.y >= 0 &&
               pt2d.x < width && pt2d.y < height &&
               pt.z >= minDepth && pt.z <= maxDepth;
    }

    __device__ __forceinline__ bool check_block_visibility(const int3 &pos) const
    {
        float scale = mapState.blockWidth();
        float3 corner = pos * scale;
        if (check_vertex_visibility(corner))
            return true;
        corner.z += scale;
        if (check_vertex_visibility(corner))
            return true;
        corner.y += scale;
        if (check_vertex_visibility(corner))
            return true;
        corner.x += scale;
        if (check_vertex_visibility(corner))
            return true;
        corner.z -= scale;
        if (check_vertex_visibility(corner))
            return true;
        corner.y -= scale;
        if (check_vertex_visibility(corner))
            return true;
        corner.x -= scale;
        corner.y += scale;
        if (check_vertex_visibility(corner))
            return true;
        corner.x += scale;
        corner.y -= scale;
        corner.z += scale;
        if (check_vertex_visibility(corner))
            return true;
        return false;
    }

    __device__ __forceinline__ void create_visible_block()
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        float z = depth.ptr(y)[x];
        if (isnan(z) || z < minDepth || z > maxDepth)
            return;

        float thresh = mapState.truncateDistance() / 2;
        float z_near = min(maxDepth, z - thresh);
        float z_far = min(maxDepth, z + thresh);
        if (z_near >= z_far)
            return;

        float3 pt_near = unproject_world(x, y, z_near) * mapState.invVoxelSize();
        float3 pt_far = unproject_world(x, y, z_far) * mapState.invVoxelSize();
        float3 dir = pt_far - pt_near;

        float length = norm(dir);
        int steps = (int)ceil(2.0 * length);
        dir = dir / (float)(steps - 1);

        for (int i = 0; i < steps; ++i)
        {
            map.CreateBlock(map.posVoxelToBlock(make_int3(pt_near)));
            pt_near += dir;
        }
    }

    __device__ __forceinline__ void CheckFullVisibility()
    {
        __shared__ bool bScan;
        if (threadIdx.x == 0)
            bScan = false;
        __syncthreads();
        uint val = 0;

        int x = blockDim.x * blockIdx.x + threadIdx.x;
        if (x < mapState.maxNumHashEntries)
        {
            HashEntry &e = map.hashEntries[x];
            if (e.next != EntryAvailable)
            {
                if (check_block_visibility(e.pos))
                {
                    bScan = true;
                    val = 1;
                }
            }
        }

        __syncthreads();
        if (bScan)
        {
            int offset = ComputeOffset<1024>(val, noVisibleBlocks);
            if (offset != -1 && x < mapState.maxNumHashEntries)
            {
                map.visibleEntries[offset] = map.hashEntries[x];
            }
        }
    }

    __device__ __forceinline__ void fuseFrameWithColour()
    {

        if (blockIdx.x >= mapState.maxNumHashEntries || blockIdx.x >= *noVisibleBlocks)
            return;

        HashEntry &entry = map.visibleEntries[blockIdx.x];
        if (entry.next == EntryAvailable)
            return;
        int3 block_pos = map.posBlockToVoxel(entry.pos);
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            int3 localPos = make_int3(threadIdx.x, threadIdx.y, i);
            float3 pos = map.posVoxelToWorld(block_pos + localPos);
            pos = RviewInv * (pos - tview);
            int2 uv = make_int2(project(pos) + make_float2(0.5, 0.5));
            if (uv.x < 0 || uv.y < 0 || uv.x >= width || uv.y >= height)
                continue;

            float dist = depth.ptr(uv.y)[uv.x];
            if (isnan(dist) || dist > maxDepth || dist < minDepth)
                continue;

            float truncateDist = mapState.truncateDistance();
            float sdf = dist - pos.z;
            if (sdf >= -truncateDist)
            {
                sdf = fmin(1.0f, sdf / truncateDist);
                int w_curr = 1;
                float3 val = make_float3(rgb.ptr(uv.y)[uv.x]);
                Voxel &prev = map.voxelBlocks[entry.next + map.posLocalToIdx(localPos)];
                if (prev.weight == 0)
                    prev = Voxel(sdf, w_curr, make_uchar3(val));
                else
                {
                    val = val / 255.f;
                    float3 old = make_float3(prev.color) / 255.f;
                    float3 res = (0.2f * val + 0.8f * old) * 255.f;
                    prev.sdf = (prev.sdf * prev.weight + sdf * w_curr) / (prev.weight + w_curr);
                    prev.weight = prev.weight + w_curr;
                    prev.color = make_uchar3(res);
                }
            }
        }
    }

    __device__ __forceinline__ void defuseFrameWithColour()
    {

        if (blockIdx.x >= mapState.maxNumHashEntries || blockIdx.x >= *noVisibleBlocks)
            return;

        HashEntry &entry = map.visibleEntries[blockIdx.x];
        if (entry.next == EntryAvailable)
            return;
        int3 block_pos = map.posBlockToVoxel(entry.pos);
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            int3 localPos = make_int3(threadIdx.x, threadIdx.y, i);
            float3 pos = map.posVoxelToWorld(block_pos + localPos);
            pos = RviewInv * (pos - tview);
            int2 uv = make_int2(project(pos) + make_float2(0.5, 0.5));
            if (uv.x < 0 || uv.y < 0 || uv.x >= width || uv.y >= height)
                continue;

            float dist = depth.ptr(uv.y)[uv.x];
            if (isnan(dist) || dist > maxDepth || dist < minDepth)
                continue;

            float truncateDist = mapState.truncateDistance();
            float sdf = dist - pos.z;
            if (sdf >= -truncateDist)
            {
                sdf = fmin(1.0f, sdf / truncateDist);
                float3 val = make_float3(rgb.ptr(uv.y)[uv.x]);
                int w_curr = 1;
                Voxel &prev = map.voxelBlocks[entry.next + map.posLocalToIdx(localPos)];
                //				val = val / 255.f;
                float3 old = make_float3(prev.color);
                float3 res = (prev.weight * old - w_curr * val);
                prev.sdf = (prev.sdf * prev.weight - sdf * w_curr);
                prev.weight = prev.weight - w_curr;
                prev.color = make_uchar3(res);

                if (prev.weight <= 0)
                {
                    prev = Voxel();
                }
                else
                {
                    prev.sdf /= prev.weight;
                    prev.color = prev.color / prev.weight;
                }
            }
        }
    }

    MapStruct map;
    float invfx, invfy;
    float fx, fy, cx, cy;
    float minDepth, maxDepth;
    int width, height;
    Matrix3f Rview;
    Matrix3f RviewInv;
    float3 tview;

    uint *noVisibleBlocks;

    cv::cuda::PtrStep<float4> nmap;
    cv::cuda::PtrStep<float> depth;
    cv::cuda::PtrStep<uchar3> rgb;
};

__global__ void create_visible_blockKernel(Fusion fuse)
{
    fuse.create_visible_block();
}

__global__ void FuseColorKernal(Fusion fuse)
{
    fuse.fuseFrameWithColour();
}

__global__ void DefuseColorKernal(Fusion fuse)
{
    fuse.defuseFrameWithColour();
}

__global__ void CheckVisibleBlockKernel(Fusion fuse)
{
    fuse.CheckFullVisibility();
}