#include "map_struct.h"
#include "vector_math.h"
#include "safe_call.h"

MapState currentState;
bool stateInitialised = false;
__device__ MapState mapState;

__device__ __host__ int MapState::maxNumVoxels() const
{
    return maxNumVoxelBlocks * blockSize3;
}

__device__ __host__ float MapState::blockWidth() const
{
    return blockSize * voxelSize;
}

__device__ __host__ int MapState::maxNumMeshVertices() const
{
    return 3 * maxNumMeshTriangles;
}

__device__ __host__ float MapState::invVoxelSize() const
{
    return 1.0f / voxelSize;
}

__device__ __host__ int MapState::numExcessEntries() const
{
    return maxNumHashEntries - maxNumBuckets;
}

__device__ __host__ float MapState::truncateDistance() const
{
    return 8.0f * voxelSize;
}

__device__ __host__ float MapState::stepScale_raycast() const
{
    return 0.5 * truncateDistance() * invVoxelSize();
}

void updateMapState()
{
    safe_call(cudaMemcpyToSymbol(mapState, &currentState, sizeof(MapState)));

    if (!stateInitialised)
        stateInitialised = true;
}

void downloadMapState()
{
    safe_call(cudaMemcpyFromSymbol(&currentState, mapState, sizeof(MapState)));
}

__device__ HashEntry::HashEntry() : pos(make_int3(0)), next(-1), offset(-1)
{
}

__device__ HashEntry::HashEntry(int3 pos, int ptr, int offset) : pos(pos), next(ptr), offset(offset)
{
}

__device__ HashEntry::HashEntry(const HashEntry &other)
{
    pos = other.pos;
    next = other.next;
    offset = other.offset;
}

__device__ void HashEntry::release()
{
    next = -1;
}

__device__ void HashEntry::operator=(const HashEntry &other)
{
    pos = other.pos;
    next = other.next;
    offset = other.offset;
}

__device__ bool HashEntry::operator==(const int3 &pos) const
{
    return (this->pos == pos);
}

__device__ bool HashEntry::operator==(const HashEntry &other) const
{
    return other.pos == pos;
}

__device__ Voxel::Voxel()
    : sdf(std::nanf("0x7fffffff")), weight(0), color(make_uchar3(0))
{
}

__device__ Voxel::Voxel(float sdf, short weight, uchar3 rgb)
    : sdf(sdf), weight(weight), color(rgb)
{
}

__device__ void Voxel::release()
{
    sdf = std::nanf("0x7fffffff");
    weight = 0;
    color = make_uchar3(0);
}

__device__ void Voxel::getValue(float &sdf, uchar3 &color) const
{
    sdf = this->sdf;
    color = this->color;
}

__device__ void Voxel::operator=(const Voxel &other)
{
    sdf = other.sdf;
    weight = other.weight;
    color = other.color;
}

__device__ uint MapStruct::Hash(const int3 &pos)
{
    int res = ((pos.x * 73856093) ^ (pos.y * 19349669) ^ (pos.z * 83492791)) % mapState.maxNumBuckets;
    if (res < 0)
        res += mapState.maxNumBuckets;

    return res;
}

__device__ HashEntry MapStruct::CreateEntry(const int3 &pos, const int &offset)
{
    int old = atomicSub(heapCounter, 1);
    if (old >= 0)
    {
        int ptr = heapMem[old];
        if (ptr != -1)
            return HashEntry(pos, ptr * mapState.blockSize3, offset);
    }
    return HashEntry(pos, EntryAvailable, 0);
}

__device__ void MapStruct::CreateBlock(const int3 &blockPos)
{
    int bucketId = Hash(blockPos);
    int *mutex = &bucketMutex[bucketId];
    HashEntry *e = &hashEntries[bucketId];
    HashEntry *eEmpty = nullptr;
    if (e->pos == blockPos && e->next != EntryAvailable)
        return;

    if (e->next == EntryAvailable && !eEmpty)
        eEmpty = e;

    while (e->offset > 0)
    {
        bucketId = mapState.maxNumBuckets + e->offset - 1;
        e = &hashEntries[bucketId];
        if (e->pos == blockPos && e->next != EntryAvailable)
            return;

        if (e->next == EntryAvailable && !eEmpty)
            eEmpty = e;
    }

    if (eEmpty)
    {
        int old = atomicExch(mutex, EntryOccupied);
        if (old == EntryAvailable)
        {
            *eEmpty = CreateEntry(blockPos, e->offset);
            atomicExch(mutex, EntryAvailable);
        }
    }
    else
    {
        int old = atomicExch(mutex, EntryOccupied);
        if (old == EntryAvailable)
        {
            int offset = atomicAdd(entryPtr, 1);
            if (offset <= mapState.numExcessEntries())
            {
                eEmpty = &hashEntries[mapState.maxNumBuckets + offset - 1];
                *eEmpty = CreateEntry(blockPos, 0);
                e->offset = offset;
            }
            atomicExch(mutex, EntryAvailable);
        }
    }
}

__device__ bool MapStruct::FindVoxel(const float3 &pos, Voxel &vox)
{
    int3 voxel_pos = posWorldToVoxel(pos);
    return FindVoxel(voxel_pos, vox);
}

__device__ bool MapStruct::FindVoxel(const int3 &pos, Voxel &vox)
{
    HashEntry entry = FindEntry(posVoxelToBlock(pos));
    if (entry.next == EntryAvailable)
        return false;
    int idx = posVoxelToIdx(pos);
    vox = voxelBlocks[entry.next + idx];
    return true;
}

__device__ Voxel MapStruct::FindVoxel(const int3 &pos)
{
    HashEntry entry = FindEntry(posVoxelToBlock(pos));
    Voxel voxel;
    if (entry.next == EntryAvailable)
        return voxel;
    return voxelBlocks[entry.next + posVoxelToIdx(pos)];
}

__device__ Voxel MapStruct::FindVoxel(const float3 &pos)
{
    int3 p = make_int3(pos);
    HashEntry entry = FindEntry(posVoxelToBlock(p));

    Voxel voxel;
    if (entry.next == EntryAvailable)
        return voxel;

    return voxelBlocks[entry.next + posVoxelToIdx(p)];
}

__device__ Voxel MapStruct::FindVoxel(const float3 &pos, HashEntry &cache, bool &valid)
{
    int3 p = make_int3(pos);
    int3 blockPos = posVoxelToBlock(p);
    if (blockPos == cache.pos)
    {
        valid = true;
        return voxelBlocks[cache.next + posVoxelToIdx(p)];
    }

    HashEntry entry = FindEntry(blockPos);
    if (entry.next == EntryAvailable)
    {
        valid = false;
        return Voxel();
    }

    valid = true;
    cache = entry;
    return voxelBlocks[entry.next + posVoxelToIdx(p)];
}

__device__ HashEntry MapStruct::FindEntry(const float3 &pos)
{
    int3 blockIdx = posWorldToBlock(pos);

    return FindEntry(blockIdx);
}

__device__ HashEntry MapStruct::FindEntry(const int3 &blockPos)
{
    uint bucketId = Hash(blockPos);
    HashEntry *e = &hashEntries[bucketId];
    if (e->next != EntryAvailable && e->pos == blockPos)
        return *e;

    while (e->offset > 0)
    {
        bucketId = mapState.maxNumBuckets + e->offset - 1;
        e = &hashEntries[bucketId];
        if (e->pos == blockPos && e->next != EntryAvailable)
            return *e;
    }
    return HashEntry(blockPos, EntryAvailable, 0);
}

__device__ int3 MapStruct::posWorldToVoxel(float3 pos) const
{
    float3 p = pos / mapState.voxelSize;
    return make_int3(p);
}

__device__ float3 MapStruct::posWorldToVoxelFloat(float3 pos) const
{
    return pos / mapState.voxelSize;
}

__device__ float3 MapStruct::posVoxelToWorld(int3 pos) const
{
    return pos * mapState.voxelSize;
}

__device__ int3 MapStruct::posVoxelToBlock(const int3 &pos) const
{
    int3 voxel = pos;

    if (voxel.x < 0)
        voxel.x -= mapState.blockSize - 1;
    if (voxel.y < 0)
        voxel.y -= mapState.blockSize - 1;
    if (voxel.z < 0)
        voxel.z -= mapState.blockSize - 1;

    return voxel / mapState.blockSize;
}

__device__ int3 MapStruct::posBlockToVoxel(const int3 &pos) const
{
    return pos * mapState.blockSize;
}

__device__ int3 MapStruct::posVoxelToLocal(const int3 &pos) const
{
    int3 local = pos % mapState.blockSize;

    if (local.x < 0)
        local.x += mapState.blockSize;
    if (local.y < 0)
        local.y += mapState.blockSize;
    if (local.z < 0)
        local.z += mapState.blockSize;

    return local;
}

__device__ int MapStruct::posLocalToIdx(const int3 &pos) const
{
    return pos.z * mapState.blockSize * mapState.blockSize + pos.y * mapState.blockSize + pos.x;
}

__device__ int3 MapStruct::posIdxToLocal(const int &idx) const
{
    uint x = idx % mapState.blockSize;
    uint y = idx % (mapState.blockSize * mapState.blockSize) / mapState.blockSize;
    uint z = idx / (mapState.blockSize * mapState.blockSize);
    return make_int3(x, y, z);
}

__device__ int3 MapStruct::posWorldToBlock(const float3 &pos) const
{
    return posVoxelToBlock(posWorldToVoxel(pos));
}

__device__ float3 MapStruct::posBlockToWorld(const int3 &pos) const
{
    return posVoxelToWorld(posBlockToVoxel(pos));
}

__device__ int MapStruct::posVoxelToIdx(const int3 &pos) const
{
    return posLocalToIdx(posVoxelToLocal(pos));
}