#ifndef __MAP_STRUCT__
#define __MAP_STRUCT__

#include <cuda.h>
#include <cuda_runtime.h>

enum
{
  EntryAvailable = -1,
  EntryOccupied = -2
};

struct MapState
{
  // constants shouldn't be changed at all
  // these are down to the basic design of the system
  // changes made to these will render system unstable
  // if not unusable at all.
  int blockSize;
  int blockSize3;
  int renderingBlockSize;
  int minMaxSubSample;

  // parameters that control the size of the
  // device memory needed in the allocation stage.
  // Note that numBuckets should always be bigger
  // than numSdfBlock as that's the requirement
  // of hash table;
  int maxNumBuckets;
  int maxNumVoxelBlocks;
  int maxNumMeshTriangles;
  int maxNumHashEntries;
  int maxNumRenderingBlocks;

  // parameters control how far the camera sees
  // should keep them in minimum as long as they
  // satisfy your needs. Larger viewing frusta
  // will significantly slow down the system.
  // as more sdf blocks will be allocated.
  float depthMin_raycast;
  float depthMax_raycast;
  float depthMin_preprocess;
  float depthMax_preprocess;

  // parameters that won't affect system performance
  // too much, generally just affect the appearance
  // of the map and are free to be modified.
  // Note that due to imperfections in the function
  // PARRALLEL SCAN, too large voxelSize will not work.
  float voxelSize;

  __device__ __host__ int maxNumVoxels() const;
  __device__ __host__ float blockWidth() const;
  __device__ __host__ int maxNumMeshVertices() const;
  __device__ __host__ float invVoxelSize() const;
  __device__ __host__ int numExcessEntries() const;
  __device__ __host__ float truncateDistance() const;
  __device__ __host__ float stepScale_raycast() const;
};

extern bool stateInitialised;
extern MapState currentState;
__device__ extern MapState mapState;

void updateMapState();
void downloadMapState();

struct RenderingBlock
{
  short2 upperLeft;
  short2 lowerRight;
  float2 zRange;
};

struct Voxel
{
  __device__ Voxel();
  __device__ Voxel(float sdf, short weight, uchar3 rgb);
  __device__ void release();
  __device__ void getValue(float &sdf, uchar3 &rgb) const;
  __device__ void operator=(const Voxel &other);

  float sdf;
  short weight;
  uchar3 color;
};

struct HashEntry
{
  __device__ HashEntry();
  __device__ HashEntry(int3 pos, int next, int offset);
  __device__ HashEntry(const HashEntry &other);
  __device__ void release();
  __device__ void operator=(const HashEntry &other);
  __device__ bool operator==(const int3 &pos) const;
  __device__ bool operator==(const HashEntry &other) const;

  int next;
  int offset;
  int3 pos;
};

struct MapStruct
{
  __device__ uint Hash(const int3 &pos);
  __device__ Voxel FindVoxel(const int3 &pos);
  __device__ Voxel FindVoxel(const float3 &pos);
  __device__ Voxel FindVoxel(const float3 &pos, HashEntry &cache, bool &valid);
  __device__ HashEntry FindEntry(const int3 &pos);
  __device__ HashEntry FindEntry(const float3 &pos);
  __device__ void CreateBlock(const int3 &blockPos);
  __device__ bool FindVoxel(const int3 &pos, Voxel &vox);
  __device__ bool FindVoxel(const float3 &pos, Voxel &vox);
  __device__ HashEntry CreateEntry(const int3 &pos, const int &offset);

  __device__ int3 posWorldToVoxel(float3 pos) const;
  __device__ int3 posVoxelToBlock(const int3 &pos) const;
  __device__ int3 posBlockToVoxel(const int3 &pos) const;
  __device__ int3 posVoxelToLocal(const int3 &pos) const;
  __device__ int3 posIdxToLocal(const int &idx) const;
  __device__ int3 posWorldToBlock(const float3 &pos) const;
  __device__ int posLocalToIdx(const int3 &pos) const;
  __device__ int posVoxelToIdx(const int3 &pos) const;
  __device__ float3 posWorldToVoxelFloat(float3 pos) const;
  __device__ float3 posVoxelToWorld(int3 pos) const;
  __device__ float3 posBlockToWorld(const int3 &pos) const;

  int *heapMem;
  int *entryPtr;
  int *heapCounter;
  int *bucketMutex;
  Voxel *voxelBlocks;
  uint *noVisibleBlocks;
  HashEntry *hashEntries;
  HashEntry *visibleEntries;
  bool device_;
};

#endif