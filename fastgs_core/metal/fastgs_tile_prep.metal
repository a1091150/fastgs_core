#include <metal_stdlib>

using namespace metal;

struct TilePrepKernelParams {
  uint num_rendered;
  uint num_tiles;
};

kernel void fastgs_identify_tile_ranges_kernel(
    constant TilePrepKernelParams& params [[buffer(0)]],
    device ulong* point_list_keys [[buffer(1)]],
    device uint* ranges [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.num_rendered) {
    return;
  }

  ulong key = point_list_keys[gid];
  uint currtile = uint(key >> 32);
  bool valid_tile = currtile != uint(-1);

  if (gid == 0) {
    ranges[2 * currtile] = 0u;
  } else {
    uint prevtile = uint(point_list_keys[gid - 1] >> 32);
    if (currtile != prevtile) {
      ranges[2 * prevtile + 1] = gid;
      if (valid_tile) {
        ranges[2 * currtile] = gid;
      }
    }
  }
  if (gid == params.num_rendered - 1 && valid_tile) {
    ranges[2 * currtile + 1] = params.num_rendered;
  }
}

kernel void fastgs_per_tile_bucket_count_kernel(
    constant TilePrepKernelParams& params [[buffer(0)]],
    device uint* ranges [[buffer(1)]],
    device uint* bucket_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.num_tiles) {
    return;
  }

  uint start = ranges[2 * gid];
  uint end = ranges[2 * gid + 1];
  int num_splats = int(end - start);
  int num_buckets = (num_splats + 31) / 32;
  bucket_count[gid] = uint(num_buckets);
}
