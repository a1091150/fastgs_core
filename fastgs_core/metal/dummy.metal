#include <metal_stdlib>
using namespace metal;

kernel void dummy_copy(
    const device float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  out[gid] = in[gid];
}
