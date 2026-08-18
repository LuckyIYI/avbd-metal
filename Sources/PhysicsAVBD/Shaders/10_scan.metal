#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Generic exclusive scan over uint, used for grid cell offsets and CSR
// adjacency offsets. Three-kernel structure:
//   scan_blocks -> scan_block_sums (single TG) -> scan_add_offsets
// Block = 256 threads x 4 grains = 1024 elements.
// ============================================================================

#define SCAN_TG 256
#define SCAN_GRAIN 4
#define SCAN_BLOCK (SCAN_TG * SCAN_GRAIN)

kernel void scan_blocks(
    device const uint* input        [[buffer(0)]],
    device uint* output             [[buffer(1)]],
    device uint* blockSums          [[buffer(2)]],
    constant uint& count            [[buffer(3)]],
    uint tid                        [[thread_position_in_threadgroup]],
    uint gid                        [[threadgroup_position_in_grid]])
{
    threadgroup uint shared[SCAN_TG];

    uint base = gid * SCAN_BLOCK;
    // Each thread loads 4 consecutive, computes local inclusive sums
    uint v[SCAN_GRAIN];
    uint sum = 0;
    for (uint i = 0; i < SCAN_GRAIN; i++) {
        uint idx = base + tid * SCAN_GRAIN + i;
        v[i] = idx < count ? input[idx] : 0;
        sum += v[i];
    }
    shared[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Blelloch-style scan over thread sums (simple Hillis-Steele, 256 wide)
    uint val = sum;
    for (uint offset = 1; offset < SCAN_TG; offset <<= 1) {
        uint other = tid >= offset ? shared[tid - offset] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        val += other;
        shared[tid] = val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint exclusive = val - sum; // exclusive prefix for this thread

    if (tid == SCAN_TG - 1) {
        blockSums[gid] = val;   // total of block
    }

    // Write exclusive scan of the 4 grains
    uint running = exclusive;
    for (uint i = 0; i < SCAN_GRAIN; i++) {
        uint idx = base + tid * SCAN_GRAIN + i;
        if (idx < count) output[idx] = running;
        running += v[i];
    }
}

// Single-threadgroup exclusive scan over block sums (loops over chunks).
kernel void scan_block_sums(
    device uint* blockSums          [[buffer(0)]],
    constant uint& numBlocks        [[buffer(1)]],
    device uint* totalOut           [[buffer(2)]],  // optional total (1 uint)
    uint tid                        [[thread_position_in_threadgroup]])
{
    threadgroup uint shared[SCAN_TG];
    threadgroup uint carry;
    if (tid == 0) carry = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint base = 0; base < numBlocks; base += SCAN_TG) {
        uint idx = base + tid;
        uint v = idx < numBlocks ? blockSums[idx] : 0;
        shared[tid] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint val = v;
        for (uint offset = 1; offset < SCAN_TG; offset <<= 1) {
            uint other = tid >= offset ? shared[tid - offset] : 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            val += other;
            shared[tid] = val;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint exclusive = val - v + carry;
        if (idx < numBlocks) blockSums[idx] = exclusive;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == SCAN_TG - 1) carry = exclusive + v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0 && totalOut) totalOut[0] = carry;
}

kernel void scan_add_offsets(
    device uint* output             [[buffer(0)]],
    device const uint* blockSums    [[buffer(1)]],
    constant uint& count            [[buffer(2)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid < count) output[gid] += blockSums[gid / SCAN_BLOCK];
}
