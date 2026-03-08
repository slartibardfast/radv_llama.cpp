# PHASE 8: Double-Buffered Staging

**Goal**: Overlap transfer and compute by rotating between two staging buffers per device.

## Status: PLANNED

## Problem

Each device has a single `sync_staging` buffer (`vk_device_struct::sync_staging`). When a transfer uses it, no other transfer can proceed on that device until the first completes. This serializes all staging operations.

Additionally, the thread-local `std::vector<char>` staging in `cpy_tensor_async` is host memory that requires an extra copy — it doesn't participate in Vulkan's command buffer pipeline.

## Proposed Changes

**`ggml/src/ggml-vulkan.cpp`** (~60 lines):

1. Replace single `sync_staging` with a pair:
   ```cpp
   struct vk_device_struct {
       vk_buffer staging[2];
       int staging_idx = 0;  // alternates 0/1
   };
   ```

2. On each staging buffer acquisition, rotate:
   - Before using `staging[staging_idx]`, ensure previous use on that buffer has completed (fence/semaphore check)
   - If still in-flight, use the other buffer
   - If both in-flight, wait on the oldest (degrades gracefully to current behavior)

3. Update `ggml_vk_ensure_sync_staging_buffer()` to manage the pair:
   - Allocate both buffers to the same size
   - Track in-flight status per buffer

4. Update all staging buffer consumers:
   - `ggml_vk_buffer_write_2d_async` (non-pinned path)
   - `ggml_vk_buffer_write_nc_async`
   - `ggml_vk_buffer_read_2d_async` (non-pinned path)
   - Cross-device copy path (Phase 7)

## Design Notes

- Two buffers is sufficient — triple-buffering adds complexity for marginal gain
- Buffer sizes match: both allocated to `max(needed_size)` to avoid per-transfer reallocation
- lavapipe (host-visible) bypasses staging entirely — unaffected
- The `deferred_memcpy` mechanism (`in_memcpys`/`out_memcpys` vectors) naturally batches host-side copies and works with either buffer

## Dependency

- Benefits greatly from Phase 7 (async copies) — together they enable true pipelining
- Without Phase 7, double-buffering still helps when multiple transfers happen in sequence (e.g., multi-tensor uploads)

## Expected Impact

- Hides staging buffer latency behind compute work
- Most visible during prompt processing (many tensor transfers per batch)
- For generation (single token), improvement is marginal — transfers are small

## Verification

- Identical output to Phase 7 baseline
- Measure prompt processing throughput improvement with 2-GPU split
- Verify both buffers are used (debug log buffer index per transfer)
- No validation layer errors
