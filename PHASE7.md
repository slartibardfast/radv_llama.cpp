# PHASE 7: Async Cross-Device Copy Pipeline

**Goal**: Eliminate the synchronous host-side block during cross-device tensor copies.

## Status: PLANNED

## Problem

`ggml_backend_vk_cpy_tensor_async()` (ggml-vulkan.cpp:10135) performs cross-device copies as:

1. `ggml_vk_buffer_read()` — **synchronous**: submits GPU A → host transfer, calls `waitForFences()`, blocks until complete
2. `ggml_vk_buffer_write_async()` — async: batches host → GPU B transfer into transfer context

Step 1 stalls the calling thread while GPU A's DMA engine drains to host memory. No other work can proceed on either device during this window.

## Proposed Changes

**`ggml/src/ggml-vulkan.cpp`** (~40 lines):

1. Replace `ggml_vk_buffer_read()` with an async read using the source device's transfer queue:
   - Create a temporary transfer context on the source device
   - Submit `vkCmdCopyBuffer` (device-local → sync_staging) without fence wait
   - Use timeline semaphore signal on completion

2. Host memcpy from source staging → destination staging (or direct if pinned)

3. Submit async write to destination device (existing path)

4. Record timeline semaphore on destination device's transfer context

The key insight: the timeline semaphore infrastructure from Phase 1 is already wired and functional. The `event_record` / `event_wait` functions use `VK_SEMAPHORE_TYPE_TIMELINE` — we just need to use them in the copy path instead of blocking fences.

## Design Constraints

- Host memcpy between staging buffers is unavoidable (Vulkan has no cross-VkDevice buffer copies)
- The memcpy itself is fast (~12 GB/s on DDR4) — the bottleneck is the fence wait before it
- lavapipe path (host-visible buffers) already bypasses this via direct memcpy — no change needed

## Expected Impact

- Removes one full GPU→host fence wait per cross-device copy
- On PCIe 4.0 x16: saves ~1-2ms per transfer (fence overhead + synchronization latency)
- Enables pipelining: GPU A can start next compute while its DMA engine transfers to host

## Verification

- Cross-device inference produces identical output to Phase 6 baseline
- Measure per-token latency reduction with 2-GPU split
- No validation layer errors (`VK_LAYER_KHRONOS_validation`)
