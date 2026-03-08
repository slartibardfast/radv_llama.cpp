# PHASE 9: Parallel Split Buffer Uploads

**Goal**: Upload model weights to multiple devices concurrently instead of sequentially.

## Status: PLANNED

## Problem

`ggml_backend_vk_split_buffer_set_tensor()` in `ggml-vulkan-multigpu.cpp` uploads tensor data to each device in a serial loop:

```cpp
for (int i = 0; i < extra->n_device; ++i) {
    split->buffer->iface.set_tensor(...);  // blocks per device
}
```

For a 7B model across 2 GPUs, this means ~3.5 GB uploaded to GPU 0, wait, then ~3.5 GB to GPU 1. Total wall time is sum of both transfers.

## Proposed Changes

**`ggml/src/ggml-vulkan-multigpu.cpp`** (~50 lines):

1. For **replicated tensors** (split_dim < 0):
   - Create transfer contexts for all devices upfront
   - Submit async writes to all devices
   - Wait on all fences at the end

2. For **split tensors** (split_dim 0 or 1):
   - Prepare host-side slice data for all devices
   - Submit async writes concurrently using each device's transfer queue
   - Barrier at end

3. Use `std::thread` or parallel submit pattern:
   - Each device has independent VkDevice, VkQueue, and transfer_queue
   - No cross-device synchronization needed during upload
   - Join/fence-wait after all submissions

## Design Notes

- Model loading is a one-time cost — this primarily speeds up startup
- For a 70B model on 4 GPUs: could reduce load time from ~30s to ~8s (4x parallelism, PCIe bandwidth limited)
- Each device's transfer queue is independent — no mutex contention with per-device locks (Phase 4)
- The quant interleaving logic (for row-split quantized tensors) operates on host data before upload — CPU-bound, not affected

## Expected Impact

- Model load time reduced by ~N/(N-1) for N devices (PCIe bandwidth limited per device)
- No impact on inference performance
- Most visible with large models on many GPUs

## Verification

- Model loads correctly, weights match single-GPU baseline
- Measure load time reduction vs sequential
- Verify each device receives correct tensor slices
