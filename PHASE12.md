# PHASE 12: dmabuf Zero-Copy Cross-Device Transfer (RADV)

**Goal**: Eliminate host staging for cross-device copies on Mesa RADV by sharing GPU buffers via Linux dmabuf file descriptors.

## Status: PLANNED (optional, driver-specific)

## Problem

All cross-device copies currently route through host memory:

```
GPU A device-local → host staging → GPU B device-local
```

This doubles PCIe traffic. On PCIe 4.0 x16, effective cross-device bandwidth is ~6 GB/s (half of the ~12 GB/s unidirectional bandwidth).

## Proposed Changes

**`ggml/src/ggml-vulkan.cpp`** (~120 lines):

1. Probe for `VK_KHR_external_memory_fd` support at device init:
   - Check both source and destination devices support the extension
   - Query `VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT` compatibility
   - Cache result in `vk_device_struct`

2. When both devices support dmabuf, create exportable staging buffers:
   - Allocate with `VkExportMemoryAllocateInfo` + `VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT`
   - Export fd via `vkGetMemoryFdKHR`
   - Import on peer device via `vkAllocateMemory` + `VkImportMemoryFdInfoKHR`

3. Cross-device copy becomes:
   ```
   GPU A: vkCmdCopyBuffer(src → shared_buffer)  // device-local to exportable
   Host:  (no copy needed — same physical pages)
   GPU B: vkCmdCopyBuffer(shared_buffer → dst)  // importable to device-local
   ```

4. Fallback to host staging when dmabuf is not available (existing Phase 3/7 path)

## Design Notes

### Driver support
- **RADV (Mesa AMD)**: Full dmabuf export/import support. This is the primary target.
- **ANV (Mesa Intel)**: dmabuf supported for integrated GPUs, less useful (shared memory anyway)
- **NVIDIA proprietary**: Does NOT support `VK_KHR_external_memory_fd` for cross-device. Uses `VK_KHR_external_memory_win32` on Windows or proprietary CUDA P2P.
- **lavapipe**: No dmabuf support (unnecessary — already host memory)

### PCIe topology matters
- dmabuf doesn't enable true GPU-to-GPU DMA over PCIe by itself
- The kernel's DMA-BUF subsystem may still route through IOMMU or host memory depending on PCIe topology
- On AMD platforms with SAM (Smart Access Memory) / resizable BAR enabled, direct peer access may be possible
- Without SAM/rBAR, dmabuf still avoids one copy (GPU → host → GPU becomes GPU → shared → GPU, but shared may be host-mapped)

### Why this is optional
- Host staging (Phases 3/7/8) works everywhere and is well-tested
- dmabuf adds driver-specific code paths that are harder to test and maintain
- The performance improvement depends on PCIe topology which varies per machine
- Best saved for when actual multi-discrete-GPU RADV testing is available

## Expected Impact

- Best case (SAM/rBAR enabled, direct P2P): ~12 GB/s cross-device bandwidth (2x improvement)
- Typical case (IOMMU routing): ~8-10 GB/s (30-60% improvement)
- Worst case (falls back to host staging): no change

## Verification

- Probe result logged at startup: "dmabuf P2P: supported/unsupported" per device pair
- Cross-device inference matches host-staging baseline
- Measure actual bandwidth with large tensor transfers
- Test fallback path on non-RADV drivers
