# PHASE 10: True Async Graph Compute

**Goal**: Allow the scheduler to overlap graph execution across multiple Vulkan backends.

## Status: PLANNED

## Problem

`ggml_backend_vk_graph_compute()` blocks until the compute fence signals:

1. Submits all compute work via `vkQueueSubmit`
2. Spin-waits on fence with `YIELD()` pauses (lines 1164-1193)
3. Only returns after all GPU work is done

This means the scheduler cannot submit work to GPU B while GPU A is still computing. The scheduler's parallel execution path (`ggml-backend.cpp:2160-2380`) requires async `graph_compute` + explicit `synchronize()` to overlap backends.

## Proposed Changes

**`ggml/src/ggml-vulkan.cpp`** (~30 lines):

1. Split `graph_compute` into submit-only mode:
   - After `vkQueueSubmit`, record the fence but do NOT wait
   - Store pending fence in `ggml_backend_vk_context`
   - Return immediately

2. Implement `ggml_backend_vk_synchronize()`:
   - Wait on the pending compute fence
   - Execute deferred memcpys (`out_memcpys`)
   - Run graph cleanup (already exists as `ggml_vk_graph_cleanup`)
   - Clear pending fence

3. Wire `synchronize` into the backend vtable (currently NULL at line 10708)

4. Keep the "almost_ready" fence optimization — it still helps by allowing CPU prep work to overlap with the final 20% of GPU compute

## Design Constraints

- The scheduler checks `backend_supports_async(backend)` before using the parallel path. This requires non-NULL `synchronize` in the vtable.
- Current behavior (blocking graph_compute) is the fallback when `synchronize` is NULL — no regression for single-GPU.
- Must ensure all `out_memcpys` are executed in `synchronize`, not in `graph_compute`. Currently they run after fence wait in `graph_cleanup`.
- Thread safety: each `ggml_backend_vk_context` is per-backend, so no cross-context issues.

## Risk

- The scheduler's parallel execution path uses barriers (`std::barrier` or OpenMP) between backends. If `synchronize` doesn't properly wait for ALL pending work, results will be incorrect.
- Need to verify that graph_cleanup can be safely deferred to `synchronize`.

## Expected Impact

- Enables true overlap: GPU A computes split N while GPU B computes split N-1
- For 2 GPUs with balanced splits: up to ~1.5-1.8x throughput improvement on prompt processing
- For generation (single token, sequential): minimal improvement (each token depends on previous)
- The "almost_ready" fence already gives ~10% overlap benefit — this extends it to full overlap

## Verification

- Output matches single-GPU baseline token-for-token
- Scheduler log confirms async execution path taken
- Measure tokens/sec improvement for prompt processing (batch)
- Measure tokens/sec for generation (expect minimal change)
- Run with `VK_LAYER_KHRONOS_validation` to catch synchronization errors
