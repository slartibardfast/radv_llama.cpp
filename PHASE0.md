# Phase 0: Backend-Ops Test Failure Fixes

## Status: IN PROGRESS (3 failures remaining)

## Overview

Fix all backend-ops test failures on RADV Vulkan (Vega 56 / RX 6800 XT). Originally 10 failures; 7 now fixed, 3 remain.

## Round 1: K-Quant Dequant Bounds Bug (2026-03-11)

**Root cause**: 5 K-quant dequant shaders (`dequant_q2_k.comp` through `dequant_q6_k.comp`) used `p.M * p.K / QUANT_K` for bounds checking instead of `p.nel / QUANT_K`. For multi-batch tensors, only the first batch was dequantized; remaining batches were uninitialized garbage in the prealloc buffer.

**Fixed**: 4 direct failures (q4_K x f16 batched MUL_MAT) + some indirect contamination failures.

Files changed: `dequant_q2_k.comp` through `dequant_q6_k.comp`

## Round 2: Push Constant & get_offsets Alignment (2026-03-12)

**Changes applied**:
1. **Buffer sizing**: `x_ne = ggml_nelements(src0)` instead of `ne01*ne00` (covers all batches/experts). Same for y_ne, d_sz.
2. **Push constant struct alignment**: Added `fusion_flags` (always 0) and `base_work_group_y` / `expert_i1` / `nbi1` fields to match upstream layout.
3. **get_offsets alignment**: Now does `batch_stride_a / QUANT_K` inside `get_offsets()`, matching upstream. Removed external `a_offset /= QUANT_K` from `mul_mat_vec.comp` and `a_offset / QUANT_K` from 12 specialized shaders.
4. **MUL_MAT_ID struct**: Added `fusion_flags`, `expert_i1`, `nbi1` fields to match upstream dispatch.

**Fixed**: iq3_xxs MUL_MAT and MUL_MAT_ID (marginal NMSE now consistently passes).

Files changed:
- `ggml/src/ggml-vulkan.cpp` — C++ push constant structs and dispatch code
- `ggml/src/vulkan-shaders/mul_mat_vec_base.comp` — push constant layout, get_offsets
- `ggml/src/vulkan-shaders/mul_mat_vec.comp` — removed a_offset /= QUANT_K
- 12 specialized `mul_mat_vec_*.comp` shaders — removed a_offset / QUANT_K

## Remaining Failures (3)

| # | Test | NMSE | Threshold | Notes |
|---|------|------|-----------|-------|
| 1 | MUL_MAT(iq4_xs x f32, m=16,n=1,k=256) | 0.040 | 0.0005 | 80x over threshold |
| 2 | MUL_MAT(bf16 x f32, m=16,n=1,k=1) | 4.2 | 0.0005 | 8400x over, huge error |
| 3 | MUL_MAT_ID(iq4_xs x f32, n_mats=4,n_used=2,m=512,n=1,k=256) | 0.032 | 0.0005 | 64x over threshold |

All 3 reproduce in isolation (`-o MUL_MAT` / `-o MUL_MAT_ID`) — NOT contamination.

Upstream llama.cpp passes all these tests on the same Vega hardware.

## What's Been Ruled Out

- **Contamination**: Failures persist when running each op type in isolation
- **Shader logic bugs**: Line-by-line GLSL comparison shows functionally identical code between fork and upstream for all mul_mat_vec paths
- **Push constant value errors**: C++ struct and GLSL struct are self-consistent in both codebases; values at each offset are correct
- **Pipeline creation differences**: wg_denoms, specialization constants, align, disable_robustness all match upstream
- **Split-mode regression**: These dispatch functions (`ggml_vk_mul_mat_vec_q_f16`, `ggml_vk_mul_mat_vec_id_q_f16`) were NOT modified by split-mode commits; they are pre-existing ik_llama.cpp divergence from upstream

## Remaining Differences Between Fork and Upstream

1. **Descriptor binding count**: Fork uses 3 bindings (A, B, D). Upstream uses 5 (A, B, D, Fuse0, Fuse1). Extra bindings present in upstream SPIR-V even when unused (fusion_flags=0).

2. **iq4_xs dequant implementation**: Fork uses 4 individual byte reads; upstream uses 1 packed 32-bit read + `unpack8`. Functionally identical but different SPIR-V.

3. **bf16 puzzle**: For bf16, the GLSL source is truly identical (same dequant, same shader logic, same push constants). The ONLY structural difference is the missing Fuse0/Fuse1 bindings. Yet bf16 k=1 produces NMSE=4.2.

4. **Subgroup reduction**: Upstream compiles subgroup reduction variants; fork always uses shmem reduction. Both paths are mathematically equivalent.

## CPY f32->iq4_nl Status

Pending investigation. These failures may share a root cause with iq4_xs MUL_MAT failures.
