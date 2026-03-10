# Phase 20: Vega Rapid Packed Math Optimization

## Goal

Exploit Vega 10's Rapid Packed Math (RPM) — packed FP16 via `f16vec2` — in Vulkan MUL_MAT fallback shaders. Vega introduced RPM but llama.cpp's Vulkan shaders don't use it in the non-DP4A fallback path.

## Background

Vega 10 (GCN5) has no native DP4A integer dot product. The current fallback dequantizes to float and uses FMA, costing 4-8 cycles per element vs 1-2 cycles for DP4A hardware. The real performance gap is 2-4×, not the naive 8× estimate.

The most promising unexploited optimization: Vega's RPM processes two FP16 values per instruction. Combined with:

- **f16vec2 packed arithmetic**: 2× throughput for dequant + accumulate
- **VGPR budgeting ≤32 registers**: Maximizes occupancy on GCN5 (max wavefronts per SIMD)
- **64-lane wavefront reductions**: Full utilization of Vega's native wave64

## Scope

### Primary target: `mul_mat_vec` (token generation bottleneck)

The mat-vec kernel dominates token generation. For quantized weights (Q4_K, Q6_K):

1. Dequantize weight block to FP16 (currently F32)
2. Multiply by activation (currently F32)
3. Accumulate via subgroup reduction (currently F32)

With RPM: steps 1-2 operate on `f16vec2` pairs, halving instruction count. Step 3 uses `subgroupAdd` on the full 64-lane wavefront.

### Secondary target: `flash_attn` scalar path

Flash attention on Vega uses the scalar (non-coopmat) path. FP16 packed K×Q dot products could improve attention throughput.

## Key Constraints

- Vega subgroup size is 64 (wave64). Shaders must not assume wave32.
- RADV is the only Vulkan driver for GCN (AMDVLK is extinct). RADV continues to improve.
- Must not regress RDNA2 performance — RPM paths should be guarded by device capability checks or specialization constants.
- `VK_KHR_shader_float16_int8` must be enabled for `float16_t` / `f16vec2` in shaders.

## Approach

1. Profile MUL_MAT_VEC on Vega to establish baseline (cycles per element, occupancy)
2. Write RPM-optimized dequant kernels for Q4_K and Q6_K using `f16vec2`
3. Add specialization constant or device check to select RPM path on GCN5
4. Benchmark against baseline on Vega, verify no regression on 6800 XT

## Verify by

MUL_MAT_VEC throughput improvement on Vega for Q4_K/Q6_K weights, measured via `test-backend-ops perf`.

## References

- AMD GCN5 ISA: Rapid Packed Math operates on `v_pk_*` instructions
- RADV driver: full RPM support through `VK_KHR_shader_float16_int8`
- llama.cpp Vulkan shaders: `vulkan-shaders/mul_mat_vec*.comp`
