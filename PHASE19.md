# Phase 19: Graph-Split Correctness

## Problem

Multi-GPU graph-split mode (`-sm graph`) produces incorrect output with greedy sampling (temp=0). Single-GPU and layer-split modes produce correct output. The issue is pre-existing — both CPU-mediated and dmabuf REDUCE paths produce identical (wrong) output, confirming the bug is upstream of REDUCE.

## Symptoms

```
Single GPU:   "The capital of France isyi..."  ← matches layer-split
Layer split:  "The capital of France isyi..."  ← correct
Graph split:  "The capital of France is Hemddddddddddddddddddd"  ← wrong
```

## Suspected Areas

1. **Weight partitioning**: The scheduler splits weight matrices across devices (~25%/75% by VRAM). If the split points are wrong or the partial sums don't align, REDUCE accumulates incorrect values.

2. **Residual accumulation**: In graph-split, device 1 does `ADD(residual)` before REDUCE. If the residual is applied to the wrong partial sum, the layer output is corrupted.

3. **F16 cast alignment**: The `CPY` op casts partial sums from F32 → F16 before REDUCE. If the cast truncates differently across devices, accumulated error could diverge.

4. **Redundant norm computation**: Both devices compute `FUSED_RMS_NORM` on the same input independently. If the REDUCE result isn't properly broadcast to both devices before the next layer's norm, the inputs diverge.

## Approach

1. Dump intermediate tensors at each REDUCE point for single-GPU vs graph-split
2. Find the first layer where outputs diverge
3. Trace back to the source of divergence (weight split, residual, norm)

## Verify by

Graph-split and single-GPU produce identical output with `--temp 0`.
