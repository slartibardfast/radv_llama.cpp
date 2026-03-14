# Benchmarks

## Test Hardware

### System A: Retro Computing Baseline

| Component | Specification |
|-----------|--------------|
| CPU | Intel Xeon X5650 (Westmere, 2010) — 6C/12T, no AVX/AVX2, DDR3-1333 triple-channel |
| GPU 0 | AMD Radeon Pro WX 2100 (Polaris 12, 2017) — 512 SPs, 2 GB GDDR5, PCIe 3.0 x16 |
| GPU 1 | lavapipe (Mesa llvmpipe) — CPU-based Vulkan software rasterizer |
| Driver | Mesa RADV (Polaris12), Mesa llvmpipe |
| OS | Arch Linux, kernel 6.18.16-2-lts |

Deliberately terrible hardware to stress-test the implementation at the low end. The Polaris 12 is a passively-cooled workstation card with roughly the compute power of a 2013 midrange GPU. lavapipe has no actual GPU hardware — every Vulkan operation runs on CPU.

### System B: Real Multi-GPU

| Component | Specification |
|-----------|--------------|
| CPU | AMD Ryzen 9 3950X (Zen 2, 2019) — 16C/32T, AVX2/FMA, DDR4 |
| GPU 0 | AMD Radeon RX 6800 XT (RDNA 2, Navi 21) — 16 GB GDDR6, PCIe 4.0 x16 |
| GPU 1 | AMD Radeon RX Vega 56/64 (GCN 5, Vega 10) — 8 GB HBM2, PCIe 3.0 x16 |
| Driver | Mesa RADV (Navi21 + Vega10) |
| OS | Arch Linux, kernel 6.12.74-1-lts |

Mixed-generation discrete AMD GPUs over PCIe. The 6800 XT (RDNA 2) is roughly 2x the compute throughput of the Vega (GCN 5). Different warp sizes (32 vs 64) and different PCIe generations test asymmetric device handling.

## Models

| Model | Quantization | Size | Parameters | Vulkan Graph Splits |
|-------|-------------|------|------------|-------------------|
| TinyLlama 1.1B Chat v1.0 | Q2_K | 459 MiB | 1.1 B | 2 (optimal) |
| Llama-2-7B | Q8_0 | 6.7 GiB | 6.7 B | 2 (optimal) |
| Llama-2-13B | Q8_0 | 13 GiB | 13 B | 2 (optimal) |

Sources: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`, `TheBloke/Llama-2-7B-GGUF`, `TheBloke/Llama-2-13B-GGUF`

### Model Compatibility Notes

Not all models work efficiently on the ik_llama.cpp Vulkan backend. Models with unsupported ops create excessive graph splits, causing constant GPU↔CPU bouncing:

| Model | Arch | Graph Splits | Status |
|-------|------|-------------|--------|
| TinyLlama 1.1B | llama (v1) | 2 | Fast |
| Llama-2-7B | llama (v2) | 2 | Fast |
| Llama-2-13B | llama (v2) | 2 | Fast |
| Llama-3.2-3B | llama | 58 | Very slow (unsupported ops per layer) |
| Llama-3.1-8B | llama | 66 | Very slow (same issue as 3.2) |
| Qwen3.5-4B | qwen35 (SSM+attn) | 306 | Very slow |
| GLM-4.7-Flash | deepseek2 (MoE) | 186 | Very slow |

Graph splits > ~5 on single GPU indicates ops falling back to CPU — a performance disaster regardless of GPU speed.

## Results

### System A: Retro Baseline (Polaris 12 + lavapipe)

#### Phase Evolution (multi-GPU, 2 token prompt, 4 eval tokens)

| Phase | Prompt eval | Token gen | Total | Key change |
|-------|------------|-----------|-------|------------|
| Phase 6 (sync copies) | 15378 ms | 685 ms/tok | 18127 ms | Baseline |
| Phase 7 (async copies) | 862 ms | 560 ms/tok | 3110 ms | Deferred fence wait |
| Phase 8 (staging pool) | 893 ms | 537 ms/tok | 3051 ms | Per-copy staging |
| Phase 9 (parallel upload) | ~1112 ms load | 537 ms/tok | ~3035 ms | Threaded uploads |

Key finding: Phase 7 delivered a **17.8x prompt eval speedup** by eliminating synchronous fence waits during cross-device copies.

#### Load Time: Serial vs Parallel Upload (Phase 9)

| Mode | Average (3 trials) |
|------|-------------------|
| Serial upload | 1117 ms |
| Parallel upload | 1112 ms |

No improvement because lavapipe's "uploads" are instant memcpy. Benefit requires two discrete GPUs.

### System B: Real Multi-GPU (RX 6800 XT + Vega)

> Results pending re-measurement after Phase 0 bug fixes.
> Pre-Phase 0 data available in git history at commit `c97afb6`.

## Runtime Flags

| Flag | Purpose |
|------|---------|
| `-smgs` | Enable split mode graph scheduling |
| `-ts 1,1` | Tensor split ratio (equal across 2 devices) |
| `-ngl 99` | Offload all layers to GPU |
| `-no-fa` | Disable flash attention (required on non-AVX2) |
| `-no-fug` | Disable fused up-gate (required on non-AVX2) |
| `-c 2048` | Context size (reduce for VRAM-limited configs) |
| `GGML_VK_VISIBLE_DEVICES=0,1` | Include non-discrete devices like lavapipe |
