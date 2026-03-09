# Phase 17: Vulkan Op Trace for Nemotron (LLM_ARCH_DECI)

## Model Architecture

Nemotron models (Nemotron-3-Nano-4B, Llama-3.1-Nemotron-51B, Llama-3.1-Nemotron-Ultra-253B) use the `LLM_ARCH_DECI` architecture. The graph is built by `build_deci()` at `llama-build-context.cpp:2319`.

Key architectural features:
- **Variable per-layer structure**: each layer can have different `n_head`, `n_head_kv`, and `n_ff`
- **Attention-free layers** (`n_head == 0`): skip attention entirely (Nemotron-51B)
- **Linear attention layers** (`n_head > 0, n_head_kv == 0`): single `wo` projection, no KV cache (Nemotron-51B)
- **FFN-free layers** (`n_ff == 0`): skip FFN entirely (Nemotron-Ultra-253B)
- **Standard layers** (`n_head > 0, n_head_kv > 0, n_ff > 0`): full attention + gated FFN with SiLU

No MoE. No recurrent layers. Pure transformer with per-layer flexibility.

## Complete Op Trace

### Embedding

| Op | Source | Vulkan |
|---|---|---|
| `GET_ROWS` | `llm_build_inp_embd` (line 510) — token embedding lookup | **YES** |

### Per-Layer: Attention Norm

| Op | Source | Vulkan |
|---|---|---|
| `FUSED_RMS_NORM` | `llm_build_norm` (line 662) — RMS norm + scale fused | **YES** |

Only emitted when `n_head > 0` (layers with attention).

### Per-Layer: QKV Projection

| Op | Source | Vulkan |
|---|---|---|
| `MUL_MAT` ×3 | `llm_build_mul_mat_qkv` (line 1774-1782) — Wq, Wk, Wv projections | **YES** |
| `RESHAPE` ×2 | `ggml_reshape_3d` (line 2369, 2376) — Q [head_dim, n_head, n_tokens], K similar | **YES** (zero-cost) |
| `ROPE` ×2 | `ggml_rope_ext` (line 2368, 2375) — rotary position embedding on Q and K | **YES** |

Only emitted for standard attention layers (`n_head > 0, n_head_kv > 0`). Linear attention layers emit only one `MUL_MAT` (wo projection).

### Per-Layer: KV Cache Store

| Op | Source | Vulkan |
|---|---|---|
| `VIEW` ×2 | `llm_build_kv_store` (line 572, 584) — cache write targets | **YES** (zero-cost) |
| `CPY` ×2 | `llm_build_kv_store` (line 575, 598) — K, V → cache | **YES** |
| `TRANSPOSE` | `llm_build_kv_store` (line 594) — V transpose (only if `v_trans`) | **YES** (zero-cost) |

### Per-Layer: Attention Compute

#### Flash Attention Path (`cparams.flash_attn = true`)

| Op | Source | Vulkan |
|---|---|---|
| `PERMUTE` | `llm_build_kqv` (line 1514) — Q permute [head_dim, n_tokens, n_head] | **YES** (zero-cost) |
| `VIEW` ×2 | `llm_build_kqv` (line 1518, 1550) — K, V views from cache | **YES** (zero-cost) |
| `FLASH_ATTN_EXT` | `llm_build_kqv` (line 1556) — fused flash attention | **YES**\* |
| `RESHAPE` | `ggml_reshape_2d` (line 1572) — output reshape | **YES** (zero-cost) |

\*Flash attention requires:
- Head size 128 (Nemotron standard) — supported
- KV type F16 — supported on all devices; Q4_0/Q8_0 also supported
- `subgroup_shuffle` — required on non-coopmat2 devices (Vega, RDNA2 both support this)
- Q type F32 — always the case

#### Non-Flash Path (`cparams.flash_attn = false`)

| Op | Source | Vulkan |
|---|---|---|
| `PERMUTE` | `llm_build_kqv` (line 1514) — Q permute | **YES** (zero-cost) |
| `VIEW` ×2 | `llm_build_kqv` (line 1518, 1578/1583) — K, V views from cache | **YES** (zero-cost) |
| `TRANSPOSE` + `CONT` | `llm_build_kqv` (line 1588) — V make contiguous (if `!v_trans`) | **YES** |
| `MUL_MAT` | `llm_build_kqv` (line 1594) — K×Q attention scores | **YES** |
| `SOFT_MAX` | `ggml_soft_max_ext` (line 1626) — softmax with mask and scale | **YES** |
| `MUL_MAT` | `llm_build_kqv` (line 1633) — V×softmax(KQ) | **YES** |
| `PERMUTE` | `llm_build_kqv` (line 1636) — output permute | **YES** (zero-cost) |
| `CONT` | `ggml_cont_2d` (line 1639) — make contiguous | **YES** |

Nemotron does NOT use `attn_soft_cap` (no Grok-style softcap), so `ggml_softcap`/`ggml_softcap_max` are never emitted.

### Per-Layer: Output Projection

| Op | Source | Vulkan |
|---|---|---|
| `MUL_MAT` | `llm_build_kqv` (line 1691) — Wo projection | **YES** |

### Per-Layer: Residual + FFN Norm

| Op | Source | Vulkan |
|---|---|---|
| `SCALE` | `build_deci` (line 2401) — residual scale (if `f_residual_scale != 0`) | **YES** |
| `ADD` | `build_deci` (line 2407) — attention output + residual | **YES** |
| `FUSED_RMS_NORM` | `llm_build_ffn` → `llm_build_norm` (line 823) — FFN input norm | **YES** |

### Per-Layer: FFN (Single-GPU, fused path)

When `cparams.fused_up_gate = true` (default) and weights have no biases (standard Nemotron):

| Op | Source | Vulkan |
|---|---|---|
| `FUSED_UP_GATE` | `llm_build_ffn` (line 835) — up×b \* silu(gate×b) in single dispatch | **YES** |
| `MUL_MAT` | `llm_build_ffn` (line 841) — down projection | **YES** |

### Per-Layer: FFN (Single-GPU, decomposed fallback)

When `fused_up_gate = false` or weights have biases:

| Op | Source | Vulkan |
|---|---|---|
| `MUL_MAT` | `llm_build_ffn` (line 868) — up projection | **YES** |
| `MUL_MAT` | `llm_build_ffn` (line 889) — gate projection | **YES** |
| `FUSED_MUL_UNARY` | `llm_build_ffn` (line 910) — silu(gate) \* up fused | **YES** |
| `MUL_MAT` | `llm_build_ffn` (line 968) — down projection | **YES** |

### Per-Layer: FFN (Multi-GPU split path)

When weight tensors have `extra` set (split across devices):

| Op | Source | Vulkan |
|---|---|---|
| `FUSED_RMS_NORM` | `do_split_norm` — per-device norm | **YES** |
| `FUSED_UP_GATE` | `llm_build_ffn` (line 774) — per-device fused FFN | **YES** |
| `MUL_MAT` | `llm_build_ffn` (line 780) — per-device down projection | **YES** |
| `CPY` (cast) | `llm_build_ffn` (line 787) — cast to reduce_type (F16) | **YES** |
| `REDUCE` | `llm_build_ffn` (line 811) — cross-device sum | N/A\*\* |
| `ADD` | `llm_build_ffn` (line 804) — residual add | **YES** |

\*\*`REDUCE` is not a standard Vulkan op — it's handled by our multi-GPU infrastructure (cross-device staging + `ADD`).

### Per-Layer: Residual

| Op | Source | Vulkan |
|---|---|---|
| `SCALE` | `build_deci` (line 2424) — FFN residual scale (if `f_residual_scale != 0`) | **YES** |
| `ADD` | `build_deci` (line 2427) — FFN output + residual | **YES** |

### Last Layer: Output Selection

| Op | Source | Vulkan |
|---|---|---|
| `GET_ROWS` | `build_deci` (line 2391-2392) — select output tokens | **YES** |

### Final: Output Norm + LM Head

| Op | Source | Vulkan |
|---|---|---|
| `FUSED_RMS_NORM` | `build_deci` (line 2439) — final RMS norm | **YES** |
| `MUL_MAT` | `build_deci` (line 2443) — lm_head projection | **YES** |
| `SCALE` | `build_deci` (line 2446) — logit scale (if `f_logit_scale != 0`) | **YES** |

## KQ Mask (Input Tensor)

| Path | Op | Vulkan |
|---|---|---|
| Flash attn, causal | Mask created directly as F16 — no op | N/A |
| Flash attn, non-causal | `CPY` F32→F16 via `ggml_cast` | **YES** |
| Non-flash | Mask stays F32 — no op | N/A |

## Summary: All Ops by Category

### Compute Ops (dispatch GPU work)

| Op | Count per layer | Vulkan |
|---|---|---|
| `FUSED_RMS_NORM` | 2 (attn_norm + ffn_norm) + 1 (final) | **YES** |
| `MUL_MAT` | 4-5 (Wq, Wk, Wv, Wo, down) | **YES** |
| `FUSED_UP_GATE` | 1 (fused FFN path) | **YES** |
| `ROPE` | 2 (Q, K) | **YES** |
| `FLASH_ATTN_EXT` | 1 (flash path) | **YES** |
| `SOFT_MAX` | 1 (non-flash path) | **YES** |
| `FUSED_MUL_UNARY` | 1 (decomposed FFN fallback) | **YES** |
| `CPY` | 2 (KV cache writes) | **YES** |
| `CONT` | 1-2 (V transpose, output) | **YES** |
| `ADD` | 2-3 (residuals) | **YES** |
| `SCALE` | 0-2 (residual/logit scale) | **YES** |
| `GET_ROWS` | 1-2 (embedding, output select) | **YES** |

### Zero-Cost Layout Ops (no GPU dispatch)

| Op | Vulkan |
|---|---|
| `RESHAPE` | **YES** |
| `VIEW` | **YES** |
| `PERMUTE` | **YES** |
| `TRANSPOSE` | **YES** |

## Verdict

**All ops in the Nemotron graph are Vulkan-supported.** Expected graph splits: **2-3** (the minimum for any model — typically embedding split and output split from the multi-GPU scheduler).

No new shaders or `supports_op` additions are needed for Nemotron.

## Conditional Ops NOT in the Standard Path

These ops can appear in `build_deci()` under non-default settings but are NOT Vulkan-supported:

| Op | Trigger | Vulkan | Impact |
|---|---|---|---|
| `HADAMARD` | `cparams.k_cache_hadamard = true` | **NO** | +2 splits per layer. Not default. |

## Flash Attention Hardware Requirements

For flash attention on Nemotron (head_size=128, KV type F16):

| GPU | coopmat2 | subgroup_shuffle | FA supported |
|---|---|---|---|
| RX 6800 XT (RDNA2) | No | Yes | **Yes** (scalar path) |
| RX Vega (GCN5) | No | Yes | **Yes** (scalar path) |
| Polaris (GCN 7790+) | No | ? | Check `subgroup_shuffle` |

If flash attention is not supported on a device, the non-flash path is used — all ops in that path are also Vulkan-supported.

## Files Referenced

| File | Lines | Role |
|---|---|---|
| `src/llama-build-context.cpp` | 2319-2454 | `build_deci()` — main graph builder |
| `src/llama-build-context.cpp` | 652-691 | `llm_build_norm()` — RMS/layer norm |
| `src/llama-build-context.cpp` | 730-998 | `llm_build_ffn()` — FFN with fused/decomposed paths |
| `src/llama-build-context.cpp` | 1487-1707 | `llm_build_kqv()` — attention compute |
| `src/llama-build-context.cpp` | 543-600 | `llm_build_kv_store()` — KV cache writes |
| `src/llama-build-context.cpp` | 1769-1807 | `llm_build_mul_mat_qkv()` — Q/K/V projections |
| `ggml/src/ggml-vulkan.cpp` | 11059-11407 | `ggml_backend_vk_supports_op()` — Vulkan op support |
