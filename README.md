# Atlas Spark Recipe Registry

Official sparkrun recipes for [Atlas](https://github.com/Atlas-Inf/atlas) — the pure-Rust & CUDA LLM inference engine engineered for NVIDIA DGX Spark (GB10 SM121) and prosumer silicon. Recipes target the verified `ghcr.io/atlas-inf/atlas-gb10:latest` and `azeezish/atlas-gb10:latest` images and the `atlas` runtime in [sparkrun](https://github.com/spark-arena/sparkrun).

## Usage

The `@atlas` namespace is reserved in sparkrun and pre-registered in standard installs:

```bash
# Flagship: Qwen 3.8 Flash-Next (~180B hybrid MoE in ~90 GB resident VRAM)
sparkrun run @atlas/qwen3.8-flash-next-nvfp4 --hosts localhost

# Nemotron 3.5 Lightning 30B with DSpark speculative drafter (sub-10ms decode)
sparkrun run @atlas/nemotron-3.5-lightning-30b-a3b-nvfp4-dspark --hosts localhost

# Qwen 3.8 27B low-latency dense hybrid
sparkrun run @atlas/qwen3.8-27b-nvfp4-latency --hosts localhost

# Multi-node Expert Parallelism across 2 Sparks (EP=2)
sparkrun run @atlas/deepseek-v4-flash-nvfp4-ep2 --hosts <spark-1>,<spark-2>
```

If you are on a custom build or older sparkrun release without the default reservation, register it manually:

```bash
sparkrun registry add https://github.com/Atlas-Inf/sparkrun-recipes.git
```

### Inspecting and Listing Recipes

The `atlas` registry ships in sparkrun's catalog. To list or inspect recipes:

```bash
sparkrun recipe list --registry atlas                  # List all Atlas recipes
sparkrun recipe show @atlas/qwen3.8-flash-next-nvfp4   # Inspect recipe flags & defaults
sparkrun recipe show @atlas/nemotron-3.5-lightning-30b-a3b-nvfp4-dspark
```

---

## Catalogue (Organized by Vendor & Family)

The recipe catalogue mirrors the structure and prioritization on [atlasinference.dev/#models](https://atlasinference.dev/#models):

### 1. Qwen

#### Qwen3.8 (Flagship)
| Recipe | Model Checkpoint | Topology | Description |
|---|---|:---:|---|
| `qwen3.8-27b-nvfp4` | `unsloth/Qwen3.8-27B-NVFP4` | Single GB10 | **DEFAULT FLAGSHIP**. 27B dense hybrid GDN + attention, NVFP4, MTP speculative decoding, FP8 KV cache, 23.59 tok/s single-stream |
| `qwen3.8-27b-nvfp4-latency` | `unsloth/Qwen3.8-27B-NVFP4` | Single GB10 | Low-concurrency / interactive profile tuned for minimal single-stream time-to-first-token and decode latency |
| `qwen3.8-27b-nvfp4-throughput` | `unsloth/Qwen3.8-27B-NVFP4` | Single GB10 | Concurrency profile measured to beat vLLM from 1 to 128 streams on GB10 |
| `qwen3.8-flash-next-nvfp4` | `RadixArk/Qwen3.8-Flash-Next-NVFP4` | Single GB10 | ~180B hybrid MoE, 8K context / 8K prefill, BF16 KV, MTP K=1, 47.7 GB PLE n-gram parallel `pread` NVMe offload (~95% util, 750–800 tok/s prefill, 36.7 tok/s decode) |
| `qwen3.8-flash-next-nvfp4-throughput` | `RadixArk/Qwen3.8-Flash-Next-NVFP4` | Single GB10 | 8K multi-sequence profile (max_num_seqs 4), BF16 KV, optimized for batched concurrent throughput |
| `qwen3.8-27b-nvfp4-unsloth` | `unsloth/Qwen3.8-27B-NVFP4` | Single GB10 | Agentic gate config: thinking ON, BF16 head + BF16 KV, 32K context, MTP K=4, SLAi scheduler |
| `qwen3.8-27b-nvfp4-unsloth-bfcl` | `unsloth/Qwen3.8-27B-NVFP4` | Single GB10 | BFCL v4 agentic tool-use benchmark profile |

#### Qwen3.6
| Recipe | Model Checkpoint | Topology | Description |
|---|---|:---:|---|
| `qwen3.6-35b-a3b-fp8-mtp` | `Qwen/Qwen3.6-35B-A3B-FP8` | Single GB10 | Native FP8, BF16 head + BF16 KV, 64K context, MTP K=2, live tool-call streaming |
| `qwen3.6-35b-a3b-nvfp4` | `nvidia/Qwen3.6-35B-A3B-NVFP4` | Single GB10 | NVFP4 MoE weights, MTP K=1, calibrated FP8 KV (128K context) |
| `qwen3.6-35b-a3b-fp8-bf16head` | `Qwen/Qwen3.6-35B-A3B-FP8` | Single GB10 | 32K safe profile with BF16 head/KV |
| `qwen3.6-35b-a3b-fp8-nvfp4head` | `Qwen/Qwen3.6-35B-A3B-FP8` | Single GB10 | NVFP4 lm-head sibling, lower memory footprint |
| `qwen3.6-27b-fp8-mtp` | `Qwen/Qwen3.6-27B-FP8` | Single GB10 | Dense hybrid SSM+Attn, MTP K=1, 60K context |
| `qwen3.6-27b-nvfp4` | `nvidia/Qwen3.6-27B-NVFP4` | Single GB10 | Dense hybrid SSM+Attn, MTP K=1, BF16 KV |
| `qwen3.6-27b-nvfp4-prefill-record` | `nvidia/Qwen3.6-27B-NVFP4` | Single GB10 | Prefill-optimized record profile |
| `qwen3.6-27b-nvfp4-unsloth` | `unsloth/Qwen3.6-27B-NVFP4` | Single GB10 | Mixed-precision NVFP4/FP8 layout loader, BFCL gate profile |
| `qwen3.6-27b-fp8` | `Qwen/Qwen3.6-27B-FP8` | Single GB10 | Pure FP8 dense baseline |

#### Qwen3.5
| Recipe | Model Checkpoint | Topology | Description |
|---|---|:---:|---|
| `qwen3.5-35b-a3b-nvfp4` | `Sehyo/Qwen3.5-35B-A3B-NVFP4` | Single GB10 | MTP K=2, ~131 tok/s |
| `qwen3.5-27b-dense-nvfp4` | `Kbenkhaled/Qwen3.5-27B-NVFP4` | Single GB10 | Dense hybrid GDN + attention |
| `qwen3.5-122b-a10b-nvfp4-single` | `Sehyo/Qwen3.5-122B-A10B-NVFP4` | Single GB10 | Tight KV budget, all 256 experts resident on one GB10 |
| `qwen3.5-122b-a10b-nvfp4-ep2` | `Sehyo/Qwen3.5-122B-A10B-NVFP4` | EP=2 (2 Sparks) | Dual-node Expert Parallelism + MTP K=2 |
| `qwen3.5-0.8b-bf16-atlas` | `Qwen/Qwen3.5-0.8B` | Single GB10 | Lightweight edge development target |

#### Qwen Extensions
| Recipe | Model Checkpoint | Topology | Description |
|---|---|:---:|---|
| `qwen3-next-80b-a3b-nvfp4` | `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` | Single GB10 | SSM + Attention + MoE with MTP speculative decoding |
| `qwen3-coder-next-fp8` | `Qwen/Qwen3-Coder-Next-FP8` | Single GB10 | Native FP8 coding target, BF16 KV cache |
| `qwen3-vl-30b-a3b-nvfp4` | `ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4` | Single GB10 | Vision-language hybrid MoE with image/video inputs |

---

### 2. Gemma

| Subfamily | Recipe | Model Checkpoint | Topology | Description |
|---|---|---|:---:|---|
| **Gemma-4** | `gemma-4-26b-a4b-nvfp4` | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16` | Single GB10 | 26B / 4B active MoE with GeGLU activation |
| **Gemma-4** | `gemma-4-31b-nvfp4` | `nvidia/Gemma-4-31B-IT-NVFP4` | Single GB10 | Dense 31B, sliding + full attention, gemma4 parser |
| **Gemma Diffusion** | `diffusion-gemma-bf16` | `google/diffusion-gemma` | Single GB10 | BF16 diffusion generative weights |
| **Gemma Diffusion** | `diffusion-gemma-fp8-dynamic` | `google/diffusion-gemma` | Single GB10 | Dynamic FP8 activation profile |

---

### 3. Nemotron

| Subfamily | Recipe | Model Checkpoint | Topology | Description |
|---|---|---|:---:|---|
| **Nemotron-3.5 Lightning** | `nemotron-3.5-lightning-30b-a3b-nvfp4-dspark` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | Single GB10 | **NEW**. Mamba-2 SSM + Attention + MoE with 6-layer DSpark drafter (gamma=4 / K=3 verify), sub-10ms decode |
| **Nemotron-3 Nano** | `nemotron-3-nano-30b-a3b-nvfp4` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | Single GB10 | 30B / 3B active Mamba-2 hybrid MoE |
| **Nemotron-3 Super** | `nemotron-3-super-120b-a12b-nvfp4` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | Single GB10 | 120B / 12B active hybrid MoE |

---

### 4. Mistral

| Subfamily | Recipe | Model Checkpoint | Topology | Description |
|---|---|---|:---:|---|
| **Mistral-Small-4** | `mistral-small-4-119b-nvfp4` | `mistralai/Mistral-Small-4-119B-2603-NVFP4` | Single GB10 | MLA attention architecture, mandatory BF16 KV cache |

---

### 5. MiniMax

| Subfamily | Recipe | Model Checkpoint | Topology | Description |
|---|---|---|:---:|---|
| **MiniMax-M2.7** | `minimax-m2.7-nvfp4-ep2` | `lukealonso/MiniMax-M2.7-NVFP4` | EP=2 (2 Sparks) | 229B / ~10B active MoE with 256 experts across 2 nodes |

---

### 6. DeepSeek

| Subfamily | Recipe | Model Checkpoint | Topology | Description |
|---|---|---|:---:|---|
| **DeepSeek-V4** | `deepseek-v4-flash-nvfp4-ep2` | `deepseek-ai/DeepSeek-V4-Flash-NVFP4` | EP=2 (2 Sparks) | Dual-node Expert Parallelism MoE profile |

---

## Directory Layout

```
recipes/
├── qwen3.8/
│   ├── qwen3.8-27b-nvfp4.yaml
│   ├── qwen3.8-27b-nvfp4-latency.yaml
│   ├── qwen3.8-27b-nvfp4-throughput.yaml
│   ├── qwen3.8-flash-next-nvfp4.yaml
│   ├── qwen3.8-flash-next-nvfp4-throughput.yaml
│   ├── qwen3.8-27b-nvfp4-unsloth.yaml
│   └── qwen3.8-27b-nvfp4-unsloth-bfcl.yaml
├── qwen3.6/
│   ├── qwen3.6-35b-a3b-fp8-mtp.yaml
│   ├── qwen3.6-35b-a3b-nvfp4.yaml
│   ├── ... (7 additional Qwen3.6 recipes)
├── qwen3.5/
│   ├── qwen3.5-35b-a3b-nvfp4.yaml
│   ├── qwen3.5-27b-dense-nvfp4.yaml
│   ├── qwen3.5-122b-a10b-nvfp4-single.yaml
│   ├── qwen3.5-122b-a10b-nvfp4-ep2.yaml
│   └── qwen3.5-0.8b-bf16-atlas.yaml
├── qwen3-next/
│   └── qwen3-next-80b-a3b-nvfp4.yaml
├── qwen3-coder-next/
│   └── qwen3-coder-next-fp8.yaml
├── qwen3-vl/
│   └── qwen3-vl-30b-a3b-nvfp4.yaml
├── gemma4/
│   ├── gemma-4-26b-a4b-nvfp4.yaml
│   └── gemma-4-31b-nvfp4.yaml
├── diffusion-gemma/
│   ├── diffusion-gemma-bf16.yaml
│   └── diffusion-gemma-fp8-dynamic.yaml
├── nemotron-3.5-lightning/
│   └── nemotron-3.5-lightning-30b-a3b-nvfp4-dspark.yaml
├── nemotron-3-nano/
│   └── nemotron-3-nano-30b-a3b-nvfp4.yaml
├── nemotron-3-super/
│   └── nemotron-3-super-120b-a12b-nvfp4.yaml
├── mistral-small-4/
│   └── mistral-small-4-119b-nvfp4.yaml
├── minimax-m2.7/
│   └── minimax-m2.7-nvfp4-ep2.yaml
└── deepseek-v4/
    └── deepseek-v4-flash-nvfp4-ep2.yaml
```

Sparkrun recurses through the `recipes/` directory structure; each recipe is referenced by qualified stem (e.g. `@atlas/qwen3.8-flash-next-nvfp4`).

---

## Hardware & Runtime Configuration

Key production-validated settings encoded across recipes:

- **Qwen 3.8 Flash-Next**: Uses `RadixArk/Qwen3.8-Flash-Next-NVFP4`. Binds the specialized SM121 `qwen3.8-flash-next` kernel target, BF16 KV cache (avoids clipping without scale tensors), and dynamically streams the 47.7 GB n-gram table from disk to stay under 90 GB VRAM.
- **Nemotron 3.5 Lightning**: Uses `--dflash` with drafter `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark`. Requires `ATLAS_DFLASH_OPTION_B=1`. Set `ATLAS_NO_TOOL_INJECT=1` for +15.58 BFCL accuracy boost.
- **Mistral Small 4**: Enforces `kv_cache_dtype: bf16` to protect the MLA compressed latent.
- **Qwen3-Coder-Next-FP8**: Configures `ssm_cache_slots: 0`, `oom_guard_mb: 1024`, and `kv_cache_dtype: bf16`.
- **EP=2 Multi-Node**: Dual-node configurations (MiniMax M2.7, Qwen3.5-122B, DeepSeek V4) pass matched `--speculative` and `--mtp-quantization` across ranks.

---

## Related Links

- **Website**: [https://atlasinference.dev](https://atlasinference.dev)
- **Atlas Inference Engine**: [https://github.com/Atlas-Inf/atlas](https://github.com/Atlas-Inf/atlas)
- **Sparkrun**: [https://github.com/spark-arena/sparkrun](https://github.com/spark-arena/sparkrun)
- **Docker Image**: `ghcr.io/atlas-inf/atlas-gb10:latest` / `azeezish/atlas-gb10:latest`
- **Discord**: [Join our Discord](https://discord.com/invite/6vDbKaKrKD)
- **X**: [@AtlasInference](https://x.com/AtlasInference)

---

## License

AGPL-3.0 — see [LICENSE](LICENSE). Matches the upstream [Atlas](https://github.com/Atlas-Inf/atlas) license.

<sub><b>Continuity notice.</b> Atlas is continuing. This repository, the <a href="https://github.com/Atlas-Inf">Atlas-Inf</a> GitHub organization, and <a href="https://atlasinference.dev">atlasinference.dev</a> are the replacement official Atlas channels. The existing website and GitHub repository remain disputed Atlas assets that have not been relinquished.</sub>
