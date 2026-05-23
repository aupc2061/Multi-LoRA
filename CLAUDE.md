# Multi-LoRA Project — Session Reference

## What this project is

Research codebase for multi-concept LoRA composition in diffusion models.
Two pipelines exist:
1. **SDXL/SD1.5 selective mixing** (older, complete) — circuit-based module-level LoRA routing
2. **SD3.5 null-mask** (active, in `sd35_nullmask/`) — novel two-component composition method

The SD3.5 work is the active focus. Everything below refers to it unless stated otherwise.

---

## SD3.5 null-mask method

Two-component interference suppression for two LoRAs on SD3.5 (DiT/JointTransformerBlock):

**Component 1 — Lateral (spatial):** Early look-ahead pass extracts cross-attention maps per concept → argmax binary spatial masks → each LoRA's hidden-state delta is applied only to its owned patch region.

**Component 2 — Vertical (structural):** SVD of base model's *runtime hidden states* at the intervention block → project each LoRA's delta into the orthogonal complement of the base model's principal activation directions. Prevents LoRAs from overwriting base model geometry/pose.

**4 methods benchmarked:**
- `merge` — naive weighted average (baseline)
- `switch` — alternate adapters by timestep at `switch_step` (baseline)
- `sd35_mask_only` — spatial masking only
- `sd35_mask_nullproj` — spatial masking + null-space projection

**Per-step cost:** 4 forward passes (base + 2 LoRA captures + injection) × 28 steps = 112 passes per image.

---

## Environment

```bash
# Remote venv (activate before running anything)
source venv_sd35/bin/activate

# Model weights location
models/dit_lora/sd35/{adapter_id}/   # one subdir per adapter
```

## Adapter inventory (`sd35_dit_lora_info.json`)

| adapter_id | HuggingFace repo | Trigger | Category |
|---|---|---|---|
| `character_east_asian_beauty` | `TDN-M/East-asian-beauty` | *(none — use override: "beautiful Asian woman")* | character |
| `character_trpfrog` | `Prgckwb/trpfrog-sd3.5-large-lora` | `an icon of trpfrog` | character |
| `style_chinese_lineart` | `Shakker-Labs/SD3.5-LoRA-Chinese-Line-Art` | `Chinese line art` | style |
| `style_pixel_art` | `nerijs/pixel-art-3.5L` | `pixel art` | style |

**Download weights** (inside activated venv):
```bash
hf download Prgckwb/trpfrog-sd3.5-large-lora \
  --local-dir models/dit_lora/sd35/character_trpfrog

hf download Shakker-Labs/SD3.5-LoRA-Chinese-Line-Art \
  --local-dir models/dit_lora/sd35/style_chinese_lineart

hf download nerijs/pixel-art-3.5L \
  --include "pixel-art-3.5L-v2_000000300.safetensors" \
  --local-dir models/dit_lora/sd35/style_pixel_art

hf download TDN-M/East-asian-beauty \
  --local-dir models/dit_lora/sd35/character_east_asian_beauty
```

---

## Running the pipeline

**Dry-run preflight check (no GPU needed):**
```bash
python -m sd35_nullmask.runner \
  --config sd35_nullmask/configs/mixing_example.json \
  --preflight_only
```

**Single pair, all 4 methods** (set `dry_run: false` in config first):
```bash
python -m sd35_nullmask.runner \
  --config sd35_nullmask/configs/mixing_example.json
```

**Full benchmark sweep** (set `dry_run: false`):
```bash
python -m sd35_nullmask.main_benchmark \
  --config sd35_nullmask/configs/benchmark_example.json
```

**LLM evaluation** (runs after benchmark, no GPU needed):
```bash
python llm_evaluate_mixing.py \
  --benchmark_dir results/sd35_nullmask_benchmark \
  --methods merge,switch,sd35_mask_only,sd35_mask_nullproj \
  --image_style reality \
  --lora_info_path sd35_dit_lora_info.json \
  --eval_mode ranking
```

**Output structure:**
```
results/sd35_nullmask_benchmark/
  benchmark_plan.json
  benchmark_summary.json
  {pair_id}/
    report.json
    merge_seed111.png
    switch_seed111.png
    sd35_mask_only_seed111.png
    sd35_mask_nullproj_seed111.png
    ...
```

---

## Key config fields (`mixing_example.json` / `benchmark_example.json`)

| Field | Default | Notes |
|---|---|---|
| `dtype` | `bfloat16` | All 4 adapters require bfloat16 |
| `denoise_steps` | 28 | Standard for SD3.5 |
| `lookahead_steps` | `[2, 4]` | Look-ahead window for mask building |
| `switch_step` | 5 | Warmup steps before masking activates |
| `svd_rank` | 1 | Rank of null-space projection basis |
| `intervention_block_start` | -1 | -1 = auto-select last block (block 23) |
| `mask_binarize_tau` | 0.7 | τ quantile threshold for binary mask (LoRA-Shop §3.2) |
| `null_proj_mu` | -1.0 | -1 = hard projection; 0 = no projection; >0 = soft |
| `trigger_token_override` | see config | Required for east_asian_beauty (no native trigger) |
| `dry_run` | `true` | **Set to false to actually generate images** |

---

## File map — `sd35_nullmask/`

| File | Purpose |
|---|---|
| `inference.py` | Core engine: `AttentionCaptureHook`, `HiddenStateCaptureHook`, `HiddenStateInjectionHook`, `SD35NullMaskEngine` |
| `masking.py` | `build_exclusive_binary_masks`, `reduce_attention_to_patch_scores` |
| `projection.py` | `compute_structural_basis` (SVD), `project_delta_to_nullspace` |
| `backend.py` | `SD35PipelineBackend`: pipeline load, adapter load, `run_method`, `run_all_methods` |
| `benchmark.py` | `run_benchmark`: full generation loop over all pairs × methods × seeds |
| `runner.py` | CLI entry point for single-pair runs |
| `main_benchmark.py` | CLI entry point for full benchmark |
| `config.py` | `SD35NullMaskConfig`, `SD35NullMaskBenchmarkConfig` |
| `inventory.py` | `InventoryAdapter`, `build_inventory_context` (uses `dit_adapter_inventory`) |
| `prompting.py` | `resolve_trigger_phrase`, `build_pair_prompt` |

**Root-level helpers:**
- `dit_adapter_inventory.py` — local module powering `inventory.py` (load/flatten/lookup/discover)
- `sd35_dit_lora_info.json` — adapter registry
- `llm_evaluate_mixing.py` — LLM-as-judge eval (method-name agnostic, works as-is)

---

## SD3.5 architecture notes (for hook debugging)

- **Model:** `StableDiffusion3Pipeline` → `SD3Transformer2DModel`
- **Blocks:** 24 `JointTransformerBlock` instances (`transformer.transformer_blocks`)
- **Block output tuple convention:** `(encoder_hidden_states, hidden_states)` — **encoder FIRST**
- **Image tokens:** 4096 for 1024×1024 (64×64 patches, patch_size=2)
- **Text encoders:** CLIP-L (77 tokens) + CLIP-G (77) + T5-XXL (≤256) → up to 410 text tokens
- **Scheduler:** `FlowMatchEulerDiscreteScheduler`, t: 1000→0, `init_noise_sigma=1.0`
- **Cross-attention projection names:** `to_q` (image Q), `add_k_proj` (text K) on `block.attn`

---

## Benchmark pairs and what they test

| Pair | Tests |
|---|---|
| `character_east_asian_beauty` + `character_trpfrog` | **Primary spatial masking** — two subjects must coexist without identity blending |
| `character_east_asian_beauty` + `style_chinese_lineart` | **Null-projection** — style overwrites global structure; proj should preserve face geometry |
| `character_east_asian_beauty` + `style_pixel_art` | **Null-projection stress** — most aggressive global style in the set |
| `character_trpfrog` + `style_pixel_art` | **Style + compact subject** — pixel art on a character with strong spatial footprint |

The diff between `sd35_mask_only` and `sd35_mask_nullproj` scores is the ablation isolating the vertical interference component.

---

## Git

- Remote: `https://github.com/aupc2061/Multi-LoRA.git`
- Local git identity: `user.name=aupc2061`, `user.email=shayak2061@gmail.com` (set per-repo)
- Auth: PAT embedded in remote URL or SSH key alias `github-aupc2061`
