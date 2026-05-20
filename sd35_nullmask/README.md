# SD3.5 Null-Masked Residual LoRA Mixing

This folder contains the SD3.5-specific prototype scaffold for the DiT null-mask mixing path.

Implemented here:

- SD3.5 config and benchmark config parsing
- adapter-inventory preflight using the curated SD3.5 manifest
- trigger-phrase resolution from inventory or explicit overrides
- exclusive binary mask construction from per-adapter patch scores
- SVD-based activation null projection utilities
- SD3.5 backend scaffold with runtime preflight
- dry-run runner and benchmark planner

Not fully implemented here yet:

- stable-diffusion-3.5 residual-hook intervention execution
- look-ahead attention extraction from live SD3.5 runs
- full image generation benchmark loop

## Example commands

Preflight only:

```bash
python -m sd35_nullmask.runner --config sd35_nullmask/configs/mixing_example.json --preflight_only
```

Benchmark planning only:

```bash
python -m sd35_nullmask.main_benchmark --config sd35_nullmask/configs/benchmark_example.json --dry_run
```
