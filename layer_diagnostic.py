"""Fast layer diagnostic: zero out channels at candidate hook points and compare images.

No SAE, no dataset, no training needed. Just generates one base image and one
ablated image per (layer, lora_id, seed) to check if the layer controls
semantics (character/object change) or just fidelity (blur/contrast).

Usage:
    python layer_diagnostic.py --config configs/layer_diagnostic.json
    python layer_diagnostic.py --lora_id character_3 --hook_modules mid_block,up_blocks.0,up_blocks.1,conv_norm_out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from sae.feature_hooks import resolve_module
from utils import get_prompt, load_lora_info


# ── Channel-zeroing hook (no SAE) ───────────────────────────────────────────

class ChannelZeroingController:
    """Zeros out a fraction of channels at the hooked layer during selected steps."""

    def __init__(
        self,
        module: Any,
        zero_fraction: float = 0.5,
        target_steps: set[int] | None = None,
        apply_to: str = "cond",
        cfg_scale: float = 7.0,
    ) -> None:
        self.module = module
        self.zero_fraction = zero_fraction
        self.target_steps = target_steps
        self.apply_to = apply_to
        self.cfg_scale = cfg_scale

        self.enabled = False
        self.step_idx = 0
        self.records: list[dict[str, float | int]] = []
        self._handle = self.module.register_forward_hook(self._hook)

    def configure(self, enabled: bool) -> None:
        self.enabled = enabled
        self.step_idx = 0
        self.records = []

    def close(self) -> None:
        self._handle.remove()

    def _select_batch_indices(self, bsz: int, device: torch.device) -> torch.Tensor:
        if self.apply_to == "all":
            return torch.arange(bsz, device=device)
        if self.cfg_scale > 1.0 and bsz >= 2:
            return torch.arange(bsz // 2, bsz, device=device)
        return torch.arange(bsz, device=device)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        tensor = output[0] if isinstance(output, tuple) else output
        step = self.step_idx
        self.step_idx += 1

        should_apply = self.target_steps is None or step in self.target_steps
        if not should_apply or not self.enabled:
            return output

        bsz = tensor.shape[0]
        batch_idx = self._select_batch_indices(bsz, tensor.device)
        if batch_idx.numel() == 0:
            return output

        channels = tensor.shape[1]
        n_zero = max(1, int(channels * self.zero_fraction))

        # Deterministic channel selection based on step for reproducibility
        gen = torch.Generator(device="cpu").manual_seed(step * 1000 + 42)
        zero_channels = torch.randperm(channels, generator=gen)[:n_zero]

        tensor_out = tensor.clone()
        pre_abs_mean = tensor[batch_idx[:, None], zero_channels[None, :]].abs().mean().item()
        tensor_out[batch_idx[:, None], zero_channels[None, :]] = 0.0
        edit_abs_mean = (tensor[batch_idx[:, None], zero_channels[None, :]] - tensor_out[batch_idx[:, None], zero_channels[None, :]]).abs().mean().item()

        self.records.append(
            {
                "step": int(step),
                "num_zeroed_channels": int(n_zero),
                "zeroed_abs_mean_pre": float(pre_abs_mean),
                "edit_abs_mean": float(edit_abs_mean),
            }
        )

        if isinstance(output, tuple):
            return (tensor_out, *output[1:])
        return tensor_out


# ── Arg parsing ──────────────────────────────────────────────────────────────

def load_config(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    return payload.get("diagnostic", payload)


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    bs_args, _ = bootstrap.parse_known_args()

    defaults: dict[str, Any] = {}
    if bs_args.config:
        defaults = load_config(bs_args.config)

    p = argparse.ArgumentParser(description="Fast layer diagnostic for hook point selection.")
    p.add_argument("--config", type=str, default=None)

    p.add_argument("--lora_id", type=str, default="character_3")
    p.add_argument("--hook_modules", type=str, default="up_blocks.3,up_blocks.2,mid_block,conv_norm_out",
                   help="Comma-separated UNet module paths to test.")
    p.add_argument("--zero_fraction", type=float, default=1.0, help="Fraction of channels to zero.")
    p.add_argument("--step_mode", type=str, default="all", help="last|all|comma-separated step ids")
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--compute_latent_mse", action="store_true")

    p.add_argument("--model_name", type=str, default="SG161222/Realistic_Vision_V5.1_noVAE")
    p.add_argument("--custom_pipeline", type=str, default="./pipelines/sd1.5_0.26.3")
    p.add_argument("--image_style", type=str, default="reality", choices=["anime", "reality"])
    p.add_argument("--lora_info_path", type=str, default="lora_info.json")
    p.add_argument("--lora_path", type=str, default="models/lora")
    p.add_argument("--lora_scale", type=float, default=0.8)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--denoise_steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=7.0)
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--apply_to", type=str, default="all", choices=["cond", "all"])

    p.add_argument("--out_dir", type=str, default="sae_data/layer_diagnostic")

    if defaults:
        valid = {a.dest for a in p._actions}
        p.set_defaults(**{k: v for k, v in defaults.items() if k in valid})

    return p.parse_args()


def parse_target_steps(spec: str, num_steps: int) -> set[int] | None:
    spec = spec.strip().lower()
    if spec == "all":
        return None
    if spec == "last":
        return {num_steps - 1}
    steps = {int(tok.strip()) for tok in spec.split(",") if tok.strip()}
    for step in steps:
        if step < 0 or step >= num_steps:
            raise ValueError(f"Step index out of range: {step} for num_steps={num_steps}")
    return steps


def infer_prompt(image_style: str, lora_info_path: str, lora_id: str) -> str:
    lora_info = load_lora_info(image_style, lora_info_path)
    init_prompt, _ = get_prompt(image_style)
    for group in lora_info.values():
        for lora in group:
            if lora["id"] == lora_id:
                return init_prompt + ", " + ", ".join(lora["trigger"])
    raise ValueError(f"LoRA id not found: {lora_id}")


def pick_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "float16" else torch.float32


def run_pipeline_once(
    pipeline: Any,
    prompt: str,
    negative_prompt: str,
    args: argparse.Namespace,
    output_type: str,
):
    pipeline.enable_lora()
    pipeline.set_adapters([args.lora_id])
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    return pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.denoise_steps,
        guidance_scale=args.cfg_scale,
        generator=gen,
        output_type=output_type,
        return_dict=False,
        cross_attention_kwargs={"scale": args.lora_scale},
    )


def generate_baseline_outputs(
    pipeline: Any,
    prompt: str,
    negative_prompt: str,
    args: argparse.Namespace,
) -> tuple[Any, torch.Tensor | None]:
    """Generate the shared no-ablation baseline once for all hook comparisons."""
    base_result = run_pipeline_once(
        pipeline=pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        args=args,
        output_type="pil",
    )
    base_img = base_result[0][0]

    base_latents = None
    if args.compute_latent_mse:
        base_latent_result = run_pipeline_once(
            pipeline=pipeline,
            prompt=prompt,
            negative_prompt=negative_prompt,
            args=args,
            output_type="latent",
        )
        base_latents = base_latent_result[0]

    return base_img, base_latents


# ── Main ─────────────────────────────────────────────────────────────────────

def generate_ablated_outputs(
    pipeline: Any,
    hook_module: Any,
    controller_kwargs: dict[str, Any],
    prompt: str,
    negative_prompt: str,
    args: argparse.Namespace,
    base_latents: torch.Tensor | None,
) -> tuple:
    """Generate ablated outputs for one hook and compare against shared baseline latents."""
    controller = ChannelZeroingController(module=hook_module, **controller_kwargs)

    # Ablated
    controller.configure(enabled=True)
    abl_result = run_pipeline_once(
        pipeline=pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        args=args,
        output_type="pil",
    )
    abl_img = abl_result[0][0]
    pil_records = list(controller.records)

    latents_mse = None
    if args.compute_latent_mse and base_latents is not None:
        controller.configure(enabled=True)
        abl_latent_result = run_pipeline_once(
            pipeline=pipeline,
            prompt=prompt,
            negative_prompt=negative_prompt,
            args=args,
            output_type="latent",
        )
        abl_latents = abl_latent_result[0]
        latents_mse = torch.mean((base_latents - abl_latents) ** 2).item()

    controller.close()
    return abl_img, pil_records, latents_mse


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hook_modules = [tok.strip() for tok in args.hook_modules.split(",") if tok.strip()]
    target_steps = parse_target_steps(args.step_mode, args.denoise_steps)

    # Load pipeline once
    pipeline = DiffusionPipeline.from_pretrained(
        args.model_name,
        custom_pipeline=args.custom_pipeline,
        use_safetensors=True,
        torch_dtype=pick_dtype(args.dtype),
    ).to(args.device)

    sched_cfg = dict(pipeline.scheduler.config)
    sched_cfg["algorithm_type"] = "dpmsolver++"
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(sched_cfg)

    lora_root = Path(args.lora_path) / args.image_style
    lora_info = load_lora_info(args.image_style, args.lora_info_path)
    all_ids = [l["id"] for g in lora_info.values() for l in g]
    for lid in all_ids:
        pipeline.load_lora_weights(str(lora_root), weight_name=f"{lid}.safetensors", adapter_name=lid)

    prompt = infer_prompt(args.image_style, args.lora_info_path, args.lora_id)
    _, negative_prompt = get_prompt(args.image_style)

    controller_kwargs = {
        "zero_fraction": args.zero_fraction,
        "target_steps": target_steps,
        "apply_to": args.apply_to,
        "cfg_scale": args.cfg_scale,
    }

    base_img, base_latents = generate_baseline_outputs(
        pipeline=pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        args=args,
    )

    base_path = out_dir / f"base_{args.lora_id}_seed{args.seed}.png"
    base_img.save(base_path)
    print(f"Saved shared baseline: {base_path.name}")

    # List UNet top-level modules for reference
    print("Available top-level UNet modules:")
    for name, _ in pipeline.unet.named_children():
        print(f"  {name}")
    print()

    manifest: list[dict[str, Any]] = []

    for hook_name in hook_modules:
        print(f"Testing layer: {hook_name} ...")
        try:
            hook_mod = resolve_module(pipeline.unet, hook_name)
        except ValueError as e:
            print(f"  SKIP — {e}")
            continue

        abl_img, pil_records, latents_mse = generate_ablated_outputs(
            pipeline, hook_mod, controller_kwargs, prompt, negative_prompt, args, base_latents
        )

        safe_name = hook_name.replace(".", "_")
        abl_path = out_dir / f"ablated_{safe_name}_{args.lora_id}_seed{args.seed}.png"
        abl_img.save(abl_path)

        mean_zeroed_abs = 0.0
        mean_edit_abs = 0.0
        if pil_records:
            mean_zeroed_abs = float(sum(float(r["zeroed_abs_mean_pre"]) for r in pil_records) / len(pil_records))
            mean_edit_abs = float(sum(float(r["edit_abs_mean"]) for r in pil_records) / len(pil_records))

        manifest.append({
            "hook_module": hook_name,
            "lora_id": args.lora_id,
            "seed": args.seed,
            "zero_fraction": args.zero_fraction,
            "step_mode": args.step_mode,
            "apply_to": args.apply_to,
            "num_edited_steps": len(pil_records),
            "mean_zeroed_abs_pre": mean_zeroed_abs,
            "mean_edit_abs": mean_edit_abs,
            "latents_mse": latents_mse,
            "base_image": base_path.name,
            "ablated_image": abl_path.name,
        })

        print(
            f"  Saved: {base_path.name} / {abl_path.name} | "
            f"edited_steps={len(pil_records)} mean_zeroed_abs={mean_zeroed_abs:.6f} "
            f"latents_mse={0.0 if latents_mse is None else latents_mse:.6f}"
        )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDone. {len(manifest)} layer(s) tested. Results in {out_dir}")


if __name__ == "__main__":
    main()
