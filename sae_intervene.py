from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from sae import SparseAutoencoder
from sae.feature_hooks import resolve_module
from utils import get_prompt, load_lora_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intervene on selected SAE features during denoising.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--lora_id", type=str, required=True)
    parser.add_argument("--feature_indices", type=str, default="")
    parser.add_argument("--delta_json", type=str, default="")
    parser.add_argument("--topk_from_delta", type=int, default=16)
    parser.add_argument("--ablate_scale", type=float, default=0.0)
    parser.add_argument("--target_steps", type=str, default="last", help="last|all|comma-separated step ids")
    parser.add_argument("--apply_to", type=str, default="cond", choices=["cond", "all"])

    parser.add_argument("--model_name", type=str, default="SG161222/Realistic_Vision_V5.1_noVAE")
    parser.add_argument("--custom_pipeline", type=str, default="./pipelines/sd1.5_0.26.3")
    parser.add_argument("--image_style", type=str, default="reality", choices=["anime", "reality"])
    parser.add_argument("--lora_info_path", type=str, default="lora_info.json")
    parser.add_argument("--lora_path", type=str, default="models/lora")
    parser.add_argument("--lora_scale", type=float, default=0.8)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--denoise_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--hook_module", type=str, default="conv_norm_out")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--save_dir", type=str, default="sae_data/intervention")
    return parser.parse_args()


def pick_dtype(dtype_name: str) -> torch.dtype:
    return torch.float16 if dtype_name == "float16" else torch.float32


def infer_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    lora_info = load_lora_info(args.image_style, args.lora_info_path)
    init_prompt, _ = get_prompt(args.image_style)
    for group in lora_info.values():
        for lora in group:
            if lora["id"] == args.lora_id:
                return init_prompt + ", " + ", ".join(lora["trigger"])
    raise ValueError(f"LoRA id not found in metadata: {args.lora_id}")


def parse_indices(args: argparse.Namespace) -> list[int]:
    if args.feature_indices.strip():
        return [int(tok.strip()) for tok in args.feature_indices.split(",") if tok.strip()]

    if args.delta_json:
        payload = json.loads(Path(args.delta_json).read_text(encoding="utf-8"))
        top_features = payload.get("top_features", [])
        if not top_features:
            raise ValueError("delta_json has no top_features.")
        return [int(entry["feature"]) for entry in top_features[: args.topk_from_delta]]

    raise ValueError("Specify either --feature_indices or --delta_json.")


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


class SAEInterventionController:
    def __init__(
        self,
        module: Any,
        sae: SparseAutoencoder,
        mean: torch.Tensor,
        std: torch.Tensor,
        feature_indices: Iterable[int],
        apply_to: str,
        cfg_scale: float,
    ) -> None:
        self.module = module
        self.sae = sae
        self.mean = mean
        self.std = std
        self.feature_indices = torch.tensor(list(feature_indices), dtype=torch.long, device=mean.device)
        self.apply_to = apply_to
        self.cfg_scale = cfg_scale

        self.enabled = False
        self.ablate_scale = 0.0
        self.target_steps: set[int] | None = None
        self.step_idx = 0
        self.records: list[dict[str, float | int]] = []

        self._handle = self.module.register_forward_hook(self._hook)

    def configure(self, enabled: bool, ablate_scale: float, target_steps: set[int] | None) -> None:
        self.enabled = enabled
        self.ablate_scale = ablate_scale
        self.target_steps = target_steps
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
        if not should_apply:
            return output

        bsz, channels, height, width = tensor.shape
        batch_idx = self._select_batch_indices(bsz, tensor.device)
        if batch_idx.numel() == 0:
            return output

        selected = tensor[batch_idx].float()
        tokens = selected.permute(0, 2, 3, 1).reshape(-1, channels)
        x_norm = (tokens - self.mean) / self.std

        with torch.no_grad():
            z = self.sae.encode(x_norm)
            pre = z[:, self.feature_indices].abs().mean().item()

            if self.enabled:
                z[:, self.feature_indices] = z[:, self.feature_indices] * self.ablate_scale

            post = z[:, self.feature_indices].abs().mean().item()
            x_hat_norm = self.sae.decode(z)

        x_hat = x_hat_norm * self.std + self.mean
        selected_hat = x_hat.reshape(selected.shape[0], height, width, channels).permute(0, 3, 1, 2)
        tensor_out = tensor.clone()
        tensor_out[batch_idx] = selected_hat.to(dtype=tensor.dtype)

        self.records.append(
            {
                "step": int(step),
                "feature_abs_mean_pre": float(pre),
                "feature_abs_mean_post": float(post),
                "edited": bool(self.enabled),
            }
        )

        if isinstance(output, tuple):
            return (tensor_out, *output[1:])
        return tensor_out


def run_once(
    pipeline: Any,
    controller: SAEInterventionController,
    prompt: str,
    negative_prompt: str,
    args: argparse.Namespace,
    intervene: bool,
    target_steps: set[int] | None,
    output_type: str,
):
    controller.configure(enabled=intervene, ablate_scale=args.ablate_scale, target_steps=target_steps)

    pipeline.enable_lora()
    pipeline.set_adapters([args.lora_id])

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.denoise_steps,
        guidance_scale=args.cfg_scale,
        generator=generator,
        output_type=output_type,
        return_dict=False,
        cross_attention_kwargs={"scale": args.lora_scale},
    )

    return result, list(controller.records)


def main() -> None:
    args = parse_args()
    feature_indices = parse_indices(args)
    target_steps = parse_target_steps(args.target_steps, args.denoise_steps)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sae = SparseAutoencoder(ckpt["input_dim"], ckpt["latent_dim"])  # type: ignore[arg-type]
    sae.load_state_dict(ckpt["model_state"])
    sae = sae.to(args.device).eval()

    mean = ckpt["mean"].to(args.device)
    std = ckpt["std"].to(args.device)

    pipeline = DiffusionPipeline.from_pretrained(
        args.model_name,
        custom_pipeline=args.custom_pipeline,
        use_safetensors=True,
        torch_dtype=pick_dtype(args.dtype),
    ).to(args.device)

    schedule_config = dict(pipeline.scheduler.config)
    schedule_config["algorithm_type"] = "dpmsolver++"
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(schedule_config)

    lora_root = Path(args.lora_path) / args.image_style
    pipeline.load_lora_weights(str(lora_root), weight_name=f"{args.lora_id}.safetensors", adapter_name=args.lora_id)

    prompt = infer_prompt(args)
    _, negative_prompt = get_prompt(args.image_style)

    hook_target = resolve_module(pipeline.unet, args.hook_module)
    controller = SAEInterventionController(
        module=hook_target,
        sae=sae,
        mean=mean,
        std=std,
        feature_indices=feature_indices,
        apply_to=args.apply_to,
        cfg_scale=args.cfg_scale,
    )

    base_result, base_records = run_once(
        pipeline=pipeline,
        controller=controller,
        prompt=prompt,
        negative_prompt=negative_prompt,
        args=args,
        intervene=False,
        target_steps=target_steps,
        output_type="latent",
    )
    int_result, int_records = run_once(
        pipeline=pipeline,
        controller=controller,
        prompt=prompt,
        negative_prompt=negative_prompt,
        args=args,
        intervene=True,
        target_steps=target_steps,
        output_type="latent",
    )

    controller.close()

    base_latents = base_result[0]
    int_latents = int_result[0]
    latents_mse = torch.mean((base_latents - int_latents) ** 2).item()

    pre_means = [row["feature_abs_mean_pre"] for row in int_records]
    post_means = [row["feature_abs_mean_post"] for row in int_records]
    mean_pre = float(sum(pre_means) / max(len(pre_means), 1))
    mean_post = float(sum(post_means) / max(len(post_means), 1))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "lora_id": args.lora_id,
        "num_selected_features": len(feature_indices),
        "selected_features": feature_indices,
        "ablate_scale": args.ablate_scale,
        "target_steps": sorted(target_steps) if target_steps is not None else "all",
        "latents_mse": latents_mse,
        "feature_abs_mean_pre": mean_pre,
        "feature_abs_mean_post": mean_post,
        "feature_abs_mean_delta": mean_post - mean_pre,
        "baseline_records": base_records,
        "intervention_records": int_records,
        "prompt": prompt,
    }

    summary_path = save_dir / f"summary_{args.lora_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.save_images:
        hook_target = resolve_module(pipeline.unet, args.hook_module)
        controller_img = SAEInterventionController(
            module=hook_target,
            sae=sae,
            mean=mean,
            std=std,
            feature_indices=feature_indices,
            apply_to=args.apply_to,
            cfg_scale=args.cfg_scale,
        )

        base_img_res, _ = run_once(
            pipeline=pipeline,
            controller=controller_img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            args=args,
            intervene=False,
            target_steps=target_steps,
            output_type="pil",
        )
        int_img_res, _ = run_once(
            pipeline=pipeline,
            controller=controller_img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            args=args,
            intervene=True,
            target_steps=target_steps,
            output_type="pil",
        )
        controller_img.close()

        base_img = base_img_res[0][0]
        int_img = int_img_res[0][0]
        base_img.save(save_dir / f"base_{args.lora_id}.png")
        int_img.save(save_dir / f"intervene_{args.lora_id}.png")

    print(f"Saved intervention summary to {summary_path}")


if __name__ == "__main__":
    main()
