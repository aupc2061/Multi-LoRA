from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from sae import LastActivationRecorder, resolve_module
from utils import get_prompt, load_lora_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect SD1.5 last-step UNet activations for base vs single-LoRA.")
    parser.add_argument("--model_name", type=str, default="SG161222/Realistic_Vision_V5.1_noVAE")
    parser.add_argument("--custom_pipeline", type=str, default="./pipelines/sd1.5_0.26.3")
    parser.add_argument("--image_style", type=str, default="reality", choices=["anime", "reality"])
    parser.add_argument("--lora_info_path", type=str, default="lora_info.json")
    parser.add_argument("--lora_path", type=str, default="models/lora")
    parser.add_argument("--lora_id", type=str, required=True)
    parser.add_argument("--lora_scale", type=float, default=0.8)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--denoise_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--seed_start", type=int, default=111)
    parser.add_argument("--num_seeds", type=int, default=8)
    parser.add_argument("--prompts_file", type=str, default=None)
    parser.add_argument("--hook_module", type=str, default="conv_norm_out")
    parser.add_argument("--keep_cfg_pair", action="store_true")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, default="sae_data/features")
    return parser.parse_args()


def get_lora_prompt(image_style: str, lora_info_path: str, lora_id: str) -> str:
    lora_info = load_lora_info(image_style, lora_info_path)
    init_prompt, _ = get_prompt(image_style)
    for group in lora_info.values():
        for lora in group:
            if lora["id"] == lora_id:
                triggers = lora["trigger"]
                return init_prompt + ", " + ", ".join(triggers)
    raise ValueError(f"LoRA id not found in metadata: {lora_id}")


def load_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompts_file:
        lines = [line.strip() for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()]
        prompts = [line for line in lines if line]
        if not prompts:
            raise ValueError("prompts_file is empty.")
        return prompts
    return [get_lora_prompt(args.image_style, args.lora_info_path, args.lora_id)]


def pick_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


def select_activation(act: torch.Tensor, cfg_scale: float, keep_cfg_pair: bool) -> torch.Tensor:
    if (not keep_cfg_pair) and cfg_scale > 1.0 and act.shape[0] == 2:
        return act[1:2]
    return act


def run_and_capture(
    pipeline: Any,
    recorder: LastActivationRecorder,
    prompt: str,
    negative_prompt: str,
    seed: int,
    args: argparse.Namespace,
    with_lora: bool,
) -> torch.Tensor:
    recorder.reset()
    if with_lora:
        pipeline.enable_lora()
        pipeline.set_adapters([args.lora_id])
    else:
        pipeline.disable_lora()

    generator = torch.Generator(device=args.device).manual_seed(seed)
    _ = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.denoise_steps,
        guidance_scale=args.cfg_scale,
        generator=generator,
        output_type="latent",
        return_dict=False,
        cross_attention_kwargs={"scale": args.lora_scale},
    )

    if recorder.last is None:
        raise RuntimeError("No activation was recorded. Check hook_module path.")

    activation = recorder.last.detach().float().cpu()
    activation = select_activation(activation, args.cfg_scale, args.keep_cfg_pair)
    return activation


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = pick_dtype(args.dtype)
    pipeline = DiffusionPipeline.from_pretrained(
        args.model_name,
        custom_pipeline=args.custom_pipeline,
        use_safetensors=True,
        torch_dtype=torch_dtype,
    ).to(args.device)

    schedule_config = dict(pipeline.scheduler.config)
    schedule_config["algorithm_type"] = "dpmsolver++"
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(schedule_config)

    lora_root = Path(args.lora_path) / args.image_style
    pipeline.load_lora_weights(str(lora_root), weight_name=f"{args.lora_id}.safetensors", adapter_name=args.lora_id)

    hook_target = resolve_module(pipeline.unet, args.hook_module)
    recorder = LastActivationRecorder(hook_target)

    prompts = load_prompts(args)
    _, negative_prompt = get_prompt(args.image_style)

    records = []
    sample_idx = 0
    for prompt_index, prompt in enumerate(prompts):
        for seed in range(args.seed_start, args.seed_start + args.num_seeds):
            base_act = run_and_capture(pipeline, recorder, prompt, negative_prompt, seed, args, with_lora=False)
            lora_act = run_and_capture(pipeline, recorder, prompt, negative_prompt, seed, args, with_lora=True)

            file_name = f"sample_{sample_idx:06d}.pt"
            save_path = out_dir / file_name
            torch.save(
                {
                    "base": base_act,
                    "lora": lora_act,
                    "meta": {
                        "lora_id": args.lora_id,
                        "prompt": prompt,
                        "prompt_index": prompt_index,
                        "seed": seed,
                        "hook_module": args.hook_module,
                        "denoise_steps": args.denoise_steps,
                        "cfg_scale": args.cfg_scale,
                        "height": args.height,
                        "width": args.width,
                    },
                },
                save_path,
            )
            records.append({"file": file_name, "prompt_index": prompt_index, "seed": seed})
            sample_idx += 1

    recorder.close()

    manifest = {
        "num_samples": len(records),
        "out_dir": str(out_dir),
        "lora_id": args.lora_id,
        "records": records,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} paired activation samples to {out_dir}")


if __name__ == "__main__":
    main()
