import argparse
import json
import os
from os.path import join

import torch
from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline

# from pipelines."sdxl_0.26.3".pipeline import StableDiffusionXLPipeline
from utils import get_prompt, load_lora_info


def _build_mask(importance, keep_ratio):
    num_steps = len(importance)
    keep_steps = max(1, int(round(num_steps * keep_ratio)))
    scores = torch.tensor(importance)
    topk = torch.topk(scores, keep_steps).indices.tolist()
    mask = [False] * num_steps
    for idx in topk:
        mask[idx] = True
    return mask


def _save_mask(save_dir, lora_id, payload):
    os.makedirs(save_dir, exist_ok=True)
    file_path = join(save_dir, f"mask_{lora_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main(args):
    # load LoRA metadata
    lora_info = load_lora_info(args.image_style, args.lora_info_path)

    # load SDXL pipeline
    if args.custom_pipeline:
        pipeline = DiffusionPipeline.from_pretrained(
            args.model_name,
            custom_pipeline=args.custom_pipeline,
            use_safetensors=True,
        ).to("cuda")
    else:
        pipeline = DiffusionPipeline.from_pretrained(
            args.model_name,
            custom_pipeline="MingZhong/StableDiffusionPipeline-with-LoRA-C"
            use_safetensors=True,
        ).to("cuda")

    # set scheduler
    schedule_config = dict(pipeline.scheduler.config)
    schedule_config["algorithm_type"] = "dpmsolver++"
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(schedule_config)

    # initialize LoRAs
    lora_path = join(args.lora_path, args.image_style)
    for element in list(lora_info.keys()):
        for lora in lora_info[element]:
            pipeline.load_lora_weights(
                lora_path,
                weight_name=lora["id"] + ".safetensors",
                adapter_name=lora["id"],
            )

    init_prompt, negative_prompt = get_prompt(args.image_style)

    all_loras = [lora for element in lora_info.values() for lora in element]
    for lora in all_loras:
        lora_id = lora["id"]
        triggers = lora["trigger"]
        prompt = init_prompt + ", " + ", ".join(triggers)

        pipeline.set_adapters([lora_id])

        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        latents_full, importance = pipeline(
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
            return_lora_step_importance=True,
        )

        mask = _build_mask(importance, args.keep_ratio)

        mse_mask = None
        if args.evaluate_mask:
            generator = torch.Generator(device="cuda").manual_seed(args.seed)
            latents_masked = pipeline(
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
                lora_timestep_mask=mask,
            )[0]
            mse_mask = torch.mean((latents_full - latents_masked) ** 2).item()

        payload = {
            "lora_id": lora_id,
            "mask": mask,
            "importance": importance,
            "keep_ratio": args.keep_ratio,
            "num_inference_steps": args.denoise_steps,
            "mse_mask": mse_mask,
        }
        _save_mask(args.mask_save_dir, lora_id, payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize a per-timestep LoRA mask for SDXL using noise-pred MSE."
    )
    parser.add_argument("--model_name", default="stabilityai/stable-diffusion-xl-base-1.0", type=str)
    parser.add_argument("--custom_pipeline", default=None, type=str)
    parser.add_argument("--mask_save_dir", default="timestep_masks", type=str)
    parser.add_argument("--keep_ratio", default=0.5, type=float)
    parser.add_argument("--evaluate_mask", action="store_true")

    parser.add_argument("--lora_path", default="models/lora", type=str)
    parser.add_argument("--lora_info_path", default="lora_info.json", type=str)
    parser.add_argument("--lora_scale", default=0.8, type=float)
    parser.add_argument("--image_style", default="reality", choices=["anime", "reality"], type=str)

    parser.add_argument("--height", default=1024, type=int)
    parser.add_argument("--width", default=1024, type=int)
    parser.add_argument("--denoise_steps", default=50, type=int)
    parser.add_argument("--cfg_scale", default=7.0, type=float)
    parser.add_argument("--seed", default=111, type=int)

    args = parser.parse_args()
    main(args)
