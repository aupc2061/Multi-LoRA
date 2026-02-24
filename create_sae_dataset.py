from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from sae.feature_hooks import LastActivationRecorder, resolve_module
from utils import get_prompt, load_lora_info


def load_dataset_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a JSON object.")

    dataset_cfg = payload.get("dataset", payload)
    if not isinstance(dataset_cfg, dict):
        raise ValueError("Config must contain an object at key 'dataset' or be a flat object.")
    return dataset_cfg


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()

    config_defaults: dict[str, Any] = {}
    if bootstrap_args.config:
        config_defaults = load_dataset_config(bootstrap_args.config)

    parser = argparse.ArgumentParser(description="Create a balanced SAE dataset and optionally collect paired activations.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--collection_mode", type=str, default="paired", choices=["paired", "base_only"])
    parser.add_argument("--fast_p100", action="store_true")
    parser.add_argument("--image_style", type=str, default="reality", choices=["anime", "reality"])
    parser.add_argument("--lora_info_path", type=str, default="lora_info.json")
    parser.add_argument("--lora_path", type=str, default="models/lora")

    parser.add_argument("--total_samples", type=int, default=250)
    parser.add_argument("--samples_per_category", type=int, default=50)
    parser.add_argument("--seed_bank", type=str, default="111,222,333,444,555")
    parser.add_argument("--split_seed", type=int, default=0)

    parser.add_argument("--model_name", type=str, default="SG161222/Realistic_Vision_V5.1_noVAE")
    parser.add_argument("--custom_pipeline", type=str, default="./pipelines/sd1.5_0.26.3")
    parser.add_argument("--lora_scale", type=float, default=0.8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--denoise_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--hook_module", type=str, default="conv_norm_out")
    parser.add_argument("--keep_cfg_pair", action="store_true")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--manifest_only", action="store_true")
    parser.add_argument("--out_dir", type=str, default="sae_data/datasets/reality_250")

    if config_defaults:
        valid_keys = {action.dest for action in parser._actions}
        filtered_defaults = {k: v for k, v in config_defaults.items() if k in valid_keys}
        parser.set_defaults(**filtered_defaults)

    return parser.parse_args()


def pick_dtype(dtype_name: str) -> torch.dtype:
    return torch.float16 if dtype_name == "float16" else torch.float32


def parse_seed_bank(seed_bank: str) -> list[int]:
    seeds = [int(token.strip()) for token in seed_bank.split(",") if token.strip()]
    if not seeds:
        raise ValueError("seed_bank cannot be empty")
    return seeds


def apply_mode_defaults(args: argparse.Namespace) -> None:
    if args.collection_mode == "base_only" and args.fast_p100:
        args.height = 512
        args.width = 512
        args.denoise_steps = 20
        args.cfg_scale = 6.5
        if args.total_samples == 250:
            args.total_samples = 120
        if args.seed_bank == "111,222,333,444,555":
            args.seed_bank = "101,202,303,404,505,606,707,808"


def get_allocations(lora_info: dict[str, list[dict[str, Any]]], total_samples: int, samples_per_category: int) -> dict[str, dict[str, int]]:
    categories = ["character", "clothing", "style", "background", "object"]
    for category in categories:
        if category not in lora_info:
            raise ValueError(f"Missing category in lora info: {category}")

    if total_samples != 250 or samples_per_category != 50:
        raise ValueError("Current dataset recipe is fixed to 250 total and 50 per category.")

    alloc: dict[str, dict[str, int]] = {}

    character_ids = sorted([entry["id"] for entry in lora_info["character"]])
    if len(character_ids) != 3:
        raise ValueError("Expected exactly 3 character LoRAs for fixed allocation plan.")
    alloc["character"] = {
        character_ids[0]: 20,
        character_ids[1]: 15,
        character_ids[2]: 15,
    }

    for category in ["clothing", "style", "background", "object"]:
        ids = sorted([entry["id"] for entry in lora_info[category]])
        if len(ids) != 2:
            raise ValueError(f"Expected exactly 2 LoRAs for category {category}.")
        alloc[category] = {ids[0]: 25, ids[1]: 25}

    return alloc


def build_base_only_templates() -> list[dict[str, str]]:
    return [
        {"type": "portrait", "template": "{base}, portrait photo of {subject}, {lighting}, {composition}, {detail}"},
        {"type": "street", "template": "{base}, {subject} in {environment}, {lighting}, {composition}, {detail}"},
        {"type": "nature", "template": "{base}, {subject} in {environment}, natural atmosphere, {composition}, {detail}"},
        {"type": "product", "template": "{base}, product-style image of {subject}, clean background, {lighting}, {detail}"},
        {"type": "architecture", "template": "{base}, wide shot of {environment}, {lighting}, {composition}, {detail}"},
        {"type": "action", "template": "{base}, {subject} {action}, {environment}, {lighting}, {detail}"},
        {"type": "closeup", "template": "{base}, close-up of {subject}, shallow depth of field, {lighting}, {detail}"},
        {"type": "documentary", "template": "{base}, documentary style scene: {subject}, {environment}, {composition}, {detail}"},
        {"type": "interior", "template": "{base}, indoor scene in {environment}, {subject}, {lighting}, {detail}"},
        {"type": "outdoor", "template": "{base}, outdoor scene in {environment}, {subject}, {lighting}, {detail}"},
    ]


def build_base_only_prompt_pool() -> dict[str, list[str]]:
    return {
        "subject": [
            "a person", "two friends", "a cyclist", "a chef", "a musician", "a cat", "a dog", "a car",
            "a bicycle", "a wooden chair", "a flower bouquet", "a coffee cup", "a city bus", "a mountain cabin",
            "a laptop on a desk", "a market stall", "a bridge", "a storefront", "a plate of food", "a landscape",
        ],
        "environment": [
            "a city street", "a cozy cafe", "a modern office", "a park", "a beach", "a mountain trail",
            "a train station", "a library", "a kitchen", "a studio", "an old town alley", "a garden",
            "a rainy sidewalk", "a snowy field", "a sunset viewpoint", "a busy market", "a rooftop",
            "a museum hall", "a suburban neighborhood", "a forest path",
        ],
        "lighting": [
            "soft natural light", "golden hour light", "overcast daylight", "dramatic side lighting",
            "warm indoor light", "cool ambient light", "neon night lighting", "backlit composition",
            "high contrast light", "diffused studio light",
        ],
        "composition": [
            "rule-of-thirds composition", "centered composition", "wide-angle framing", "telephoto framing",
            "eye-level shot", "low-angle shot", "high-angle shot", "symmetrical framing", "dynamic framing", "minimal framing",
        ],
        "detail": [
            "high detail", "realistic textures", "sharp focus", "fine-grained details", "natural color tones",
            "balanced contrast", "clean background", "depth and clarity", "cinematic mood", "photorealistic quality",
        ],
        "action": [
            "walking", "running", "reading", "cooking", "playing guitar", "talking", "looking at the camera",
            "taking a photo", "crossing the street", "sitting and relaxing",
        ],
    }


def build_base_only_rows(args: argparse.Namespace, base_prompt: str, negative_prompt: str, seed_bank: list[int]) -> list[dict[str, Any]]:
    templates = build_base_only_templates()
    pool = build_base_only_prompt_pool()
    prompt_occurrence: dict[str, int] = defaultdict(int)

    rows: list[dict[str, Any]] = []
    for idx in range(args.total_samples):
        template_spec = templates[idx % len(templates)]

        prompt = template_spec["template"].format(
            base=base_prompt,
            subject=pool["subject"][idx % len(pool["subject"])],
            environment=pool["environment"][idx % len(pool["environment"])],
            lighting=pool["lighting"][idx % len(pool["lighting"])],
            composition=pool["composition"][idx % len(pool["composition"])],
            detail=pool["detail"][idx % len(pool["detail"])],
            action=pool["action"][idx % len(pool["action"])],
        )

        # If the same prompt appears multiple times due to template/pool cycling,
        # rotate seeds per prompt occurrence to avoid duplicate (prompt, seed) rows.
        occurrence = prompt_occurrence[prompt]
        seed = seed_bank[occurrence % len(seed_bank)]
        prompt_occurrence[prompt] += 1

        rows.append(
            {
                "sample_id": None,
                "lora_id": None,
                "category": "base",
                "prompt": prompt,
                "prompt_type": template_spec["type"],
                "seed": seed,
                "negative_prompt": negative_prompt,
                "cfg_scale": args.cfg_scale,
                "lora_scale": args.lora_scale,
                "height": args.height,
                "width": args.width,
                "denoise_steps": args.denoise_steps,
                "hook_module": args.hook_module,
                "collection_mode": "base_only",
                "split": None,
            }
        )

    return rows


def build_category_templates() -> dict[str, list[dict[str, str]]]:
    category_phrase = {
        "character": "portrait focus",
        "clothing": "fashion detail emphasis",
        "style": "overall visual style emphasis",
        "background": "environment and scene emphasis",
        "object": "object-centric framing",
    }

    templates: dict[str, list[dict[str, str]]] = {}
    for category, phrase in category_phrase.items():
        templates[category] = [
            {"type": "faithful", "template": "{base}, {triggers}"},
            {"type": "faithful", "template": "{base}, {triggers}, {phrase}"},
            {"type": "faithful", "template": "{base}, {triggers}, highly detailed"},
            {"type": "faithful", "template": "{base}, {triggers}, cinematic lighting"},
            {"type": "faithful", "template": "{base}, {triggers}, natural skin texture"},
            {"type": "faithful", "template": "{base}, {triggers}, realistic photography"},
            {"type": "paraphrase", "template": "{base}, featuring {name}, emphasize {lead_trigger}, {phrase}"},
            {"type": "paraphrase", "template": "{base}, include {name} with focus on {lead_trigger}"},
            {"type": "paraphrase", "template": "{base}, depict {name}; preserve the look suggested by {lead_trigger}"},
            {"type": "weak", "template": "{base}, {name}, {phrase}"},
        ]

    return templates


def render_prompt(base_prompt: str, lora_entry: dict[str, Any], template_spec: dict[str, str], category: str) -> str:
    triggers = lora_entry["trigger"]
    return template_spec["template"].format(
        base=base_prompt,
        triggers=", ".join(triggers),
        lead_trigger=triggers[0],
        name=lora_entry["name"],
        phrase={
            "character": "portrait focus",
            "clothing": "fashion detail emphasis",
            "style": "overall visual style emphasis",
            "background": "environment and scene emphasis",
            "object": "object-centric framing",
        }[category],
    )


def assign_split(rows: list[dict[str, Any]], split_seed: int) -> None:
    rng = random.Random(split_seed)
    by_category: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_category[row["category"]].append(idx)

    for category, indices in by_category.items():
        rng.shuffle(indices)
        if len(indices) != 50:
            raise ValueError(f"Expected 50 rows for category {category}, got {len(indices)}")

        for i, idx in enumerate(indices):
            if i < 40:
                rows[idx]["split"] = "train"
            elif i < 45:
                rows[idx]["split"] = "val"
            else:
                rows[idx]["split"] = "test"


def assign_split_ratio(rows: list[dict[str, Any]], split_seed: int, train_ratio: float = 0.8, val_ratio: float = 0.1) -> None:
    rng = random.Random(split_seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(1, n_train), max(1, n - 2)) if n >= 3 else max(1, n)
    n_val = min(max(1, n_val), max(1, n - n_train - 1)) if n - n_train >= 2 else max(0, n - n_train)

    for rank, idx in enumerate(indices):
        if rank < n_train:
            rows[idx]["split"] = "train"
        elif rank < n_train + n_val:
            rows[idx]["split"] = "val"
        else:
            rows[idx]["split"] = "test"


def build_manifest(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    apply_mode_defaults(args)
    base_prompt, negative_prompt = get_prompt(args.image_style)
    seed_bank = parse_seed_bank(args.seed_bank)

    if args.collection_mode == "base_only":
        rows = build_base_only_rows(args, base_prompt, negative_prompt, seed_bank)
    else:
        lora_info = load_lora_info(args.image_style, args.lora_info_path)
        alloc = get_allocations(lora_info, args.total_samples, args.samples_per_category)
        templates = build_category_templates()

        lora_lookup: dict[str, dict[str, Any]] = {}
        for category, entries in lora_info.items():
            for entry in entries:
                lora_lookup[entry["id"]] = entry

        rows: list[dict[str, Any]] = []
        for category in ["character", "clothing", "style", "background", "object"]:
            cat_templates = templates[category]
            for lora_id, count in alloc[category].items():
                entry = lora_lookup[lora_id]
                for i in range(count):
                    template_spec = cat_templates[i % len(cat_templates)]
                    seed = seed_bank[(i // len(cat_templates)) % len(seed_bank)]
                    prompt = render_prompt(base_prompt, entry, template_spec, category)
                    rows.append(
                        {
                            "sample_id": None,
                            "lora_id": lora_id,
                            "category": category,
                            "prompt": prompt,
                            "prompt_type": template_spec["type"],
                            "seed": seed,
                            "negative_prompt": negative_prompt,
                            "cfg_scale": args.cfg_scale,
                            "lora_scale": args.lora_scale,
                            "height": args.height,
                            "width": args.width,
                            "denoise_steps": args.denoise_steps,
                            "hook_module": args.hook_module,
                            "collection_mode": "paired",
                            "split": None,
                        }
                    )

    if len(rows) != args.total_samples:
        raise ValueError(f"Expected {args.total_samples} rows, got {len(rows)}")

    duplicate_key_count = Counter((row["lora_id"], row["prompt"], row["seed"]) for row in rows)
    duplicate_keys = [key for key, cnt in duplicate_key_count.items() if cnt > 1]
    if duplicate_keys:
        raise ValueError(f"Found duplicate rows by (lora_id, prompt, seed): {duplicate_keys[:3]}")

    if args.collection_mode == "base_only":
        assign_split_ratio(rows, args.split_seed, train_ratio=0.8, val_ratio=0.1)
    else:
        assign_split(rows, args.split_seed)

    for idx, row in enumerate(rows):
        row["sample_id"] = f"sample_{idx:06d}"

    stats = {
        "num_rows": len(rows),
        "collection_mode": args.collection_mode,
        "by_category": dict(Counter(row["category"] for row in rows)),
        "by_lora_id": dict(Counter(row["lora_id"] for row in rows)),
        "by_prompt_type": dict(Counter(row["prompt_type"] for row in rows)),
        "by_split": dict(Counter(row["split"] for row in rows)),
        "by_split_and_category": {
            split: dict(Counter(row["category"] for row in rows if row["split"] == split))
            for split in ["train", "val", "test"]
        },
        "seed_bank": seed_bank,
        "image_style": args.image_style,
        "hook_module": args.hook_module,
        "height": args.height,
        "width": args.width,
        "denoise_steps": args.denoise_steps,
    }

    return rows, stats


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
    cfg_scale: float,
    lora_scale: float,
    height: int,
    width: int,
    denoise_steps: int,
    with_lora: bool,
    lora_id: str,
    keep_cfg_pair: bool,
    device: str,
) -> torch.Tensor:
    recorder.reset()
    if with_lora:
        pipeline.enable_lora()
        pipeline.set_adapters([lora_id])
    else:
        pipeline.disable_lora()

    generator = torch.Generator(device=device).manual_seed(seed)
    _ = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=denoise_steps,
        guidance_scale=cfg_scale,
        generator=generator,
        output_type="latent",
        return_dict=False,
        cross_attention_kwargs={"scale": lora_scale},
    )

    if recorder.last is None:
        raise RuntimeError("No activation was recorded. Check hook_module path.")

    activation = recorder.last.detach().float().cpu()
    activation = select_activation(activation, cfg_scale, keep_cfg_pair)
    return activation


def collect_samples(args: argparse.Namespace, rows: list[dict[str, Any]], out_dir: Path) -> None:
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

    if args.collection_mode == "paired":
        lora_info = load_lora_info(args.image_style, args.lora_info_path)
        lora_root = Path(args.lora_path) / args.image_style
        for category_entries in lora_info.values():
            for lora in category_entries:
                pipeline.load_lora_weights(
                    str(lora_root),
                    weight_name=f"{lora['id']}.safetensors",
                    adapter_name=lora["id"],
                )

    hook_target = resolve_module(pipeline.unet, args.hook_module)
    recorder = LastActivationRecorder(hook_target)

    for i, row in enumerate(rows):
        base_act = run_and_capture(
            pipeline=pipeline,
            recorder=recorder,
            prompt=row["prompt"],
            negative_prompt=row["negative_prompt"],
            seed=row["seed"],
            cfg_scale=row["cfg_scale"],
            lora_scale=row["lora_scale"],
            height=row["height"],
            width=row["width"],
            denoise_steps=row["denoise_steps"],
            with_lora=False,
            lora_id=row["lora_id"],
            keep_cfg_pair=args.keep_cfg_pair,
            device=args.device,
        )
        file_name = f"{row['sample_id']}.pt"
        payload: dict[str, Any] = {
            "base": base_act,
            "meta": row,
        }
        if args.collection_mode == "paired":
            lora_act = run_and_capture(
                pipeline=pipeline,
                recorder=recorder,
                prompt=row["prompt"],
                negative_prompt=row["negative_prompt"],
                seed=row["seed"],
                cfg_scale=row["cfg_scale"],
                lora_scale=row["lora_scale"],
                height=row["height"],
                width=row["width"],
                denoise_steps=row["denoise_steps"],
                with_lora=True,
                lora_id=row["lora_id"],
                keep_cfg_pair=args.keep_cfg_pair,
                device=args.device,
            )
            payload["lora"] = lora_act

        torch.save(payload, out_dir / file_name)
        row["file"] = file_name

        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            print(f"Collected {i + 1}/{len(rows)} samples ({args.collection_mode})")

    recorder.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, stats = build_manifest(args)
    (out_dir / "dataset_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))

    if args.manifest_only:
        print(f"Manifest-only mode: wrote manifest to {out_dir}")
        return

    collect_samples(args, rows, out_dir)
    (out_dir / "dataset_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Finished collecting dataset in {out_dir}")


if __name__ == "__main__":
    main()
