from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from sae import SparseAutoencoder
from sae.feature_hooks import resolve_module
from sae_intervene import SAEInterventionController, parse_target_steps, pick_dtype, run_once
from utils import get_prompt, load_lora_info


def load_ablation_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a JSON object.")

    ablation_cfg = payload.get("ablation", payload)
    if not isinstance(ablation_cfg, dict):
        raise ValueError("Config must contain an object at key 'ablation' or be a flat object.")
    return ablation_cfg


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()

    config_defaults: dict[str, Any] = {}
    if bootstrap_args.config:
        config_defaults = load_ablation_config(bootstrap_args.config)

    parser = argparse.ArgumentParser(description="Run batched SAE ablations and compare top-vs-random features.")
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--delta_json", type=str, required=True)
    parser.add_argument("--lora_ids", type=str, required=True, help="Comma-separated lora ids.")

    parser.add_argument("--k_list", type=str, default="16")
    parser.add_argument("--ablate_scales", type=str, default="0.0,0.5")
    parser.add_argument("--step_modes", type=str, default="last")
    parser.add_argument("--seeds", type=str, default="111,222,333")
    parser.add_argument("--random_draws", type=int, default=3)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)

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
    parser.add_argument("--hook_module", type=str, default="conv_norm_out")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--rng_seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="sae_data/ablation")

    if config_defaults:
        valid_keys = {action.dest for action in parser._actions}
        filtered_defaults = {k: v for k, v in config_defaults.items() if k in valid_keys}
        parser.set_defaults(**filtered_defaults)

    return parser.parse_args()


def parse_csv_ints(spec: str) -> list[int]:
    values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected non-empty integer list")
    return values


def parse_csv_floats(spec: str) -> list[float]:
    values = [float(token.strip()) for token in spec.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected non-empty float list")
    return values


def parse_csv_str(spec: str) -> list[str]:
    values = [token.strip() for token in spec.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected non-empty string list")
    return values


def infer_prompt_for_lora(image_style: str, lora_info_path: str, lora_id: str) -> str:
    lora_info = load_lora_info(image_style, lora_info_path)
    init_prompt, _ = get_prompt(image_style)
    for group in lora_info.values():
        for lora in group:
            if lora["id"] == lora_id:
                return init_prompt + ", " + ", ".join(lora["trigger"])
    raise ValueError(f"LoRA id not found in metadata: {lora_id}")


def load_top_features(delta_json: Path, max_k: int) -> list[int]:
    payload = json.loads(delta_json.read_text(encoding="utf-8"))
    top = payload.get("top_features", [])
    if len(top) < max_k:
        raise ValueError(f"delta_json has only {len(top)} features, but max k requested is {max_k}")
    return [int(entry["feature"]) for entry in top[:max_k]]


def sample_random_features(latent_dim: int, k: int, rng: random.Random) -> list[int]:
    if k > latent_dim:
        raise ValueError(f"k={k} cannot exceed latent_dim={latent_dim}")
    return sorted(rng.sample(range(latent_dim), k))


def bootstrap_mean_diff(a: list[float], b: list[float], n_boot: int, rng_seed: int) -> dict[str, float]:
    if not a or not b:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    rng = random.Random(rng_seed)
    diffs: list[float] = []
    for _ in range(n_boot):
        a_s = [a[rng.randrange(len(a))] for _ in range(len(a))]
        b_s = [b[rng.randrange(len(b))] for _ in range(len(b))]
        diffs.append(sum(a_s) / len(a_s) - sum(b_s) / len(b_s))

    diffs.sort()
    lo_idx = int(0.025 * (len(diffs) - 1))
    hi_idx = int(0.975 * (len(diffs) - 1))
    return {
        "mean_diff": sum(diffs) / len(diffs),
        "ci_low": diffs[lo_idx],
        "ci_high": diffs[hi_idx],
    }


def make_run_args(base: argparse.Namespace, lora_id: str, seed: int, scale: float) -> argparse.Namespace:
    return argparse.Namespace(
        lora_id=lora_id,
        seed=seed,
        device=base.device,
        height=base.height,
        width=base.width,
        denoise_steps=base.denoise_steps,
        cfg_scale=base.cfg_scale,
        lora_scale=base.lora_scale,
        ablate_scale=scale,
    )


def aggregate_results(rows: list[dict[str, Any]], bootstrap_samples: int, rng_seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["lora_id"], row["feature_set_type"], row["k"], row["ablate_scale"], row["step_mode"])
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, items in grouped.items():
        vals = [float(item["latents_mse"]) for item in items]
        pre_vals = [float(item["feature_abs_mean_pre"]) for item in items]
        post_vals = [float(item["feature_abs_mean_post"]) for item in items]
        summary_rows.append(
            {
                "lora_id": key[0],
                "feature_set_type": key[1],
                "k": key[2],
                "ablate_scale": key[3],
                "step_mode": key[4],
                "num_runs": len(items),
                "latents_mse_mean": sum(vals) / len(vals),
                "latents_mse_std": float(torch.tensor(vals).std(unbiased=False).item()) if len(vals) > 1 else 0.0,
                "feature_abs_mean_pre_mean": sum(pre_vals) / len(pre_vals),
                "feature_abs_mean_post_mean": sum(post_vals) / len(post_vals),
            }
        )

    effect_rows: list[dict[str, Any]] = []
    for lora_id in sorted({row["lora_id"] for row in rows}):
        for k in sorted({int(row["k"]) for row in rows}):
            for scale in sorted({float(row["ablate_scale"]) for row in rows}):
                for step_mode in sorted({row["step_mode"] for row in rows}):
                    top_vals = [
                        float(row["latents_mse"])
                        for row in rows
                        if row["lora_id"] == lora_id
                        and row["k"] == k
                        and row["ablate_scale"] == scale
                        and row["step_mode"] == step_mode
                        and row["feature_set_type"] == "top"
                    ]
                    rnd_vals = [
                        float(row["latents_mse"])
                        for row in rows
                        if row["lora_id"] == lora_id
                        and row["k"] == k
                        and row["ablate_scale"] == scale
                        and row["step_mode"] == step_mode
                        and row["feature_set_type"] == "random"
                    ]
                    if not top_vals or not rnd_vals:
                        continue

                    ci = bootstrap_mean_diff(top_vals, rnd_vals, bootstrap_samples, rng_seed)
                    effect_rows.append(
                        {
                            "lora_id": lora_id,
                            "k": k,
                            "ablate_scale": scale,
                            "step_mode": step_mode,
                            "top_mean_latents_mse": sum(top_vals) / len(top_vals),
                            "random_mean_latents_mse": sum(rnd_vals) / len(rnd_vals),
                            "mean_diff_top_minus_random": (sum(top_vals) / len(top_vals)) - (sum(rnd_vals) / len(rnd_vals)),
                            "bootstrap_mean_diff": ci["mean_diff"],
                            "bootstrap_ci_low": ci["ci_low"],
                            "bootstrap_ci_high": ci["ci_high"],
                            "num_top_runs": len(top_vals),
                            "num_random_runs": len(rnd_vals),
                        }
                    )

    return summary_rows, effect_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lora_ids = parse_csv_str(args.lora_ids)
    k_list = parse_csv_ints(args.k_list)
    scales = parse_csv_floats(args.ablate_scales)
    step_modes = parse_csv_str(args.step_modes)
    seeds = parse_csv_ints(args.seeds)

    max_k = max(k_list)
    top_features = load_top_features(Path(args.delta_json), max_k)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    input_dim = int(ckpt["input_dim"])
    latent_dim = int(ckpt["latent_dim"])

    sae = SparseAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
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
    lora_info = load_lora_info(args.image_style, args.lora_info_path)
    all_lora_ids = [lora["id"] for group in lora_info.values() for lora in group]
    for lora_id in all_lora_ids:
        pipeline.load_lora_weights(str(lora_root), weight_name=f"{lora_id}.safetensors", adapter_name=lora_id)

    hook_target = resolve_module(pipeline.unet, args.hook_module)
    _, negative_prompt = get_prompt(args.image_style)

    rng = random.Random(args.rng_seed)
    run_rows: list[dict[str, Any]] = []

    for lora_id in lora_ids:
        prompt = infer_prompt_for_lora(args.image_style, args.lora_info_path, lora_id)

        for step_mode in step_modes:
            target_steps = parse_target_steps(step_mode, args.denoise_steps)

            for k in k_list:
                top_set = top_features[:k]
                random_sets = [sample_random_features(latent_dim, k, rng) for _ in range(args.random_draws)]

                for scale in scales:
                    for seed in seeds:
                        # Top set run
                        controller_top = SAEInterventionController(
                            module=hook_target,
                            sae=sae,
                            mean=mean,
                            std=std,
                            feature_indices=top_set,
                            apply_to=args.apply_to,
                            cfg_scale=args.cfg_scale,
                        )
                        run_args = make_run_args(args, lora_id=lora_id, seed=seed, scale=scale)
                        try:
                            base_result, _ = run_once(
                                pipeline=pipeline,
                                controller=controller_top,
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                args=run_args,
                                intervene=False,
                                target_steps=target_steps,
                                output_type="latent",
                            )
                            int_result, int_records = run_once(
                                pipeline=pipeline,
                                controller=controller_top,
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                args=run_args,
                                intervene=True,
                                target_steps=target_steps,
                                output_type="latent",
                            )
                        finally:
                            controller_top.close()

                        base_latents = base_result[0]
                        int_latents = int_result[0]
                        latents_mse = torch.mean((base_latents - int_latents) ** 2).item()
                        pre = [float(r["feature_abs_mean_pre"]) for r in int_records]
                        post = [float(r["feature_abs_mean_post"]) for r in int_records]

                        run_rows.append(
                            {
                                "lora_id": lora_id,
                                "feature_set_type": "top",
                                "random_draw_idx": -1,
                                "k": k,
                                "ablate_scale": scale,
                                "step_mode": step_mode,
                                "seed": seed,
                                "latents_mse": latents_mse,
                                "feature_abs_mean_pre": float(sum(pre) / max(len(pre), 1)),
                                "feature_abs_mean_post": float(sum(post) / max(len(post), 1)),
                                "feature_abs_mean_delta": float((sum(post) - sum(pre)) / max(len(pre), 1)),
                                "selected_features": top_set,
                            }
                        )

                        # Random set runs
                        for draw_idx, random_set in enumerate(random_sets):
                            controller_rand = SAEInterventionController(
                                module=hook_target,
                                sae=sae,
                                mean=mean,
                                std=std,
                                feature_indices=random_set,
                                apply_to=args.apply_to,
                                cfg_scale=args.cfg_scale,
                            )
                            try:
                                base_result_r, _ = run_once(
                                    pipeline=pipeline,
                                    controller=controller_rand,
                                    prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    args=run_args,
                                    intervene=False,
                                    target_steps=target_steps,
                                    output_type="latent",
                                )
                                int_result_r, int_records_r = run_once(
                                    pipeline=pipeline,
                                    controller=controller_rand,
                                    prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    args=run_args,
                                    intervene=True,
                                    target_steps=target_steps,
                                    output_type="latent",
                                )
                            finally:
                                controller_rand.close()

                            base_latents_r = base_result_r[0]
                            int_latents_r = int_result_r[0]
                            latents_mse_r = torch.mean((base_latents_r - int_latents_r) ** 2).item()
                            pre_r = [float(r["feature_abs_mean_pre"]) for r in int_records_r]
                            post_r = [float(r["feature_abs_mean_post"]) for r in int_records_r]

                            run_rows.append(
                                {
                                    "lora_id": lora_id,
                                    "feature_set_type": "random",
                                    "random_draw_idx": draw_idx,
                                    "k": k,
                                    "ablate_scale": scale,
                                    "step_mode": step_mode,
                                    "seed": seed,
                                    "latents_mse": latents_mse_r,
                                    "feature_abs_mean_pre": float(sum(pre_r) / max(len(pre_r), 1)),
                                    "feature_abs_mean_post": float(sum(post_r) / max(len(post_r), 1)),
                                    "feature_abs_mean_delta": float((sum(post_r) - sum(pre_r)) / max(len(pre_r), 1)),
                                    "selected_features": random_set,
                                }
                            )

                        if len(run_rows) % 10 == 0:
                            print(f"Completed {len(run_rows)} runs")

    summary_rows, effect_rows = aggregate_results(run_rows, args.bootstrap_samples, args.rng_seed)

    write_jsonl(out_dir / "results.jsonl", run_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "effects.csv", effect_rows)

    report = {
        "num_runs": len(run_rows),
        "checkpoint": args.checkpoint,
        "delta_json": args.delta_json,
        "lora_ids": lora_ids,
        "k_list": k_list,
        "ablate_scales": scales,
        "step_modes": step_modes,
        "seeds": seeds,
        "random_draws": args.random_draws,
        "effects": effect_rows,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved ablation outputs to {out_dir}")


if __name__ == "__main__":
    main()
