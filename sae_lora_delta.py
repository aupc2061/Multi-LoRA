from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sae import SparseAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute SAE feature deltas between base and LoRA activations.")
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--tokens_per_sample", type=int, default=4096)
    parser.add_argument("--output", type=str, default="sae_data/delta_summary.json")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])


def sample_tokens(x: torch.Tensor, n: int) -> torch.Tensor:
    if n <= 0 or x.shape[0] <= n:
        return x
    idx = torch.randperm(x.shape[0])[:n]
    return x[idx]


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    model = SparseAutoencoder(ckpt["input_dim"], ckpt["latent_dim"])
    model.load_state_dict(ckpt["model_state"])
    model = model.to(args.device).eval()

    mean = ckpt["mean"].to(args.device)
    std = ckpt["std"].to(args.device)

    files = sorted(Path(args.feature_dir).glob("sample_*.pt"))
    if not files:
        raise ValueError(f"No feature files found in {args.feature_dir}")

    delta_sum = None
    count = 0

    with torch.no_grad():
        for path in files:
            payload = torch.load(path, map_location="cpu")
            if "lora" not in payload:
                raise ValueError(
                    f"File {path} has no 'lora' tensor. sae_lora_delta.py requires paired base-vs-lora samples."
                )
            base = flatten_tokens(payload["base"].float())
            lora = flatten_tokens(payload["lora"].float())

            base = sample_tokens(base, args.tokens_per_sample)
            lora = sample_tokens(lora, args.tokens_per_sample)

            n = min(base.shape[0], lora.shape[0])
            base = base[:n].to(args.device)
            lora = lora[:n].to(args.device)

            base = (base - mean) / std
            lora = (lora - mean) / std

            z_base = model.encode(base)
            z_lora = model.encode(lora)
            delta = (z_lora - z_base).abs().mean(dim=0)

            if delta_sum is None:
                delta_sum = delta
            else:
                delta_sum += delta
            count += 1

    if count == 0 or delta_sum is None:
        raise RuntimeError("No valid feature samples were processed.")

    delta_mean = delta_sum / count
    topk = min(args.topk, delta_mean.numel())
    vals, idx = torch.topk(delta_mean, k=topk)

    summary = {
        "num_files": count,
        "topk": topk,
        "top_features": [
            {"feature": int(i.item()), "delta": float(v.item())}
            for i, v in zip(idx.cpu(), vals.cpu())
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved delta summary to {out_path}")


if __name__ == "__main__":
    main()
