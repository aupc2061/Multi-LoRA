from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sae import SparseAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sparse autoencoder on collected UNet activations.")
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="sae_data/checkpoints")
    parser.add_argument("--mode", type=str, default="both", choices=["base", "lora", "both", "diff"])
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--tokens_per_sample", type=int, default=4096)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1_coeff", type=float, default=1e-3)
    parser.add_argument("--latent_mult", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected activation shape [B, C, H, W], got {tuple(x.shape)}")
    return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])


def sample_tokens(x: torch.Tensor, n: int) -> torch.Tensor:
    if n <= 0 or x.shape[0] <= n:
        return x
    idx = torch.randperm(x.shape[0])[:n]
    return x[idx]


def build_tokens_from_file(path: Path, mode: str, tokens_per_sample: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    base = flatten_tokens(payload["base"].float())
    lora = flatten_tokens(payload["lora"].float())

    if mode == "base":
        out = base
    elif mode == "lora":
        out = lora
    elif mode == "diff":
        out = lora - base
    else:
        out = torch.cat([base, lora], dim=0)

    return sample_tokens(out, tokens_per_sample)


def load_dataset(feature_dir: Path, mode: str, max_files: int, tokens_per_sample: int) -> torch.Tensor:
    files = sorted(feature_dir.glob("sample_*.pt"))
    if not files:
        raise ValueError(f"No feature files found in {feature_dir}")
    if max_files > 0:
        files = files[:max_files]

    all_tokens: List[torch.Tensor] = []
    for file_path in files:
        all_tokens.append(build_tokens_from_file(file_path, mode, tokens_per_sample))

    return torch.cat(all_tokens, dim=0)


def split_dataset(x: torch.Tensor, val_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    num = x.shape[0]
    num_val = int(num * val_ratio)
    if num > 1:
        num_val = max(1, min(num_val, num - 1))
    else:
        num_val = 0
    perm = torch.randperm(num)
    val_idx = perm[:num_val]
    train_idx = perm[num_val:]
    return x[train_idx], x[val_idx]


def evaluate(model: SparseAutoencoder, loader: DataLoader, l1_coeff: float, device: str) -> dict:
    model.eval()
    mse_meter = 0.0
    l1_meter = 0.0
    total = 0
    act_sum = None

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            recon, z = model(x)
            mse = nn.functional.mse_loss(recon, x)
            l1 = z.abs().mean()

            bsz = x.shape[0]
            total += bsz
            mse_meter += mse.item() * bsz
            l1_meter += l1.item() * bsz

            fire = (z > 0).float().mean(dim=0)
            if act_sum is None:
                act_sum = fire * bsz
            else:
                act_sum += fire * bsz

    if total == 0 or act_sum is None:
        return {
            "mse": 0.0,
            "l1": 0.0,
            "loss": 0.0,
            "dead_fraction": 0.0,
        }

    fire_rate = act_sum / total
    dead_fraction = (fire_rate < 1e-4).float().mean().item()

    return {
        "mse": mse_meter / max(total, 1),
        "l1": l1_meter / max(total, 1),
        "loss": (mse_meter / max(total, 1)) + l1_coeff * (l1_meter / max(total, 1)),
        "dead_fraction": dead_fraction,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = load_dataset(feature_dir, args.mode, args.max_files, args.tokens_per_sample)

    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    x = (x - mean) / std

    train_x, val_x = split_dataset(x, args.val_ratio)

    train_loader = DataLoader(TensorDataset(train_x), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(val_x), batch_size=args.batch_size, shuffle=False, drop_last=False)

    input_dim = train_x.shape[1]
    latent_dim = int(round(input_dim * args.latent_mult))
    model = SparseAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_val = float("inf")
    best_path = output_dir / "sae_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_mse = 0.0
        train_l1 = 0.0
        total = 0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(args.device)
            recon, z = model(batch_x)

            mse = nn.functional.mse_loss(recon, batch_x)
            l1 = z.abs().mean()
            loss = mse + args.l1_coeff * l1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bsz = batch_x.shape[0]
            total += bsz
            train_mse += mse.item() * bsz
            train_l1 += l1.item() * bsz

        train_stats = {
            "mse": train_mse / max(total, 1),
            "l1": train_l1 / max(total, 1),
            "loss": (train_mse / max(total, 1)) + args.l1_coeff * (train_l1 / max(total, 1)),
        }
        val_stats = evaluate(model, val_loader, args.l1_coeff, args.device)

        row = {
            "epoch": epoch,
            "train": train_stats,
            "val": val_stats,
        }
        history.append(row)
        print(json.dumps(row))

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": input_dim,
                    "latent_dim": latent_dim,
                    "mean": mean,
                    "std": std,
                    "args": vars(args),
                    "history": history,
                },
                best_path,
            )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved best SAE checkpoint to {best_path}")


if __name__ == "__main__":
    main()
