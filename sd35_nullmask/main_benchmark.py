from __future__ import annotations

import argparse
import json

from .benchmark import run_benchmark
from .config import SD35NullMaskBenchmarkConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the SD3.5 null-mask benchmark plan.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SD35NullMaskBenchmarkConfig.from_config(args.config)
    if args.dry_run:
        config.dry_run = True
    result = run_benchmark(config, config_path=args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
