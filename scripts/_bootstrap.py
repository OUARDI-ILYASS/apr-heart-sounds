"""Shared entry-point boilerplate for every phase script.

Handles four things identically everywhere so the phase scripts stay thin:
argument parsing, config loading + validation, seeding, and logger creation.
Doing this in one place means an ablation run and a baseline run go through
exactly the same code path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

# Make `src` importable regardless of the working directory the user ran from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import load_config, ConfigDict          # noqa: E402
from src.config.schema import validate_config, derived_shapes  # noqa: E402
from src.utils.logging import get_logger, banner               # noqa: E402
from src.utils.seed import set_global_seed                     # noqa: E402


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to the config file (or an ablation override).")
    parser.add_argument("--base-config", default=None,
                        help="Base config to merge an ablation override onto.")
    parser.add_argument("--override", nargs="*", default=None, metavar="KEY=VALUE",
                        help="Ad-hoc overrides, e.g. --override cnn.training.epochs=5")
    parser.add_argument("--tag", default=None,
                        help="Suffix appended to output directories, for parallel runs.")
    parser.add_argument("--force", action="store_true",
                        help="Recompute artifacts even if they already exist.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")
    return parser


def setup(phase: str, description: str,
          parser: Optional[argparse.ArgumentParser] = None
          ) -> Tuple[ConfigDict, object, argparse.Namespace]:
    """Parse args, load and validate config, seed everything, build a logger."""
    parser = parser or build_parser(description)
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config if not Path(args.config).is_absolute() \
        else Path(args.config)
    base_path = None
    if args.base_config:
        base_path = PROJECT_ROOT / args.base_config \
            if not Path(args.base_config).is_absolute() else Path(args.base_config)

    cfg = load_config(config_path, base_path=base_path,
                      overrides=args.override, project_root=PROJECT_ROOT)

    # Validate BEFORE doing any work. A bad n_mels should cost one second, not
    # twenty minutes of feature extraction followed by a shape error.
    validate_config(cfg.to_dict())

    if args.tag:
        cfg["experiment"]["name"] = f"{cfg['experiment']['name']}_{args.tag}"

    reports_dir = Path(cfg["_abs_paths"]["reports_dir"]) / f"phase_{phase}"
    logger = get_logger(phase, log_file=reports_dir / "run.log")

    banner(logger, f"PHASE {phase}: {description}")
    logger.info(f"Experiment : {cfg['experiment']['name']}")
    logger.info(f"Config     : {config_path}  (hash {cfg['_config_hash']})")

    seed_info = set_global_seed(int(cfg["experiment"]["seed"]))
    logger.info(f"Seed       : {seed_info['seed']} "
                f"(torch={seed_info['torch_seeded']}, cuda={seed_info['cuda_seeded']})")

    shapes = derived_shapes(cfg.to_dict())
    logger.info(f"Shapes     : {shapes}")

    return cfg, logger, args
