"""Helper runner for single-seed HMCL or Dynamic-Curvature experiments."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HMCL continual learning on a stream")
    parser.add_argument("backbone", nargs="?", default="meru_vit_b", choices=["meru_vit_l", "meru_vit_b", "hycoclip_vit_b"])
    parser.add_argument("seed", nargs="?", type=int, default=42)
    parser.add_argument("--features-root", type=Path, default=Path("feature_storage/meru_vit_b"))
    parser.add_argument("--method", default="dynamic-curvature", choices=["hmcl", "dynamic-curvature", "ewc"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    main_py = repo_root / "main.py"
    out = args.output or (repo_root / "results" / f"{args.method}_{args.backbone}_seed{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(main_py),
        "--feature-storage-dir",
        str(args.features_root),
        "--backbone",
        args.backbone,
        "--seed",
        str(args.seed),
        "--method",
        args.method,
        "--output",
        str(out),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    subprocess.run(cmd, check=True, cwd=str(repo_root), env=env)


if __name__ == "__main__":
    main()
