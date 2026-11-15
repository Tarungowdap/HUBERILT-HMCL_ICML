"""Download official pretrained MERU and HyCoCLIP checkpoints."""

import argparse
import urllib.request
from pathlib import Path
from loguru import logger

MODELS = {
    "meru_vit_l": "https://dl.fbaipublicfiles.com/meru/meru_vit_l.pth",
    "meru_vit_b": "https://dl.fbaipublicfiles.com/meru/meru_vit_b.pth",
    "hycoclip_vit_b": "https://huggingface.co/avik-pal/hycoclip/resolve/main/hycoclip_vit_b.pth",
}


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("File already exists at {}", dest)
        return
    logger.info("Downloading from {} to {}...", url, dest)
    urllib.request.urlretrieve(url, dest)
    logger.info("Downloaded successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download backbone checkpoints")
    parser.add_argument("dest_dir", nargs="?", type=Path, default=Path("checkpoints"))
    parser.add_argument("model", nargs="?", default="all", choices=["all", "meru_vit_l", "meru_vit_b", "hycoclip_vit_b"])
    args = parser.parse_args()

    targets = list(MODELS.keys()) if args.model == "all" else [args.model]
    for m in targets:
        dest = args.dest_dir / f"{m}.pth"
        download_file(MODELS[m], dest)


if __name__ == "__main__":
    main()
