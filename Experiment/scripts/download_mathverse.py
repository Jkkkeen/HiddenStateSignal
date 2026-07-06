#!/usr/bin/env python3
"""Download MathVerse files to the project data directory."""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MathVerse from Hugging Face.")
    parser.add_argument("--repo-id", default="AI4Math/MathVerse")
    parser.add_argument("--split", default="testmini", help="Dataset split/file stem.")
    parser.add_argument(
        "--output-dir",
        default="data/mathverse",
        help="Destination directory under the current project.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default="https://hf-mirror.com",
        help="Hugging Face endpoint or mirror.",
    )
    parser.add_argument(
        "--hf-home",
        default="/data2/hjk/models/huggingface",
        help="Hugging Face cache directory on /data2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.hf_home) / "hub"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_dir = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=output_dir,
        allow_patterns=[
            f"{args.split}.json",
            f"{args.split}.parquet",
            "images.zip",
            "images/**",
        ],
    )
    local_path = Path(local_dir)
    print(f"Downloaded MathVerse files to: {local_path}")

    zip_path = local_path / "images.zip"
    images_dir = local_path / "images"
    if zip_path.exists() and not images_dir.exists():
        print(f"Extracting {zip_path} -> {images_dir}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(local_path)

    print("Available files:")
    for path in sorted(local_path.rglob("*")):
        if path.is_file():
            print(path.relative_to(local_path))


if __name__ == "__main__":
    main()
