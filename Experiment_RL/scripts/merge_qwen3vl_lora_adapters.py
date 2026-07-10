#!/usr/bin/env python3
"""Merge exported VERL LoRA adapters into standalone Qwen3-VL HF models."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        action="append",
        required=True,
        help="JSON object with name, model_dir, adapter_dir, output_dir.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--hf-home", default="/data2/hjk/cache/huggingface")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def merge_one(spec: dict[str, str], args: argparse.Namespace) -> None:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    name = spec["name"]
    model_dir = Path(spec["model_dir"])
    adapter_dir = Path(spec["adapter_dir"])
    output_dir = Path(spec["output_dir"])
    marker = output_dir / "MERGED_LORA_DONE.json"
    if marker.exists() and (output_dir / "config.json").exists():
        print(f"{name}: merged full model already exists at {output_dir}", flush=True)
        return

    print(f"{name}: loading processor from {model_dir}", flush=True)
    processor = AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"{name}: loading base model from {model_dir}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        dtype=dtype_from_name(args.dtype),
        device_map="auto",
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    print(f"{name}: loading adapter from {adapter_dir}", flush=True)
    model = PeftModel.from_pretrained(model, adapter_dir)
    print(f"{name}: merging adapter", flush=True)
    model = model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{name}: saving merged model to {output_dir}", flush=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    marker.write_text(
        json.dumps(
            {
                "name": name,
                "source_model_dir": str(model_dir),
                "source_adapter_dir": str(adapter_dir),
                "output_dir": str(output_dir),
                "dtype": args.dtype,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main() -> None:
    args = parse_args()
    os.environ["HF_HOME"] = args.hf_home
    os.environ["TRANSFORMERS_CACHE"] = args.hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.hf_home) / "hub")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    specs = [json.loads(item) for item in args.spec]
    for spec in specs:
        merge_one(spec, args)
    print("MERGE_LORA_ADAPTERS_DONE", flush=True)


if __name__ == "__main__":
    main()
