# Downloads a gated Hugging Face model (default: meta-llama/Llama-3.2-3B)
# straight into your PROJECT folder (no symlinks to C: caches).
#
# Usage (from this folder):
#   python fetch_llama_model.py
#   # or specify a different model or output dir:
#   python fetch_llama_model.py --model-id meta-llama/Llama-3.2-3B --out-dir models\llama-3.2-3b
#
# Prereqs:
#   - You already ran: hf login   (or hf auth login --token hf_xxx)
#   - You have accepted the model’s license on its model card

from __future__ import annotations
import argparse
import os
from pathlib import Path
from datetime import datetime
import shutil
import sys

from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.errors import HfHubHTTPError

def human_bytes(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"

def parse_args():
    p = argparse.ArgumentParser(description="Fetch a Hugging Face model into this project folder.")
    p.add_argument("--model-id", default="meta-llama/Llama-3.2-3B",
                   help="Hugging Face model repo id")
    p.add_argument("--revision", default="main", help="Branch/commit tag to download")
    p.add_argument("--out-dir", default=None,
                   help="Relative or absolute path where files will be stored. "
                        "Defaults to ./models/<sanitized-model-id>")
    p.add_argument("--include", default=None,
                   help="Comma-separated patterns to include (e.g., 'config.json,*.safetensors'). "
                        "By default downloads the full repo.")
    return p.parse_args()

def sanitize_model_id(mid: str) -> str:
    # e.g. "meta-llama/Llama-3.2-3B" -> "meta-llama__Llama-3.2-3B"
    return mid.replace("/", "__")

def ensure_writable(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    test = path / ".write_test"
    with open(test, "w", encoding="utf-8") as f:
        f.write("ok")
    test.unlink()

def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else project_root / "models" / sanitize_model_id(args.model_id)
    out_dir = out_dir.resolve()

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Target model directory: {out_dir}")
    ensure_writable(out_dir)

    # Optional include filter
    allow_patterns = None
    if args.include:
        allow_patterns = [pat.strip() for pat in args.include.split(",") if pat.strip()]
        print(f"[INFO] Limiting download to: {allow_patterns}")

    # Small sanity check: confirm access (and license acceptance) before downloading
    api = HfApi()
    try:
        info = api.model_info(args.model_id, revision=args.revision)
        print(f"[INFO] Found model: {info.modelId} (sha: {info.sha})")
    except HfHubHTTPError as e:
        print("[ERROR] Could not access the model metadata. "
              "Have you accepted the model license and are you logged in?")
        print(f"        {e}")
        sys.exit(1)

    try:
        # snapshot_download will normally populate the global cache; by setting
        # local_dir and local_dir_use_symlinks=False we copy files *into* out_dir
        # so your project is self-contained (no links to C:).
        local_path = snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,   # ensure real files, not symlinks
            allow_patterns=allow_patterns,  # None = everything
            ignore_patterns=None,
            tqdm_class=None,                # default progress
        )
    except HfHubHTTPError as e:
        print("[ERROR] Download failed (auth or license issue?).")
        print(f"        {e}")
        sys.exit(1)

    # Summarize what we fetched
    total_bytes = 0
    file_count = 0
    for p in out_dir.rglob("*"):
        if p.is_file():
            file_count += 1
            try:
                total_bytes += p.stat().st_size
            except OSError:
                pass

    print(f"[OK] Download complete into: {local_path}")
    print(f"[OK] Files: {file_count:,} | Size: {human_bytes(total_bytes)}")

    # Write a small manifest for reproducibility
    manifest = out_dir / "_MODEL_FETCH_INFO.txt"
    with open(manifest, "w", encoding="utf-8") as f:
        f.write(
            "Model fetch manifest\n"
            f"Timestamp: {datetime.utcnow().isoformat()}Z\n"
            f"Repo ID:   {args.model_id}\n"
            f"Revision:  {args.revision}\n"
            f"Location:  {out_dir}\n"
            f"Included:  {', '.join(allow_patterns) if allow_patterns else 'ALL'}\n"
        )
    print(f"[INFO] Wrote manifest: {manifest}")

if __name__ == "__main__":
    main()
