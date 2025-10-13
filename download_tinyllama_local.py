"""
Helper to download TinyLlama model files to a local directory so the application can load them
without attempting to download at runtime. This script attempts to use `huggingface_hub` if available
or falls back to `git` + `git-lfs` instructions.

Usage:
    python download_tinyllama_local.py --output ml_models/tinyllama

Note: model files are large (~2GB). Ensure you have sufficient disk space and, if using
Hugging Face, that you are logged in (`huggingface-cli login`) if the model requires auth.
"""
import argparse
import os
import subprocess
import sys

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def run(cmd):
    print(f"> {cmd}")
    subprocess.check_call(cmd, shell=True)


def download_with_hf(output_dir: str):
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except Exception as e:
        print("huggingface_hub not available:", e)
        return False

    try:
        print(f"Downloading model snapshot for {MODEL_ID} to {output_dir} using huggingface_hub...")
        snapshot_download(repo_id=MODEL_ID, cache_dir=output_dir, local_dir=output_dir, repo_type="model")
        print("✅ Download complete")
        return True
    except Exception as e:
        print("❌ huggingface_hub download failed:", e)
        return False


def download_with_git(output_dir: str):
    print("Attempting to download with git (requires git-lfs if large files)...")
    print("If the model repo is large, ensure git-lfs is installed and configured.")
    try:
        run(f"git clone https://huggingface.co/{MODEL_ID} \"{output_dir}\"")
        print("✅ Download complete via git clone")
        return True
    except Exception as e:
        print("❌ git clone failed:", e)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="ml_models/tinyllama", help="Output directory for model files")
    args = parser.parse_args()

    out = os.path.abspath(args.output)
    os.makedirs(out, exist_ok=True)

    # Try huggingface_hub first
    if download_with_hf(out):
        print("Model downloaded into:", out)
        print("Set LOCAL_TINYLLAMA_PATH=", out)
        return

    # Fallback to git clone
    if download_with_git(out):
        print("Model downloaded into:", out)
        print("Set LOCAL_TINYLLAMA_PATH=", out)
        return

    print("Failed to download the model automatically.\n\nManual steps:\n1) Install git-lfs and authenticate if necessary.\n2) Clone the repo: git clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 ml_models/tinyllama\n3) Or use the huggingface_hub snapshot_download function after logging in with huggingface-cli login.")


if __name__ == '__main__':
    main()
