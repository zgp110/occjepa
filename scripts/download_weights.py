"""Download the weights OccJEPA needs into $OCCJEPA_WEIGHTS (default ./weights):
  * the released V-JEPA 2.1 ViT-G (gigantic, 384) checkpoint of Meta (encoder + predictor; 30.2 GB), and optionally ViT-g;
  * the OccJEPA weights (head, predictor fine-tuning and LoRA; 276 MB), from $OCCJEPA_CKPT_URL.
  python scripts/download_weights.py [--vitg]
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from occjepa import vjepa as V  # noqa: E402

BASE = "https://dl.fbaipublicfiles.com/vjepa2"


def fetch(url, path, size=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if size and os.path.exists(path) and os.path.getsize(path) == size:
        print("ok", path)
        return
    subprocess.run(["curl", "-L", "--fail", "-C", "-", "--retry", "5", "-o", path, url], check=True)
    if size:
        assert os.path.getsize(path) == size, f"{path}: {os.path.getsize(path)} bytes, expected {size}"
    print("downloaded", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vitg", action="store_true", help="also the ViT-g world model (ablations)")
    a = ap.parse_args()
    for name in ["vitG"] + (["vitg"] if a.vitg else []):
        f = V.MODEL_ZOO[name]["file"] + ".pt"
        fetch(f"{BASE}/{f}", V.ckpt_path(name), V.EXPECTED_BYTES[name])
    url = os.environ.get("OCCJEPA_CKPT_URL")
    if url:
        fetch(url, os.path.join(V.DEFAULT_CKPT_DIR, "occjepa_vitG.pt"))
    else:
        print("set OCCJEPA_CKPT_URL to the OccJEPA weights link of the project page to download occjepa_vitG.pt")


if __name__ == "__main__":
    main()
