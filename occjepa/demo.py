"""Amodal masks for one object in a video.
  python -m occjepa.demo --frames FRAMES_DIR --visible MASKS_DIR --ckpt weights/occjepa_vitG.pt --out OUT_DIR [--video]
FRAMES_DIR: the frames as .jpg/.png (sorted by name). MASKS_DIR: the object's visible mask in every frame, same names
(any image format; > 127 = visible). Writes OUT_DIR/<frame>.png (amodal mask inside the image), OUT_DIR/boxes.json
(amodal boxes, xywh, which may extend beyond the image) and, with --video, OUT_DIR/overlay.mp4 (needs imageio-ffmpeg).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image
from pycocotools import mask as cocomask

from . import infer
from .model import OccJEPA

EXT = (".jpg", ".jpeg", ".png")


def make_sequence(frames_dir, masks_dir):
    fr = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(EXT))
    assert fr, f"no frames in {frames_dir}"
    W, H = Image.open(os.path.join(frames_dir, fr[0])).size
    rles = []
    for f in fr:
        cand = glob.glob(os.path.join(masks_dir, os.path.splitext(f)[0] + ".*"))
        assert cand, f"no visible mask for frame {f} in {masks_dir}"
        m = np.array(Image.open(cand[0]).convert("L").resize((W, H), Image.NEAREST)) > 127
        rles.append(cocomask.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode("ascii"))
    return dict(file_names=fr, rles=rles, height=H, width=W), fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--visible", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--world_model", default="vitG")
    ap.add_argument("--no_recrop", action="store_true")
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s, names = make_sequence(a.frames, a.visible)
    model = OccJEPA("B2_lora_ft", a.world_model).load(a.ckpt)
    res = infer.run(model, [s], os.path.join(a.frames, ""), recrop=not a.no_recrop, return_masks=not a.no_recrop)
    boxes = (res["pass1"] if a.no_recrop else res["rc"])[0]
    json.dump({n: [float(v) for v in b] for n, b in zip(names, boxes)}, open(os.path.join(a.out, "boxes.json"), "w"),
              indent=1)
    if a.no_recrop:
        print("boxes written (masks are produced with the re-crop pass)")
        return
    mk = res["masks"][0]
    H, W = s["height"], s["width"]
    frames = []
    for t, n in enumerate(names):
        c = mk["mask"][t]
        h, w = int(round(H * mk["sc"])), int(round(W * mk["sc"]))
        inner = c[mk["oy"]:mk["oy"] + h, mk["ox"]:mk["ox"] + w]
        m = np.array(Image.fromarray(inner.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
        Image.fromarray(m.astype(np.uint8) * 255).save(os.path.join(a.out, os.path.splitext(n)[0] + ".png"))
        if a.video:
            rgb = np.array(Image.open(os.path.join(a.frames, n)).convert("RGB")).astype(float)
            vis = cocomask.decode({"counts": s["rles"][t], "size": [H, W]}).astype(bool)
            rgb[m & vis] = 0.5 * rgb[m & vis] + 0.5 * 255
            rgb[m & ~vis] = 0.28 * rgb[m & ~vis] + 0.72 * np.array([194, 57, 155])
            frames.append(rgb.clip(0, 255).astype(np.uint8))
    if a.video:
        import imageio_ffmpeg
        import subprocess
        tmp = os.path.join(a.out, "_frames")
        os.makedirs(tmp, exist_ok=True)
        for k, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(tmp, f"{k:05d}.png"))
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", "8", "-i",
                        os.path.join(tmp, "%05d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", os.path.join(a.out, "overlay.mp4")], check=True)
    print(f"{len(names)} amodal masks written to {a.out}")


if __name__ == "__main__":
    main()
