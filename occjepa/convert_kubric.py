"""Convert TACO's Kubric-Static / Kubric-Dynamic benchmark zips (HF JasonAplp/TACO benchmarks/) into the compact MOVi
layout read by data.movi.MoviRAM, so that eval_movi / score_movi run on them unchanged:

  <out>/test/frames.npy  uint8 [V, 24, 256, 256, 3]   origin_rgb/NN.png (RGBA -> RGB), area-resized 512 -> 256
  <out>/test/segvis.npy  uint8 [V, 24, 256, 256]      visible_mask/NN.png > 127 after area-resize >= 0.5  -> 1 (object 0)
  <out>/test/amodal.npy  uint8 [V, 24, 256, 256/8]    amodal_mask/NN.png, same rule, np.packbits along W
  <out>/test/index.json  T, H, W, videos (sample ids from val.json), instances [[v, 0]], eoras_instance_list "v-0" (all),
                         meta: bbox_frames = frames with a visible pixel (256 px), n_inst 1
One target object per sample (the benchmark gives only its visible / amodal masks: no other objects' masks, so the
"other objects" mask of MoviRAM is empty).  Members are read as bytes and decoded with PIL only; nothing is executed.
python -m occjepa.convert_kubric --zip kubric_static.zip --root kubric_static --out $OCCJEPA_DATA/kubric_taco/static
"""
import argparse
import io
import json
import os
import time
import zipfile

import numpy as np
from PIL import Image

from .npyio import NpyWriter

T, S, SRC = 24, 256, 512


def png(z, name):
    return Image.open(io.BytesIO(z.read(name)))


def mask(z, name):
    m = np.asarray(png(z, name).convert("L"), np.float32) / 255.0
    assert m.shape == (SRC, SRC), m.shape
    return m.reshape(S, SRC // S, S, SRC // S).mean((1, 3)) >= 0.5        # area resize, majority rule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--root", required=True, help="top folder inside the zip (kubric_static / kubric_dynamic)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = zipfile.ZipFile(a.zip)
    names = set(z.namelist())
    assert not any(n.startswith("/") or ".." in n.split("/") for n in names), "unsafe member path"
    ids = json.loads(z.read(f"{a.root}/val.json"))
    assert isinstance(ids, list) and all(isinstance(s, str) and s.isdigit() for s in ids), "unexpected val.json"
    d = os.path.join(a.out, "test")
    os.makedirs(d, exist_ok=True)
    V = len(ids)
    wf = NpyWriter(os.path.join(d, "frames.npy"), (V, T, S, S, 3), np.uint8)
    ws = NpyWriter(os.path.join(d, "segvis.npy"), (V, T, S, S), np.uint8)
    wa = NpyWriter(os.path.join(d, "amodal.npy"), (V, T, S, S // 8), np.uint8)
    meta, t0 = {}, time.time()
    for v, sid in enumerate(ids):
        fr = np.zeros((T, S, S, 3), np.uint8)
        sv = np.zeros((T, S, S), np.uint8)
        am = np.zeros((T, S, S // 8), np.uint8)
        vis_frames = []
        for t in range(T):
            base = f"{a.root}/{sid}"
            rgb = np.asarray(png(z, f"{base}/origin_rgb/{t:02d}.png").convert("RGB").resize((S, S), Image.BOX))
            fr[t] = rgb
            m = mask(z, f"{base}/visible_mask/{t:02d}.png")
            g = mask(z, f"{base}/amodal_mask/{t:02d}.png")
            sv[t] = m.astype(np.uint8)
            am[t] = np.packbits(g, axis=-1)
            if m.any():
                vis_frames.append(t)
        wf.write_rows([v], fr[None])
        ws.write_rows([v], sv[None])
        wa.write_rows([v], am[None])
        meta[str(v)] = dict(sample=sid, num_frames=str(T), height=str(S), width=str(S), bbox_frames=[vis_frames], n_inst="1")
        if v % 50 == 0:
            print(f"{v}/{V} {time.time() - t0:.0f}s", flush=True)
    for w in (wf, ws, wa):
        w.close()
    idx = dict(split="test", T=T, H=S, W=S, videos=list(range(V)), instances=[[v, 0] for v in range(V)],
               eoras_instance_list=[f"{v}-0" for v in range(V)], meta=meta, source=os.path.abspath(a.zip), root=a.root,
               note="TACO Kubric benchmark converted by occjepa.convert_kubric (512 -> 256 area resize)")
    json.dump(idx, open(os.path.join(d, "index.json"), "w"))
    print(f"DONE {V} samples {time.time() - t0:.0f}s -> {d}", flush=True)


if __name__ == "__main__":
    main()
