"""SAIL-VOS val (Diffusion-VAS protocol): predict every sequence of the grid cache (two 24-frame windows [0,24) and [1,25),
the later one overwriting), paste back into the 800x1280 frame (> 0) UNION the visible mask, export the Diffusion-VAS
prediction format and score inline:
  mIoU           mean over all frames of IoU(P, A)
  mIoU_occ       Diffusion-VAS' occluded IoU (5x5-eroded occluded parts), averaged over the frames where it is >= 1e-6
  mIoU_occ_fixed the same per-frame value on the fixed set of frames whose eroded GT occluded part is non-empty, a
                 miss counting as 0 (the comparison on common frames of the paper)
With --recrop (default) a second pass re-crops every object around its pass-1 prediction (data/sailvos_rc.py).
  python -m occjepa.eval_sailvos --ckpt weights/occjepa_vitG.pt --out results/sailvos [--split svval_grid] [--limit N]
The exported json can also be scored with Diffusion-VAS' official eval/metric_diffusion_vas_sailvos.py.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from . import paths
from .data import sailvos as SD
from .model import OccJEPA


def windows(L, T):
    st = list(range(0, max(1, L - T + 1), T))
    if L > T and st[-1] + T < L:
        st.append(L - T)
    return st


@torch.no_grad()
def predict(model, ds, ids, S):
    """-> [L,H,W] bool per id (union visible), later windows overwrite earlier frames."""
    out = [np.zeros((ds.L, ds.H, ds.W), bool) for _ in ids]
    for w in windows(ds.L, ds.T):
        lg = model.predict(ds.batch(ids, S=S, t0=w))
        for j, i in enumerate(ids):
            out[j][w:w + ds.T] = ds.paste(i, lg[j], t0=w)
    return out


def run_pass(model, ds, ids, path, bs, log):
    ex = SD.PredExporter()
    full, occ = [], []
    t0 = time.time()
    for k in range(0, len(ids), bs):
        bid = ids[k:k + bs]
        preds = predict(model, ds, bid, model.cfg["S"])
        for j, i in enumerate(bid):
            a, m = ds.amo(i, 0, ds.L), ds.modal(i, 0, ds.L)
            p = preds[j] | m
            ex.add(ds, i, p, 0)
            for t in range(ds.L):
                f, o, has = SD.dvas_frame(p[t], a[t], m[t])
                full.append(f)
                if has:
                    occ.append(o)
        if (k // bs) % 50 == 0:
            log(f"{k + len(bid)}/{len(ids)} sequences, {(time.time() - t0) / (k + len(bid)):.2f} s/seq")
    ex.write(path)
    sc = SD.score(full, occ)
    sc["mIoU_occ_fixed"] = sc.pop("mIoU_occ_strict")
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="B2_lora_ft")
    ap.add_argument("--world_model", default="vitG")
    ap.add_argument("--split", default="svval_grid")
    ap.add_argument("--root", default=paths.SAILVOS_CACHE)
    ap.add_argument("--no_recrop", action="store_true")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = lambda *x: print(*x, flush=True)  # noqa: E731
    SD.CONFIG.update(split_map={"train": a.split, "test": a.split}, crop="inframe", rand_start=False)
    model = OccJEPA(a.arm, a.world_model).load(a.ckpt)
    ds = SD.SailvosRAM(a.split, root=a.root, crop="inframe")
    ids = [int(i) for i in np.nonzero(ds.meta["ok"])[0]]
    if a.limit:
        ids = ids[:a.limit]
    res = dict(pass1=run_pass(model, ds, ids, os.path.join(a.out, "pass1.json"), a.bs, log))
    log("pass 1:", res["pass1"])
    if not a.no_recrop:
        from .data.sailvos_rc import SvRC
        ds.close()
        ds2 = SvRC(a.split, os.path.join(a.out, "pass1.json"), root=a.root, crop="inframe")
        res["rc"] = run_pass(model, ds2, ids, os.path.join(a.out, "recrop.json"), a.bs, log)
        log("re-crop:", res["rc"])
    json.dump(dict(res, n_seq=len(ids), split=a.split, ckpt=a.ckpt), open(os.path.join(a.out, "results.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
