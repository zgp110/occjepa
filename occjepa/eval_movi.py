"""MOVi-D / MOVi-B (EoRaS test lists) and Kubric-Static / Kubric-Dynamic (TACO test sets) export; score with
occjepa.score_movi.
  python -m occjepa.eval_movi --data movid|movib|kubric_static|kubric_dynamic --mode cropped|uncropped \
      --ckpt weights/occjepa_vitG.pt --out results/movid_uncropped.npz [--limit N]
  cropped    one pass. Boxes = the standard per-frame boxes (square, side 2 x max(h, w) of the visible bbox, clamp
             [32, 256], shifted inside the frame, frames without a visible part interpolated). Protocol region = the
             EoRaS rectangle per frame; input pixels outside it are zeroed and the pasted prediction is clipped to it
             and unioned with the visible mask.
  uncropped  whole-frame input: pass 1 = standard boxes; pass 2 = re-crop (boxes recomputed with the same rule on the
             pass-1 prediction). pred2 = the model's output.
Output npz: inst (index ids), pred1 [N,T,H,W/8] packed bits, pred2 (uncropped), boxes.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import paths
from .data.movi import MoviRAM
from .model import OccJEPA

DATA = dict(paths.MOVI, **{k: os.path.dirname(v) for k, v in paths.KUBRIC.items()})
MIN_SIDE = 32


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def eoras_ids(ds):
    idx = ds.index
    vid_of = {v: k for k, v in enumerate(idx["videos"])}
    ii_of = {(int(v), int(o)): k for k, (v, o) in enumerate(idx["instances"])}
    return sorted(ii_of[(vid_of[int(s.split("-")[0])], int(s.split("-")[1]))] for s in idx["eoras_instance_list"])


def eoras_rect(m, H, W):
    """EoRaS per-frame region for visible mask m [H,W] -> bool [H,W] (empty if m is empty)."""
    out = np.zeros((H, W), bool)
    r, c = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    if len(r) == 0:
        return out
    x0, x1, y0, y1 = int(r[0]), int(r[-1]) + 1, int(c[0]), int(c[-1]) + 1   # EoRaS: x = rows, max exclusive
    xc, yc = (x0 + x1) // 2, (y0 + y1) // 2
    xl, yl = int((x1 - x0) * 2.0), int((y1 - y0) * 2.0)
    a0, a1 = max(0, xc - xl // 2), min(H, xc + xl // 2)
    b0, b1 = max(0, yc - yl // 2), min(W, yc + yl // 2)
    out[a0:a1 + 1, b0:b1 + 1] = True                                          # EoRaS slices [min:max+1]
    return out


def box_rule(U, H, W):
    """MoviRAM._stats box arithmetic (line by line) applied to masks U [T,H,W] -> int32 [T,3] (y0, x0, side)."""
    T = len(U)
    cy, cx, sd = np.full(T, np.nan), np.full(T, np.nan), np.full(T, np.nan)
    for t in range(T):
        if not U[t].any():
            continue
        ys, xs = np.nonzero(U[t])
        cy[t], cx[t] = (ys.min() + ys.max() + 1) / 2, (xs.min() + xs.max() + 1) / 2
        sd[t] = max(ys.max() + 1 - ys.min(), xs.max() + 1 - xs.min()) * 2.0
    ok = ~np.isnan(cy)
    box = np.zeros((T, 3), np.int32)
    if not ok.any():
        box[:] = (0, 0, H)
        return box
    tt = np.arange(T)
    cy, cx, sd = (np.interp(tt, tt[ok], v[ok]) for v in (cy, cx, sd))
    for t in range(T):
        side = int(min(max(round(sd[t]), MIN_SIDE), H, W))
        box[t] = (int(np.clip(round(cy[t] - side / 2), 0, H - side)), int(np.clip(round(cx[t] - side / 2), 0, W - side)), side)
    return box


class Boxes:
    """Temporarily replace ds.box rows (MoviRAM.batch / paste read self.box[i])."""

    def __init__(self, ds, ids, boxes):
        self.ds, self.ids, self.boxes = ds, ids, boxes

    def __enter__(self):
        self.old = self.ds.box[self.ids].copy()
        self.ds.box[self.ids] = self.boxes

    def __exit__(self, *a):
        self.ds.box[self.ids] = self.old


@torch.no_grad()
def predict(model, ds, ids, rects=None):
    """Logits [B,T,o,o] for instances ids with the CURRENT ds.box rows; rects (list of bool [T,H,W]) zero the input
    outside the protocol region (RGB, visible mask, other objects' masks)."""
    S = model.cfg["S"]
    b = ds.batch(ids, S=S)
    if rects is not None:
        for k, i in enumerate(ids):
            r = torch.from_numpy(rects[k]).cuda().float()[:, None]
            gS, gO = ds._grid(i, S, "cuda"), ds._grid(i, b["modal_t"].shape[-1], "cuda")
            rs = (F.grid_sample(r, gS, mode="bilinear", align_corners=False)[:, 0] > 0.5).float()   # [T,S,S]
            ro = (F.grid_sample(r, gO, mode="bilinear", align_corners=False)[:, 0] > 0.5).float()
            b["clip"][k] = b["clip"][k] * rs[None]
            b["modal"][k] = b["modal"][k] * rs
            b["occl"][k] = b["occl"][k] * rs
            b["modal_t"][k] = b["modal_t"][k] * ro
    return model.predict(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=sorted(DATA), required=True)
    ap.add_argument("--arm", default="B2_lora_ft")
    ap.add_argument("--world_model", default="vitG")
    ap.add_argument("--mode", choices=("cropped", "uncropped"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force_same", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    ds = MoviRAM("test", root=DATA[a.data])
    ids = eoras_ids(ds)
    if a.limit:
        ids = ids[:a.limit]
    log(f"{a.data} test: {len(ds.inst)} instances, EoRaS list {len(ids)}; load {time.time() - t0:.0f}s")
    model = OccJEPA(a.arm, a.world_model).load(a.ckpt)
    assert model.cfg["S"] == 256 and ds.T == 24
    H, W = ds.H, ds.W
    n = len(ids)
    pred1 = np.zeros((n, ds.T, H, W // 8), np.uint8)
    pred2 = np.zeros_like(pred1) if a.mode == "uncropped" else None
    box1 = ds.box[ids].copy()
    box2 = np.zeros_like(box1)
    st = dict(rect_vis_outside=0, frames=0, frames_no_rect=0, boxes_changed=0)
    t1 = time.time()
    for k0 in range(0, n, a.bs):
        bid = ids[k0:k0 + a.bs]
        if a.mode == "cropped":
            rects = []
            for i in bid:
                m = ds.modal(i)
                r = np.stack([eoras_rect(m[t], H, W) for t in range(ds.T)])
                st["rect_vis_outside"] += int((m & ~r).sum())
                st["frames_no_rect"] += int((~r.any((1, 2))).sum())
                st["frames"] += ds.T
                rects.append(r)
            lg = predict(model, ds, bid, rects)
            for j, i in enumerate(bid):
                p = (ds.paste(i, lg[j]) & rects[j]) | ds.modal(i)
                pred1[k0 + j] = np.packbits(p, axis=-1)
        else:
            lg = predict(model, ds, bid)
            U = []
            for j, i in enumerate(bid):
                p = ds.paste(i, lg[j])                      # already union visible
                pred1[k0 + j] = np.packbits(p, axis=-1)
                U.append(p)
            nb = np.stack([ds.box[i] if a.force_same else box_rule(U[j], H, W) for j, i in enumerate(bid)])
            st["boxes_changed"] += int((nb != ds.box[bid]).any(-1).sum())
            st["frames"] += nb.shape[0] * nb.shape[1]
            box2[k0:k0 + len(bid)] = nb
            with Boxes(ds, bid, nb):
                lg2 = predict(model, ds, bid)
                for j, i in enumerate(bid):
                    pred2[k0 + j] = np.packbits(ds.paste(i, lg2[j]), axis=-1)
        if (k0 // a.bs) % 50 == 0:
            log(f"{k0 + len(bid)}/{n}  {(time.time() - t1) / (k0 + len(bid)):.3f}s/inst")
    st["sec"] = time.time() - t1
    if a.force_same:
        same = float((pred1 == pred2).mean())
        st["force_same_byte_agreement"] = same
        log(f"UNIT force_same: packed-byte agreement pass2 vs pass1 = {same:.6f}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out = dict(inst=np.asarray(ids, np.int64), pred1=pred1, box1=box1, mode=np.array(a.mode), data=np.array(a.data),
               run=np.array("occjepa"), arm=np.array(a.arm), ckpt=np.array(a.ckpt), stats=np.array(json.dumps(st)))
    if pred2 is not None:
        out.update(pred2=pred2, box2=box2)
    np.savez_compressed(a.out, **out)
    log("stats", json.dumps(st))
    log("DONE", a.out)


if __name__ == "__main__":
    main()
