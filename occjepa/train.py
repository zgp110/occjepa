"""Train OccJEPA on SAIL-VOS + MOVi-D (the paper's recipe):
  python -m occjepa.train --tag occjepa_vitG [--world_model vitG] [--arm B2_lora_ft] [--steps 6000]
Data: the SAIL-VOS crop caches cache_<--sv_train> / cache_<--sv_test> under $OCCJEPA_CACHE/sailvos (scripts/
prepare_sailvos.sh) and MOVi-D train under $OCCJEPA_DATA/movi/movid (scripts/prepare_movi.sh). Each step takes a
SAIL-VOS batch or, with probability 0.5, a MOVi-D batch (data/mix.py). Whole SAIL-VOS videos with index % 30 == 0 are
held out for the in-run validation. Writes $OCCJEPA_RUNS/<tag>/: args.json, val_<step>.json, ckpt_<step>.pt, ckpt.pt,
mix_stream.npz. Batch 8, AdamW, 300 warm-up steps then cosine decay; learning rates: head 1e-3, predictor 1e-4 (1e-3
when randomly initialised), LoRA 5e-4; weight decay 0.05 on head and predictor matrices; gradient clipping at 1.0.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from . import paths
from .data import mix as MX
from .data import sailvos as SD
from .model import ARMS, DEFAULTS, OccJEPA


def summarize(res):
    """Frame-pooled means (Diffusion-VAS style)."""
    fl = np.concatenate([np.asarray(x, float) for x in res["full"] if len(x)]) if res["full"] else np.zeros(0)
    oc = np.concatenate([np.asarray(x, float) for x in res["occ"] if len(x)]) if res["occ"] else np.zeros(0)
    return dict(mIoU=float(fl.mean() * 100), mIoU_occ=float(oc.mean() * 100),
                mIoU_occ_dvas=float(oc[oc >= 1e-6].mean() * 100), n_frames=int(len(fl)), n_occ=int(len(oc)))


@torch.no_grad()
def evaluate(model, ds, ids, S, bs=8):
    model.eval()
    res = dict(full=[], occ=[], occ_vis=[])
    for k in range(0, len(ids), bs):
        bid = ids[k:k + bs]
        lg = model.predict(ds.batch(bid, S=S))
        for j, i in enumerate(bid):
            full, occ, occ_vis = SD.eval_masks(ds, i, ds.paste(i, lg[j]))
            res["full"].append(full)
            res["occ"].append(occ)
            res["occ_vis"].append(occ_vis)
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arm", default="B2_lora_ft", choices=list(ARMS))
    ap.add_argument("--world_model", default="vitG", choices=["vitG", "vitg"])
    ap.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lora_first", type=int, default=None, help="default: the last 12 encoder blocks")
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--n_val", type=int, default=256)
    ap.add_argument("--sv_train", default="svtrain_grid")
    ap.add_argument("--sv_test", default="svval_grid")
    ap.add_argument("--sv_root", default=paths.SAILVOS_CACHE)
    ap.add_argument("--stop_after", type=int, default=0, help="debug: stop after N steps (the schedule keeps --steps)")
    a = ap.parse_args()
    depth = {"vitG": 48, "vitg": 40}[a.world_model]
    lora_first = a.lora_first if a.lora_first is not None else depth - 12
    out = os.path.join(paths.RUNS, a.tag)
    os.makedirs(out, exist_ok=True)
    a.lora_first = lora_first
    json.dump(dict(vars(a), **{k: v for k, v in DEFAULTS.items() if k not in vars(a)}),
              open(os.path.join(out, "args.json"), "w"), indent=1)
    torch.backends.cuda.matmul.allow_tf32 = True
    SD.CONFIG.update(split_map={"train": a.sv_train, "test": a.sv_test}, crop="inframe", rand_start=True,
                     allow_incomplete=False)
    tr = MX.SailvosMixHR("train", root=a.sv_root)
    vis_ok = tr.vis_area.sum(1) >= 64
    is_val = (tr.inst[:, 0] % 30) == 0
    train_ids = np.nonzero(vis_ok & ~is_val)[0]
    val_all = np.nonzero(vis_ok & is_val & (tr.occ_area.max(1) >= 16))[0]
    val_ids = val_all[np.random.default_rng(0).permutation(len(val_all))[:a.n_val]]
    print(f"train instances {len(train_ids)} val {len(val_ids)}", flush=True)
    model = OccJEPA(a.arm, a.world_model, steps=a.steps, lora_first=lora_first)
    model.make_optimizer()
    model.train()
    rng = np.random.default_rng(1)
    t0, run = time.time(), 0.0
    for step in range(a.steps):
        ids = rng.choice(train_ids, a.bs, replace=False)
        fl = rng.random(a.bs) < 0.5
        run += model.train_step(tr.batch(ids, S=model.cfg["S"], flip=fl, trunc=None), step)
        if (step + 1) % 50 == 0:
            el = time.time() - t0
            print(f"step {step + 1} {el / (step + 1):.2f}s/it mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G "
                  f"loss={run / 50:.3f}", flush=True)
            run = 0.0
        if a.stop_after and step + 1 >= a.stop_after:
            print(f"stopped after {step + 1} steps (--stop_after)", flush=True)
            return
        if (step + 1) % a.val_every == 0 or step + 1 == a.steps:
            vs = summarize(evaluate(model, tr, list(val_ids), model.cfg["S"]))
            print(f"VAL step {step + 1} mIoU {vs['mIoU']:.1f} mIoU_occ {vs['mIoU_occ']:.1f}", flush=True)
            json.dump(vs, open(os.path.join(out, f"val_{step + 1}.json"), "w"), indent=1)
            model.save(os.path.join(out, f"ckpt_{step + 1}.pt"))
    model.save(os.path.join(out, "ckpt.pt"))
    info = tr.save_stream(os.path.join(out, "mix_stream.npz"))
    print(f"DONE {out} mix stream {info}", flush=True)


if __name__ == "__main__":
    main()
