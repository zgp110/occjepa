"""TAO-Amodal evaluation (Diffusion-VAS protocol): AP25 / AP50 / AP75 of the predicted amodal boxes, one pass and with
re-crop.
  val  (Diffusion-VAS release):  python -m occjepa.eval_tao --annot data/tao_amodal/diffusion_vas_tao_amodal_val.json \
           --track_ids data/tao_amodal/tao_amodal_track_ids_abs2rel_val.json --frames data/tao_amodal/frames/ \
           --ckpt weights/occjepa_vitG.pt --out results/tao_val
  test (our held-out split, occjepa.build_tao_split):  --annot data/tao_amodal/test_seqs.json
           --track_ids data/tao_amodal/test_abs2rel.json ...
Writes OUT/preds.npz (boxes [N, 25, 4] of both passes, per-frame IoUs) and OUT/results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import infer, tao as TA
from .model import OccJEPA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", required=True)
    ap.add_argument("--track_ids", required=True)
    ap.add_argument("--frames", required=True, help="root that the sequences' file_names are relative to")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="B2_lora_ft", choices=["B2_lora_ft", "B2_lora_scr", "C_lora"])
    ap.add_argument("--world_model", default="vitG")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seqs = TA.load_annot(a.annot, a.track_ids)
    if a.limit:
        seqs = seqs[:a.limit]
    t0 = time.time()
    model = OccJEPA(a.arm, a.world_model).load(a.ckpt)
    out = infer.run(model, seqs, os.path.join(a.frames, ""))
    res, ious = {}, {}
    for k in ("pass1", "rc"):
        res[k], ious[k] = infer.evaluate(seqs, out[k])
    mod = [[TA.compute_iou(g, m) for g, m in zip(s["amodal_bboxes"], TA.modal_boxes(s))] for s in seqs]
    res["visible_mask_box"] = TA.score(mod)
    np.savez_compressed(os.path.join(a.out, "preds.npz"), pass1=out["pass1"], rc=out["rc"],
                        iou_pass1=np.array(ious["pass1"]), iou_rc=np.array(ious["rc"]))
    json.dump(dict(summary=res, n_seq=len(seqs), arm=a.arm, ckpt=a.ckpt, seconds=time.time() - t0),
              open(os.path.join(a.out, "results.json"), "w"), indent=1)
    for k, v in res.items():
        print(f"{k:18s} AP25 {v['ap25']:.2f}  AP50 {v['ap50']:.2f}  AP75 {v['ap75']:.2f}  ({v['n_frames']} frames)")


if __name__ == "__main__":
    main()
