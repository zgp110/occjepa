"""Two-pass inference of OccJEPA on sequences in the TAO-Amodal format (dict with file_names, rles, height, width):
pass 1 crops around the visible mask, pass 2 (re-crop) around the pass-1 amodal mask; both on the padded canvas.
Returns per-frame amodal boxes (xywh, original pixels; bbox of the predicted mask union the visible-mask box) of both
passes and, optionally, the pass-2 amodal masks on the padded canvas (offset oy, ox; working resolution).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import tao as TA


def _boxes_and_canvas(lg, ds, vs, ids, pred, keep_canvas):
    """pass a: per-frame boxes into pred[seq, t]; canvas masks per sequence (later window overwrites)."""
    canv = []
    for v in vs:
        pH, pW = (v["H0"] // 4) * v["sc"], (v["W0"] // 4) * v["sc"]
        oy, ox = int(np.ceil(pH)) + 1, int(np.ceil(pW)) + 1
        v.update(pH=pH, pW=pW, oy=oy, ox=ox)
        canv.append(np.zeros((v["L"], v["H"] + 2 * oy, v["W"] + 2 * ox), bool) if keep_canvas else None)
    for j, (vi, st) in enumerate(ds.windows):
        v = vs[vi]
        for k in range(TA.T):
            t = min(st + k, v["L"] - 1)
            pb = TA.crop_pred_box(lg[j][k], v["box"][t], v["sc"])
            pred[ids[vi]][t] = v["mbox"][t] if pb is None else TA.union_bboxes(pb, v["mbox"][t])
            if keep_canvas:
                y0, x0, s = (int(q) for q in v["box"][t])
                m = F.interpolate(lg[j][k][None, None].float(), size=(s, s), mode="bilinear", align_corners=False)[0, 0] > 0
                ya, xa = y0 + v["oy"], x0 + v["ox"]
                canv[vi][t] = False
                canv[vi][t, ya:ya + s, xa:xa + s] = m.cpu().numpy()
    return canv


@torch.no_grad()
def run(model, seqs, frames_root, recrop=True, seq_per_batch=2, max_side=720, return_masks=False, log=print):
    torch.backends.cuda.matmul.allow_tf32 = True
    N = len(seqs)
    L = max(len(s["file_names"]) for s in seqs)
    pred1 = np.zeros((N, L, 4), np.float64)
    pred2 = np.zeros((N, L, 4), np.float64)
    masks = {}
    S = model.cfg["S"]
    ds = TA.Windows()
    for it, k0 in enumerate(range(0, N, seq_per_batch)):
        ids = list(range(k0, min(N, k0 + seq_per_batch)))
        vs = [TA.load_seq(seqs[i], frames_root, max_side) for i in ids]
        ds.set(vs)
        wid = list(range(len(ds.windows)))
        lg = model.predict(ds.batch(wid, S=S))
        canv = _boxes_and_canvas(lg, ds, vs, ids, pred1, keep_canvas=recrop)
        del lg
        if not recrop:
            continue
        for vi, v in enumerate(vs):
            vin_c = np.zeros_like(canv[vi])
            vin_c[:, v["oy"]:v["oy"] + v["H"], v["ox"]:v["ox"] + v["W"]] = v["vin"]
            v["box_pass1"] = v["box"].copy()
            v["box"] = TA.tao_rc_boxes(canv[vi] | vin_c, v["oy"], v["ox"], v["H"], v["W"], 2.0, v["pH"], v["pW"])
        ds.set(vs)
        lg2 = model.predict(ds.batch(wid, S=S))
        canv2 = _boxes_and_canvas(lg2, ds, vs, ids, pred2, keep_canvas=return_masks)
        if return_masks:
            for vi, v in enumerate(vs):
                vin_c = np.zeros_like(canv2[vi])
                vin_c[:, v["oy"]:v["oy"] + v["H"], v["ox"]:v["ox"] + v["W"]] = v["vin"]
                masks[ids[vi]] = dict(mask=canv2[vi] | vin_c, oy=v["oy"], ox=v["ox"], sc=v["sc"])
        del lg2
        if log and ((it + 1) % 25 == 0 or k0 + seq_per_batch >= N):
            log(f"{min(N, k0 + seq_per_batch)}/{N} sequences")
    out = dict(pass1=pred1, rc=pred2 if recrop else None)
    if return_masks:
        out["masks"] = masks
    return out


def evaluate(seqs, preds):
    """Diffusion-VAS box metric of [N, 25, 4] predictions against the sequences' amodal boxes."""
    ious = [[TA.compute_iou(g, preds[i, t]) for t, g in enumerate(s["amodal_bboxes"])] for i, s in enumerate(seqs)]
    return TA.score(ious), ious
