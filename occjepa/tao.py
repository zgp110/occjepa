"""TAO-Amodal under the Diffusion-VAS protocol: sequences, object-centric crops on the padded canvas, two-pass inference
(re-crop) and the box metric. Functions are the ones that produced the paper's numbers.

Per sequence (25 annotated frames): frames resized to max side 720; per-frame square crop of side 2 x the longer side of
the visible-mask box (Eq. 2), which may extend H/4, W/4 beyond the image (the canvas Diffusion-VAS predicts on); two
24-frame windows [0,24) and [1,25), the later one overwriting the shared frames. The predicted amodal box = bbox of the
predicted mask (logits > 0, pasted at the crop size) UNION the visible-mask box. Re-crop: pass 2 recomputes the crops
with the same rule on U_t = pass-1 amodal mask on the canvas, union the visible mask.
Metric: AP25/50/75 = % of frames whose box IoU with the ground-truth amodal box exceeds 0.25/0.5/0.75 (strict).
"""
from __future__ import annotations

import ast
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as cocomask

T = 24


# ------------------------------------------------------------------------------------------------ metric (Diffusion-VAS)
def compute_iou(box1, box2):  # verbatim logic of their eval_utils.compute_iou (xywh, no +1)
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union != 0 else 0


def get_bbox_from_mask(mask):  # their get_bbox_from_mask: [x_min, y_min, w, h] with w = x_max - x_min + 1
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def union_bboxes(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    x0, y0 = min(x1, x2), min(y1, y2)
    return [x0, y0, max(x1 + w1, x2 + w2) - x0, max(y1 + h1, y2 + h2) - y0]


def load_annot(path, track_ids_path):
    seqs = json.load(open(path))
    tid = json.load(open(track_ids_path))
    for s in seqs:
        s["rel_track_id"] = int(s["track_id"]) - tid[str(int(s["video_id"]))] + 1
        s["keys"] = [str((int(s["video_id"]), f.split("/")[-1], str(s["rel_track_id"]))) for f in s["file_names"]]
    return seqs


def modal_boxes(s):
    out = []
    for r in s["rles"]:
        m = cocomask.decode({"counts": r, "size": [s["height"], s["width"]]})
        out.append(get_bbox_from_mask(m))
    return out


def score(per_seq_ious):
    """per_seq_ious: list (sequences) of lists (frames) of IoU. Pooled over frames, as their metric."""
    v = np.concatenate([np.asarray(x, float) for x in per_seq_ious])
    return dict(ap25=float((v > 0.25).mean() * 100), ap50=float((v > 0.5).mean() * 100),
                ap75=float((v > 0.75).mean() * 100), miou=float(v.mean() * 100), n_frames=int(len(v)))


def _parse(b):
    return ast.literal_eval(b) if isinstance(b, str) else b


# ------------------------------------------------------------------------------------------------ crops and windows
def boxes_padded(vis, H, W, enlarge, pH, pW):
    """The in-frame box rule, but the crop may extend up to (pH, pW) outside the image (zero padding)."""
    L = len(vis)
    min_side = min(H, W) / 8
    max_side = min(H + 2 * pH, W + 2 * pW)
    cy, cx, sd = np.full(L, np.nan), np.full(L, np.nan), np.full(L, np.nan)
    for t in range(L):
        if vis[t].any():
            ys, xs = np.nonzero(vis[t])
            cy[t], cx[t] = (ys.min() + ys.max() + 1) / 2, (xs.min() + xs.max() + 1) / 2
            sd[t] = max(ys.max() + 1 - ys.min(), xs.max() + 1 - xs.min()) * enlarge
    ok = ~np.isnan(cy)
    box = np.zeros((L, 3), np.int32)
    if not ok.any():
        box[:] = (0, 0, min(H, W))
        return box
    tt = np.arange(L)
    cy, cx, sd = (np.interp(tt, tt[ok], v[ok]) for v in (cy, cx, sd))
    for t in range(L):
        side = int(min(max(round(sd[t]), min_side), max_side))
        box[t] = (int(np.clip(round(cy[t] - side / 2), -pH, H + pH - side)),
                  int(np.clip(round(cx[t] - side / 2), -pW, W + pW - side)), side)
    return box


def load_seq(s, frames_root, max_side=720, crop="padded", enlarge=2.0):
    H0, W0 = int(s["height"]), int(s["width"])
    sc = min(1.0, max_side / max(H0, W0))
    W, H = int(round(W0 * sc)), int(round(H0 * sc))
    rgb, vin, mbox, bad = [], [], [], 0
    for f, r in zip(s["file_names"], s["rles"]):
        im = Image.open(frames_root + f).convert("RGB")
        if im.size != (W0, H0):
            bad += 1
            im = im.resize((W0, H0), Image.BILINEAR)
        rgb.append(np.array(im.resize((W, H), Image.BILINEAR)))
        m = cocomask.decode({"counts": r, "size": [H0, W0]})
        mbox.append(get_bbox_from_mask(m))
        vin.append(np.array(Image.fromarray(m * 255).resize((W, H), Image.NEAREST)) > 127)
    rgb, vin = np.stack(rgb), np.stack(vin)
    assert crop == "padded", crop
    box = boxes_padded(vin, H, W, enlarge, (H0 // 4) * sc, (W0 // 4) * sc)
    return dict(rgb=rgb, vin=vin, amo=vin, box=box, H=H, W=W, L=len(rgb), H0=H0, W0=W0, sc=sc, mbox=mbox,
                bad_size=bad)


class Windows:
    """The sequences of one batch and their 24-frame windows; batch() = the crop sampler of the paper's runs."""

    def __init__(self):
        self.vids, self.windows = [], []

    def set(self, seqs):
        self.vids, self.windows = seqs, []
        for vi, v in enumerate(seqs):
            L = v["L"]
            starts = list(range(0, max(1, L - T + 1), T))
            if L > T and starts[-1] + T < L:
                starts.append(L - T)
            self.windows += [(vi, st) for st in starts]

    def _frames(self, w):
        vi, st = self.windows[w]
        v = self.vids[vi]
        ft = [min(st + k, v["L"] - 1) for k in range(T)]   # pad short videos by repeating the last frame
        return v, ft

    def batch(self, ids, S=256, out_res=128, device="cuda"):
        clips, mod, occ, amo_t, mod_t = [], [], [], [], []
        for w in ids:
            v, ft = self._frames(w)
            fr = torch.from_numpy(v["rgb"][ft]).to(device).permute(0, 3, 1, 2).float()
            m = torch.from_numpy(v["vin"][ft]).to(device).float()[:, None]
            am = torch.from_numpy(v["amo"][ft]).to(device).float()[:, None]
            b = torch.from_numpy(v["box"][ft]).to(device).float()

            def grid(out):
                j = (torch.arange(out, device=device).float() + 0.5) / out
                y = b[:, 0:1] + j[None] * b[:, 2:3]
                x = b[:, 1:2] + j[None] * b[:, 2:3]
                yn, xn = 2 * y / v["H"] - 1, 2 * x / v["W"] - 1
                return torch.stack([xn[:, None, :].expand(-1, out, -1), yn[:, :, None].expand(-1, -1, out)], -1)

            gS, gO = grid(S), grid(out_res)
            gs = lambda t, g: F.grid_sample(t, g, mode="bilinear", align_corners=False)  # noqa: E731
            clips.append(gs(fr, gS).permute(1, 0, 2, 3))
            mod.append((gs(m, gS)[:, 0] > 0.5).float())
            occ.append(torch.zeros_like(mod[-1]))
            amo_t.append((gs(am, gO)[:, 0] > 0.5).float())
            mod_t.append((gs(m, gO)[:, 0] > 0.5).float())
        return dict(clip=torch.stack(clips), modal=torch.stack(mod), occl=torch.stack(occ),
                    amodal_t=torch.stack(amo_t), modal_t=torch.stack(mod_t))


def crop_pred_box(lg, b, sc):
    """lg [out,out] crop-space logits, b = (y0, x0, side) working-res crop -> xywh box (original pixels) of lg > 0,
    or None if empty.  Upsampled to the crop size in ORIGINAL pixels, then offset by the crop origin."""
    y0, x0, s = (float(q) / sc for q in b)
    n = max(1, int(round(s)))
    up = F.interpolate(lg[None, None].float(), size=(n, n), mode="bilinear", align_corners=False)[0, 0] > 0
    rows = torch.nonzero(up.any(1)).flatten()
    if len(rows) == 0:
        return None
    cols = torch.nonzero(up.any(0)).flatten()
    ymin, ymax, xmin, xmax = int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])
    return [int(round(x0)) + xmin, int(round(y0)) + ymin, xmax - xmin + 1, ymax - ymin + 1]


def bbox_of(m):
    """bool [H,W] -> (y0, y1, x0, x1) exclusive ends, or None."""
    r, c = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    if len(r) == 0:
        return None
    return int(r[0]), int(r[-1]) + 1, int(c[0]), int(c[-1]) + 1


def tao_rc_boxes(U, oy, ox, H, W, enlarge, pH, pW):
    """boxes_padded line by line, for a mask U on the padded canvas whose pixel (oy, ox) is image pixel (0, 0):
    the bbox is taken in IMAGE coordinates (canvas index - offset), then the identical arithmetic (same rounding)."""
    L = len(U)
    min_side = min(H, W) / 8
    max_side = min(H + 2 * pH, W + 2 * pW)
    cy, cx, sd = np.full(L, np.nan), np.full(L, np.nan), np.full(L, np.nan)
    for t in range(L):
        if U[t].any():
            ys, xs = np.nonzero(U[t])
            ys, xs = ys - oy, xs - ox
            cy[t], cx[t] = (ys.min() + ys.max() + 1) / 2, (xs.min() + xs.max() + 1) / 2
            sd[t] = max(ys.max() + 1 - ys.min(), xs.max() + 1 - xs.min()) * enlarge
    ok = ~np.isnan(cy)
    box = np.zeros((L, 3), np.int32)
    if not ok.any():
        box[:] = (0, 0, min(H, W))
        return box
    tt = np.arange(L)
    cy, cx, sd = (np.interp(tt, tt[ok], v[ok]) for v in (cy, cx, sd))
    for t in range(L):
        side = int(min(max(round(sd[t]), min_side), max_side))
        box[t] = (int(np.clip(round(cy[t] - side / 2), -pH, H + pH - side)),
                  int(np.clip(round(cx[t] - side / 2), -pW, W + pW - side)), side)
    return box
