"""Score eval_movi exports (CPU).
  python -m occjepa.score_movi score --data movid --pred F.npz --which pred2 --out S.json [--per_frame P.npz]
  python -m occjepa.score_movi score --data movid --baseline vm|convex --out S.json
  python -m occjepa.score_movi pair --a P_a.npz --b P_b.npz --out D.json     (a - b, paired, video-clustered bootstrap)
Countings, per instance of the test instance list and per frame t (IoU = float32(inter) / (float32(union) + 1e-6), as
EoRaS model/loss.py::iou):
  E (EoRaS test_movi.py): frames = metadata bbox_frames; mIoU = mean IoU(pred, amodal); mIoU_occ = mean of
    IoU(pred & ~vis, amodal & ~vis) over the frames where it is != 1 (empty/empty = 0 is counted).
  A (whole frame; the Kubric numbers): frames with a non-empty amodal mask (includes fully hidden frames).
  D (Diffusion-VAS SAIL-VOS rule): frames = bbox_frames; occluded IoU of the 5x5-eroded parts, IoU < 1e-6 dropped.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import cv2

from . import paths

DATA = dict(paths.MOVI, **{k: os.path.dirname(v) for k, v in paths.KUBRIC.items()})
class Rows:
    def __init__(self, path):
        self.f = open(path, "rb")
        ver = np.lib.format.read_magic(self.f)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if ver == (1, 0)
                                 else np.lib.format.read_array_header_2_0)(self.f)
        assert not fortran
        self.off, self.shape, self.dtype = self.f.tell(), shape, np.dtype(dtype)
        self.n = int(np.prod(shape[1:]))

    def row(self, i):
        self.f.seek(self.off + i * self.n * self.dtype.itemsize)
        return np.fromfile(self.f, self.dtype, self.n).reshape(self.shape[1:])

    def close(self):
        os.posix_fadvise(self.f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        self.f.close()


def hull(m):
    pts = cv2.findNonZero(m.astype(np.uint8))
    out = np.zeros(m.shape, np.uint8)
    if pts is not None:
        cv2.fillPoly(out, [cv2.convexHull(pts)], 1)
    return out.astype(bool)


def iou(p, g):
    i = np.logical_and(p, g).sum()
    u = np.logical_or(p, g).sum()
    return float(np.float32(i) / (np.float32(u) + np.float32(1e-6)))


def erode(m):
    return cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


N_BOOT, SEED = 2000, 0


def eoras_ids(idx):
    vid_of = {v: k for k, v in enumerate(idx["videos"])}
    ii_of = {(int(v), int(o)): k for k, (v, o) in enumerate(idx["instances"])}
    return sorted(ii_of[(vid_of[int(s.split("-")[0])], int(s.split("-")[1]))] for s in idx["eoras_instance_list"])


def summarize(f):
    E, A = f["inE"], f["inA"]
    occ, od = f["occ"], f["occ_d"]
    eo, ao = E & (occ != 1.0), A & (occ != 1.0)
    dd = E & (od >= 1e-6)
    m = lambda x, k: float(100 * x[k].mean()) if k.any() else float("nan")  # noqa: E731
    return dict(E_mIoU=m(f["full"], E), E_mIoU_occ=m(occ, eo), A_mIoU=m(f["full"], A), A_mIoU_occ=m(occ, ao),
                D_mIoU_occ=m(od, dd), n=dict(E=int(E.sum()), E_occ=int(eo.sum()), A=int(A.sum()), A_occ=int(ao.sum()),
                                             D_occ=int(dd.sum())))


def score(a):
    d = os.path.join(DATA[a.data], "test")
    idx = json.load(open(os.path.join(d, "index.json")))
    W = idx["W"]
    ids = eoras_ids(idx)
    if a.pred:
        z = np.load(a.pred)
        assert str(z["data"]) == a.data, (str(z["data"]), a.data)
        inst = z["inst"].tolist()
        assert inst == ids[:len(inst)], "export instances differ from the EoRaS list order"
        P = z[a.which]
        ids = inst
        meta = dict(pred=os.path.abspath(a.pred), which=a.which, mode=str(z["mode"]), run=str(z["run"]), arm=str(z["arm"]),
                    ckpt=str(z["ckpt"]))
    else:
        P = None
        meta = dict(baseline=a.baseline)
    seg, amo = Rows(os.path.join(d, "segvis.npy")), Rows(os.path.join(d, "amodal.npy"))
    rows = {k: [] for k in ("pos", "video", "t", "inE", "inA", "full", "occ", "occ_d")}
    cur_v = sv = None
    for k, ii in enumerate(ids):
        v, o = idx["instances"][ii]
        if v != cur_v:
            cur_v, sv = v, seg.row(v)
        m_all = sv == o + 1
        a_all = np.unpackbits(amo.row(ii), axis=-1)[..., :W].astype(bool)
        bf = set(idx["meta"][str(idx["videos"][v])]["bbox_frames"][o])
        if P is not None:
            p_all = np.unpackbits(P[k], axis=-1)[..., :W].astype(bool)
        for t in range(a_all.shape[0]):
            m, g = m_all[t], a_all[t]
            inE, inA = t in bf, bool(g.any())
            if not (inE or inA):
                continue
            if P is not None:
                p = p_all[t]
            elif a.baseline == "vm":
                p = m
            else:
                p = hull(m) | m
            go, po = g & ~m, p & ~m
            rows["pos"].append(k)
            rows["video"].append(v)
            rows["t"].append(t)
            rows["inE"].append(inE)
            rows["inA"].append(inA)
            rows["full"].append(iou(p, g))
            rows["occ"].append(iou(po, go))
            rows["occ_d"].append(iou(erode(po), erode(go)) if inE else np.nan)
    seg.close()
    amo.close()
    f = {k: np.asarray(v, bool if k in ("inE", "inA") else (np.float32 if k in ("full", "occ", "occ_d") else np.int32))
         for k, v in rows.items()}
    out = dict(summary=summarize(f), n_instances=len(ids), **meta)
    if a.baseline:
        out["reference"] = dict(eoras_paper="MOVi-D: VM 56.92; Convex 60.18 / 16.48")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    if a.per_frame:
        np.savez_compressed(a.per_frame, **f)
    print(json.dumps(out["summary"]))


def pair(a):
    fa, fb = dict(np.load(a.a)), dict(np.load(a.b))
    for k in ("pos", "video", "t", "inE", "inA"):
        assert np.array_equal(fa[k], fb[k]), f"frame sets differ in {k}"
    vids = np.unique(fa["video"])
    rows_of = {v: np.nonzero(fa["video"] == v)[0] for v in vids}
    keys = ("E_mIoU", "E_mIoU_occ", "A_mIoU", "A_mIoU_occ", "D_mIoU_occ")
    sa, sb = summarize(fa), summarize(fb)
    rng = np.random.default_rng(SEED)
    boot = {k: [] for k in keys}
    for _ in range(N_BOOT):
        sel = np.concatenate([rows_of[v] for v in rng.choice(vids, len(vids))])
        ba, bb = summarize({k: x[sel] for k, x in fa.items()}), summarize({k: x[sel] for k, x in fb.items()})
        for k in keys:
            boot[k].append(ba[k] - bb[k])
    out = dict(a=os.path.abspath(a.a), b=os.path.abspath(a.b), n_videos=int(len(vids)), n_boot=N_BOOT, seed=SEED,
               diff={k: dict(point=sa[k] - sb[k], ci95=[float(np.nanpercentile(boot[k], 2.5)),
                                                        float(np.nanpercentile(boot[k], 97.5))]) for k in keys},
               a_summary=sa, b_summary=sb)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["diff"]))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("score")
    s.add_argument("--data", choices=sorted(DATA), required=True)
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--pred")
    g.add_argument("--baseline", choices=("vm", "convex"))
    s.add_argument("--which", default="pred1", choices=("pred1", "pred2"))
    s.add_argument("--out", required=True)
    s.add_argument("--per_frame")
    p = sp.add_parser("pair")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    score(a) if a.cmd == "score" else pair(a)


if __name__ == "__main__":
    main()
