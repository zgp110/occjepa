"""SAIL-VOS object-centred crop cache, built from a Diffusion-VAS sequence json (HF kaihuac/diffusion_vas_datasets:
diffusion_vas_sailvos_{train,val}.json, 25-frame object sequences with per-frame COCO RLEs 'segmentation' = amodal and
'visible_mask' = modal) and the extracted official SAIL-VOS frames (800x1280 .bmp).

Crop rule (= the MOVi and TAO-Amodal pipelines): per frame a square box around the target's VISIBLE bbox, side
2 x max(h, w), clamped to [min(H, W)/8, min(H, W)] = [100, 800] px and shifted inside the frame; fully hidden frames get
centre/side interpolated from the visible frames. --resample grid (used for the paper): F.grid_sample(bilinear,
align_corners=False, zero padding) on the full frame, crops at S = 256, targets at O = 128 (masks > 0.5).

The paper's caches:
  python -m occjepa.sailvos_cache --split svtrain_grid --seq_json $D/diffusion_vas_sailvos_train.json \
      --rgb_root $OCCJEPA_DATA/sailvos/train_raw --resample grid --no_pad \
      --exclude occjepa/resources/sailvos_train_excluded_cross_video.json      (117 sequences span two videos)
  python -m occjepa.sailvos_cache --split svval_grid --seq_json $D/diffusion_vas_sailvos_val.json \
      --rgb_root $OCCJEPA_DATA/sailvos/val_raw --resample grid --no_pad

Layout of <root>/cache_<split>/ (row i = sequence src_index[i] of the json): index.json; meta.npz (vid, obj_id,
image_ids [N,L], file_idx, modal/amodal bboxes, vis/amo/occ areas, crop boxes box_in [N,L,3] (y0, x0, side), edge,
ok, src_index); rle.bin + rle_off/rle_len (full-resolution GT RLEs for the metric); rgb.npy [N,L,S,S,3] uint8;
modal/amodal.npy [N,L,S,S/8] and modal_t/amodal_t.npy [N,L,O,O/8] packed bits; done/<chunk>.ok (resumable builds).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict

import numpy as np

from pycocotools import mask as cocomask

from . import paths

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = paths.SAILVOS_CACHE
CG = f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/memory.current"   # optional memory read-out (Linux cgroup v2)
ARR = ("rgb", "modal", "amodal", "modal_t", "amodal_t")
SPLIT_VIDEOS = os.path.join(HERE, "resources", "sailvos_official_split_videos.json")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cg_gib():
    try:
        return int(open(CG).read()) / 2 ** 30
    except OSError:
        return 0.0


def drop(path, sync=False):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        if sync:
            os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def row_shapes(L, S, O):
    return dict(rgb=(L, S, S, 3), modal=(L, S, S // 8), amodal=(L, S, S // 8), modal_t=(L, O, O // 8),
                amodal_t=(L, O, O // 8))


def npy_create(path, shape, dtype=np.uint8):
    """Pre-sized .npy (sparse until written); reused as is if it already has this header (resume)."""
    dtype = np.dtype(dtype)
    if os.path.exists(path):
        with open(path, "rb") as f:
            v = np.lib.format.read_magic(f)
            rd = np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0
            shp, fo, dt = rd(f)
            off = f.tell()
        if tuple(shp) == tuple(shape) and dt == dtype and not fo:
            return off
        raise SystemExit(f"{path} exists with another shape {shp} != {shape}; delete it to rebuild")
    hdr = {"descr": np.lib.format.dtype_to_descr(dtype), "fortran_order": False, "shape": tuple(shape)}
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, hdr)
        off = f.tell()
        f.truncate(off + int(np.prod(shape)) * dtype.itemsize)
    return off


# ---------------------------------------------------------------------------------------------------------------- #
def seq_boxes(mb, H, W, enlarge, min_side, pH, pW):
    """mb [L,4] modal bbox xywh (w = 0: hidden) -> (box_in, box_pad) [L,3] (y0, x0, side).  Exactly the rules of
    data.movi.MoviRAM._stats (inframe) and tao.boxes_padded (padded)."""
    L = len(mb)
    box_in = np.zeros((L, 3), np.int32)
    box_pad = np.zeros((L, 3), np.int32)
    ok = (mb[:, 2] > 0) & (mb[:, 3] > 0)
    if not ok.any():
        box_in[:] = (0, 0, min(H, W))
        box_pad[:] = (0, 0, min(H, W))
        return box_in, box_pad
    cy = np.where(ok, mb[:, 1] + mb[:, 3] / 2, np.nan)
    cx = np.where(ok, mb[:, 0] + mb[:, 2] / 2, np.nan)
    sd = np.where(ok, np.maximum(mb[:, 2], mb[:, 3]) * enlarge, np.nan)
    tt = np.arange(L)
    cy, cx, sd = (np.interp(tt, tt[ok], v[ok]) for v in (cy, cx, sd))
    max_pad = min(H + 2 * pH, W + 2 * pW)
    for t in range(L):
        side = int(min(max(round(sd[t]), min_side), H, W))
        box_in[t] = (int(np.clip(round(cy[t] - side / 2), 0, H - side)),
                     int(np.clip(round(cx[t] - side / 2), 0, W - side)), side)
        side = int(min(max(round(sd[t]), min_side), max_pad))
        box_pad[t] = (int(np.clip(round(cy[t] - side / 2), -pH, H + pH - side)),
                      int(np.clip(round(cx[t] - side / 2), -pW, W + pW - side)), side)
    return box_in, box_pad


def _rle(obj):
    """json RLE (dict with str/bytes counts, or uncompressed list counts) -> compressed pycocotools RLE."""
    if isinstance(obj, str):  # some COCO-style dumps store the dict as a python-literal string
        import ast
        obj = ast.literal_eval(obj)
    if isinstance(obj.get("counts"), list):
        return cocomask.frPyObjects(obj, *obj["size"])
    c = obj["counts"]
    return {"size": [int(obj["size"][0]), int(obj["size"][1])], "counts": c.encode() if isinstance(c, str) else c}


def _sub(name, subs):
    """--path_sub OLD=NEW string replacements on the json's frame names (e.g. '.png=.bmp' or 'rgb/=')."""
    for x in subs or ():
        o, n = x.split("=", 1)
        name = name.replace(o, n)
    return name


def check_videos(a, vnames, vid):
    """The video index feeds the in-run validation hold-out (inst[:,0] % 30 == 0): a wrong file-name layout would
    give one video for everything (empty train set) or one 'video' per sequence (silent train/val leak)."""
    cnt = np.bincount(vid, minlength=len(vnames))
    order = np.argsort(-cnt)
    log(f"videos: {len(vnames)}; sequences per video min/median/max {cnt.min()}/{int(np.median(cnt))}/{cnt.max()}; "
        f"top {[(vnames[j], int(cnt[j])) for j in order[:5]]}")
    assert len(vnames) >= 2 or len(vid) < 50, f"all sequences map to one video {vnames}: check image_file_names layout"
    bad = [v for v in vnames if os.path.splitext(v)[1].lower() in (".bmp", ".png", ".jpg", ".jpeg")]
    assert not bad, f"video names look like image files {bad[:5]}: image_file_names have no '<video>/' prefix"
    vs = a.video_set
    if vs == "auto":
        vs = "train" if a.split.startswith("train") else "val" if a.split.startswith("val") else "none"
    if vs == "none":
        log("WARNING video names not checked against the official split (--video_set none)")
        return
    allowed = set(json.load(open(a.video_list))[f"{vs}_videos"])
    bad = [v for v in vnames if v not in allowed]
    assert not bad, (f"{len(bad)} video names are not official SAIL-VOS {vs} videos, e.g. {bad[:5]}: fix the "
                     f"file-name layout with --path_sub (or --video_set none)")
    log(f"videos OK: all {len(vnames)} are official {vs} videos ({len(allowed)} in the split)")


def load_exclude(path):
    """--exclude file -> set of sequence indices of seq_json to drop: a json list, or a dict with 'drop_seq_indices'
    (resources/sailvos_train_excluded_cross_video.json).  None if no file."""
    if not path:
        return None
    d = json.load(open(path))
    lst = d["drop_seq_indices"] if isinstance(d, dict) else d
    return {int(j) for j in lst}


def refresh_ok(a, out, idx):
    """Re-check the frame files (resume / index built before extraction) and update meta.npz ok + index.json."""
    root = os.path.abspath(a.rgb_root)
    flist = idx["files"]
    miss = [f for f in flist if not os.path.isfile(os.path.join(root, f.lstrip("/")))]
    z = np.load(os.path.join(out, "meta.npz"))
    meta = {k: z[k] for k in z.files}
    if miss:
        ms = {j for j, f in enumerate(flist) if f in set(miss)}
        ok = np.array([not any(int(j) in ms for j in row) for row in meta["file_idx"]])
    else:
        ok = np.ones(len(meta["ok"]), bool)
    if root != idx["rgb_root"] or not np.array_equal(ok, meta["ok"]) or len(miss) != idx["n_missing_files"]:
        log(f"refresh: rgb_root {idx['rgb_root']} -> {root}; missing files {idx['n_missing_files']} -> {len(miss)}; "
            f"ok rows {int(meta['ok'].sum())} -> {int(ok.sum())}")
        meta["ok"] = ok
        np.savez(os.path.join(out, "meta.npz.tmp.npz"), **meta)
        os.replace(os.path.join(out, "meta.npz.tmp.npz"), os.path.join(out, "meta.npz"))
        idx.update(rgb_root=root, n_missing_files=len(miss), missing_files=miss[:200],
                   n_seq_missing_frames=int((~ok).sum()))
        json.dump(idx, open(os.path.join(out, "index.json"), "w"), indent=1)
    return idx


def build_index(a, out):
    t0 = time.time()
    log(f"loading {a.seq_json} (cgroup {cg_gib():.1f} GiB)")
    seqs = json.load(open(a.seq_json))
    drop(a.seq_json)
    t_load = time.time() - t0
    n_json = len(seqs)
    src_index = np.arange(n_json, dtype=np.int64)
    excl = load_exclude(a.exclude)
    if excl is not None:   # --exclude: drop these sequence indices of seq_json; meta 'src_index' maps row -> json index
        assert all(0 <= j < n_json for j in excl), "--exclude index out of range"
        src_index = np.array([j for j in range(n_json) if j not in excl], np.int64)
        seqs = [seqs[j] for j in src_index]
        log(f"--exclude {a.exclude}: dropped {len(excl)} of {n_json} sequences -> {len(seqs)}")
    if a.limit:
        seqs = seqs[:a.limit]
        src_index = src_index[:a.limit]
    N = len(seqs)
    L = len(seqs[0]["image_ids"])
    H = W = None
    files, fidx = {}, np.zeros((N, L), np.int32)
    videos = {}
    vid = np.zeros(N, np.int32)
    obj_id = np.zeros(N, np.int64)
    cat_id = np.zeros(N, np.int64)
    image_ids = np.zeros((N, L), np.int64)
    bbox_json = np.zeros((N, L, 4), np.float32)
    mbox = np.zeros((N, L, 4), np.float32)
    abox = np.zeros((N, L, 4), np.float32)
    vis_area = np.zeros((N, L), np.int64)
    amo_area = np.zeros((N, L), np.int64)
    occ_area = np.zeros((N, L), np.int64)
    rle_off = np.zeros((N, L, 2), np.int64)
    rle_len = np.zeros((N, L, 2), np.int32)
    keys = {}
    n_dup = 0
    sizes = set()
    pos = 0
    with open(os.path.join(out, "rle.bin.tmp"), "wb") as fb:
        for i, s in enumerate(seqs):
            assert len(s["image_ids"]) == L == len(s["segmentation"]) == len(s["visible_mask"]) == \
                len(s["image_file_names"]), f"seq {i}: inconsistent lengths"
            names = [_sub(n, a.path_sub) for n in s["image_file_names"]]
            vname = names[0].lstrip("/").split("/")[0]
            vid[i] = videos.setdefault(vname, len(videos))
            obj_id[i] = int(s["obj_id"])
            cat_id[i] = int(s.get("category_id", -1))
            for t in range(L):
                fidx[i, t] = files.setdefault(names[t], len(files))
                image_ids[i, t] = int(s["image_ids"][t])
                k = (int(s["image_ids"][t]), int(s["obj_id"]))
                if k in keys:
                    n_dup += 1
                keys[k] = i
            bj = s.get("bbox")
            if bj is not None:
                bbox_json[i] = np.asarray([b if b is not None else [0, 0, 0, 0] for b in bj], np.float32)
            A = [_rle(r) for r in s["segmentation"]]
            M = [_rle(r) for r in s["visible_mask"]]
            for r in A + M:
                sizes.add(tuple(r["size"]))
            amo_area[i] = cocomask.area(A)
            vis_area[i] = cocomask.area(M)
            abox[i] = cocomask.toBbox(A)
            mbox[i] = cocomask.toBbox(M)
            occ_area[i] = amo_area[i] - np.array([cocomask.area(cocomask.merge([x, y], intersect=True))
                                                  for x, y in zip(A, M)])
            for t in range(L):
                for c, r in enumerate((A[t], M[t])):
                    b = r["counts"]
                    fb.write(b)
                    rle_off[i, t, c] = pos
                    rle_len[i, t, c] = len(b)
                    pos += len(b)
            if (i + 1) % 2000 == 0:
                log(f"index {i + 1}/{N}")
        fb.flush()
        os.fsync(fb.fileno())
    del seqs
    vnames = [None] * len(videos)
    for k, v in videos.items():
        vnames[v] = k
    check_videos(a, vnames, vid)
    assert len(sizes) == 1, f"RLE sizes not uniform: {sizes}"
    H, W = sizes.pop()
    min_side = min(H, W) / 8
    pH, pW = H // 4, W // 4
    box_in = np.zeros((N, L, 3), np.int32)
    box_pad = np.zeros((N, L, 3), np.int32)
    for i in range(N):
        box_in[i], box_pad[i] = seq_boxes(mbox[i].astype(np.float64), H, W, a.enlarge, min_side, pH, pW)
    diff = np.any(box_in != box_pad, -1)
    pad_row = np.full((N, L), -1, np.int32)
    if not a.no_pad:
        pad_row[diff] = np.arange(int(diff.sum()), dtype=np.int32)
    has_a = amo_area > 0
    edge = np.stack([has_a & (abox[..., 1] <= 0), has_a & (abox[..., 1] + abox[..., 3] >= H),
                     has_a & (abox[..., 0] <= 0), has_a & (abox[..., 0] + abox[..., 2] >= W)], -1)
    # frame files
    flist = [None] * len(files)
    for k, v in files.items():
        flist[v] = k
    miss = [f for f in flist if not os.path.isfile(os.path.join(a.rgb_root, f.lstrip("/")))]
    miss_set = set(miss)
    ok = np.array([not any(flist[j] in miss_set for j in fidx[i]) for i in range(N)]) if miss else np.ones(N, bool)
    order = np.lexsort((image_ids[:, 0], vid))  # by video, then first frame: frame-cache locality
    np.savez(os.path.join(out, "meta.npz"), vid=vid, obj_id=obj_id, cat_id=cat_id, image_ids=image_ids,
             file_idx=fidx, bbox_json=bbox_json, modal_bbox=mbox, amodal_bbox=abox, vis_area=vis_area,
             amo_area=amo_area, occ_area=occ_area, box_in=box_in, box_pad=box_pad, pad_row=pad_row, edge=edge, ok=ok,
             order=order, src_index=src_index)
    np.save(os.path.join(out, "rle_off.npy"), rle_off)
    np.save(os.path.join(out, "rle_len.npy"), rle_len)
    os.replace(os.path.join(out, "rle.bin.tmp"), os.path.join(out, "rle.bin"))
    S, O = a.S, a.O
    per_seq = sum(int(np.prod(v)) for v in row_shapes(L, S, O).values())
    per_frame = per_seq // L
    M = int((pad_row >= 0).sum())
    idx = dict(split=a.split, seq_json=os.path.abspath(a.seq_json), n_seq_json=n_json,
               exclude=os.path.abspath(a.exclude) if a.exclude else None, n_excluded=0 if excl is None else len(excl),
               rgb_root=os.path.abspath(a.rgb_root), N=N, L=L,
               H=int(H), W=int(W), S=S, O=O, enlarge=a.enlarge, min_side=min_side, pad=[pH, pW], M_pad=M,
               chunk=a.chunk, path_sub=a.path_sub, resample=a.resample, videos=vnames,
               video_seq_counts=np.bincount(vid, minlength=len(vnames)).tolist(), files=flist, n_missing_files=len(miss), missing_files=miss[:200],
               n_seq_missing_frames=int((~ok).sum()), n_dup_keys=n_dup, n_unique_keys=len(keys),
               frac_frames_pad_differs=float(diff.mean()), bytes_per_seq_inframe=per_seq,
               bytes_pad=M * per_frame, bytes_total=N * per_seq + M * per_frame + pos,
               rle_bytes=pos, t_json_load=t_load, t_index=time.time() - t0,
               crop_rule="per-frame square, side 2 x max(h,w) of visible bbox, clamp [min(H,W)/8, min(H,W)], shifted "
                         "inside the frame; hidden frames interpolated; " + (
                             "resize cv2 INTER_AREA (shrink) / LINEAR (grow), masks thresholded at coverage > 1/2"
                             if a.resample == "cv2" else "MoviRAM._grid + grid_sample bilinear align_corners=False "
                                                         "(MOVi/TAO kernel), masks > 0.5"),
               notes="row i = sequence src_index[i] of seq_json (meta.npz; = i without --exclude); RLE 0 = amodal "
                     "('segmentation'), 1 = modal ('visible_mask')")
    json.dump(idx, open(os.path.join(out, "index.json"), "w"), indent=1)
    log(f"index: N={N} L={L} HxW={H}x{W} videos={len(videos)} frames={len(flist)} missing={len(miss)} "
        f"dup_keys={n_dup} pad_frames={M} ({diff.mean() * 100:.1f}%) -> est. {idx['bytes_total'] / 1e9:.1f} GB, "
        f"{time.time() - t0:.0f}s")
    return idx


# ---------------------------------------------------------------------------------------------------------------- #
_W = {}


def _init(out):
    import cv2
    cv2.setNumThreads(1)
    idx = json.load(open(os.path.join(out, "index.json")))
    z = np.load(os.path.join(out, "meta.npz"))
    if idx.get("resample", "cv2") == "grid":
        import torch
        torch.set_num_threads(1)
        _W["torch"] = torch
    _W.update(idx=idx, meta={k: z[k] for k in ("file_idx", "box_in", "box_pad", "pad_row", "ok")},
              rle_off=np.load(os.path.join(out, "rle_off.npy")), rle_len=np.load(os.path.join(out, "rle_len.npy")),
              rle_fd=os.open(os.path.join(out, "rle.bin"), os.O_RDONLY), frames=OrderedDict(), out=out, cv2=cv2)
    _W["fd"], _W["off"], _W["rb"] = {}, {}, {}
    for name, (p, shape) in array_specs(out, idx).items():
        _W["off"][name] = npy_create(p, shape)   # exists already (created by the parent): header check only
        _W["fd"][name] = os.open(p, os.O_WRONLY)
        _W["rb"][name] = int(np.prod(shape[1:]))


def array_specs(out, idx):
    shp = row_shapes(idx["L"], idx["S"], idx["O"])
    specs = {}
    for k in ARR:
        specs[k] = (os.path.join(out, k + ".npy"), (idx["N"],) + shp[k])
        if idx["M_pad"] > 0:
            specs[k + "_pad"] = (os.path.join(out, k + "_pad.npy"), (idx["M_pad"],) + shp[k][1:])
    return specs


def _frame(j):
    fr = _W["frames"]
    if j in fr:
        fr.move_to_end(j)
        return fr[j]
    p = os.path.join(_W["idx"]["rgb_root"], _W["idx"]["files"][j].lstrip("/"))
    fd = os.open(p, os.O_RDONLY)
    try:
        n = os.fstat(fd).st_size
        buf = os.pread(fd, n, 0)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    im = _W["cv2"].imdecode(np.frombuffer(buf, np.uint8), _W["cv2"].IMREAD_COLOR)
    assert im is not None, p
    im = np.ascontiguousarray(im[:, :, ::-1])
    H, W = _W["idx"]["H"], _W["idx"]["W"]
    assert im.shape[:2] == (H, W), (p, im.shape)
    fr[j] = im
    if len(fr) > 40:
        fr.popitem(last=False)
    return im


def _decode(i, t, c):
    o, n = int(_W["rle_off"][i, t, c]), int(_W["rle_len"][i, t, c])
    b = os.pread(_W["rle_fd"], n, o)
    return cocomask.decode({"size": [_W["idx"]["H"], _W["idx"]["W"]], "counts": b})


def crop_resize(img, y0, x0, s, out):
    """Square crop (y0, x0, side) of img (zero padding outside), resized to out x out."""
    cv2 = _W["cv2"] if "cv2" in _W else __import__("cv2")
    H, W = img.shape[:2]
    if y0 >= 0 and x0 >= 0 and y0 + s <= H and x0 + s <= W:
        reg = img[y0:y0 + s, x0:x0 + s]
    else:
        reg = np.zeros((s, s) + img.shape[2:], img.dtype)
        ya, xa, yb, xb = max(0, y0), max(0, x0), min(H, y0 + s), min(W, x0 + s)
        if yb > ya and xb > xa:
            reg[ya - y0:yb - y0, xa - x0:xb - x0] = img[ya:yb, xa:xb]
    if s == out:
        return np.ascontiguousarray(reg)
    return cv2.resize(np.ascontiguousarray(reg), (out, out),
                      interpolation=cv2.INTER_AREA if s > out else cv2.INTER_LINEAR)


def grid_crop(x, b, out):
    """x [C,H,W] float tensor -> [C,out,out]: data.movi.MoviRAM._grid + F.grid_sample (bilinear, align_corners=False,
    zero padding), same float32 arithmetic."""
    torch = _W["torch"]
    H, W = x.shape[-2:]
    bb = torch.tensor([[float(v) for v in b]], dtype=torch.float32)                  # [1,3]
    j = (torch.arange(out).float() + 0.5) / out
    y = bb[:, 0:1] + j[None] * bb[:, 2:3]
    xx = bb[:, 1:2] + j[None] * bb[:, 2:3]
    yn, xn = 2 * y / H - 1, 2 * xx / W - 1
    g = torch.stack([xn[:, None, :].expand(-1, out, -1), yn[:, :, None].expand(-1, -1, out)], -1)
    return torch.nn.functional.grid_sample(x[None], g, mode="bilinear", align_corners=False)[0]


def _crops_grid(img, A, M, b, S, O):
    """img [H,W,3] uint8, A / M [H,W] (0/255) -> or, already converted, (xt [3,H,W], at [1,H,W], mt [1,H,W]) float
    tensors passed as img with A = M = None (one conversion per frame for the in-frame and the padded box)."""
    torch = _W["torch"]
    if A is None:
        xt, at, mt = img
    else:
        xt, at, mt = to_tensors(img, A, M)
    r = grid_crop(xt, b, S).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
    m = lambda x, o: np.packbits((grid_crop(x, b, o)[0] > 0.5).numpy(), axis=-1)  # noqa: E731
    return dict(rgb=r, modal=m(mt, S), amodal=m(at, S), modal_t=m(mt, O), amodal_t=m(at, O))


def to_tensors(img, A, M):
    torch = _W["torch"]
    return (torch.from_numpy(img).permute(2, 0, 1).float(), torch.from_numpy(A > 0).float()[None],
            torch.from_numpy(M > 0).float()[None])


def _crops(img, A, M, b, S, O):
    if "torch" in _W:
        return _crops_grid(img, A, M, b, S, O)
    y0, x0, s = (int(v) for v in b)
    r = crop_resize(img, y0, x0, s, S)
    m = lambda x, o: np.packbits(crop_resize(x, y0, x0, s, o) >= 128, axis=-1)  # noqa: E731
    return dict(rgb=r, modal=m(M, S), amodal=m(A, S), modal_t=m(M, O), amodal_t=m(A, O))


def _work(job):
    cid, ids = job
    idx, meta = _W["idx"], _W["meta"]
    L, S, O = idx["L"], idx["S"], idx["O"]
    t0 = time.time()
    while cg_gib() > 250:
        log(f"chunk {cid}: user cgroup {cg_gib():.0f} GiB > 250, waiting")
        time.sleep(30)
    written = []
    for i in ids:
        if not meta["ok"][i]:
            continue
        written.append(int(i))
        rows = {k: [] for k in ARR}
        for t in range(L):
            img = _frame(int(meta["file_idx"][i, t]))
            A = _decode(i, t, 0) * np.uint8(255)
            M = _decode(i, t, 1) * np.uint8(255)
            if "torch" in _W:   # grid kernel: convert the frame and masks once, use for both boxes
                img, A, M = to_tensors(img, A, M), None, None
            c = _crops(img, A, M, meta["box_in"][i, t], S, O)
            for k in ARR:
                rows[k].append(c[k])
            pr = int(meta["pad_row"][i, t])
            if pr >= 0:
                cp = _crops(img, A, M, meta["box_pad"][i, t], S, O)
                for k in ARR:
                    os.pwrite(_W["fd"][k + "_pad"], np.ascontiguousarray(cp[k]).tobytes(),
                              _W["off"][k + "_pad"] + pr * _W["rb"][k + "_pad"])
        for k in ARR:
            os.pwrite(_W["fd"][k], np.ascontiguousarray(np.stack(rows[k])).tobytes(),
                      _W["off"][k] + int(i) * _W["rb"][k])
    for fd in _W["fd"].values():
        os.fdatasync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.posix_fadvise(_W["rle_fd"], 0, 0, os.POSIX_FADV_DONTNEED)
    with open(os.path.join(_W["out"], "done", f"{cid:06d}.ok"), "w") as f:
        f.write(" ".join(str(i) for i in written))   # rows actually written (rows with missing frames skipped)
        f.flush()
        os.fsync(f.fileno())
    return cid, len(ids), time.time() - t0


def chunk_done(out, job, ok):
    """done/<cid>.ok exists and lists every buildable (ok) row of the chunk."""
    cid, ids = job
    p = os.path.join(out, "done", f"{cid:06d}.ok")
    if not os.path.exists(p):
        return False
    with open(p) as f:
        w = {int(x) for x in f.read().split()}
    return all(int(i) in w for i in ids if ok[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seq_json", required=True)
    ap.add_argument("--rgb_root", required=True)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--O", type=int, default=128)
    ap.add_argument("--enlarge", type=float, default=2.0)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_pad", action="store_true", help="do not store the sparse padded-crop rows")
    ap.add_argument("--index_only", action="store_true", help="index + size estimate only (no crops)")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N sequences only")
    ap.add_argument("--resample", default="cv2", choices=["cv2", "grid"],
                    help="crop kernel: cv2 area/linear (default, = val cache) or grid = the MOVi / TAO grid_sample kernel")
    ap.add_argument("--video_set", default="auto", choices=["auto", "train", "val", "none"],
                    help="official video list the sequence video names must belong to (auto: from --split)")
    ap.add_argument("--video_list", default=SPLIT_VIDEOS)
    ap.add_argument("--exclude", default=None,
                    help="json with sequence indices of --seq_json to drop (list, or dict with 'drop_seq_indices'); "
                         "meta.npz 'src_index' maps cache rows back to json indices")
    ap.add_argument("--path_sub", action="append", default=[],
                    help="OLD=NEW replacement applied to image_file_names before joining with --rgb_root (repeatable)")
    a = ap.parse_args()
    assert 1 <= a.workers <= 4, "<= 4 workers (CPU budget)"
    out = os.path.join(a.root, f"cache_{a.split}")
    os.makedirs(os.path.join(out, "done"), exist_ok=True)
    if os.path.exists(os.path.join(out, "index.json")):
        idx = json.load(open(os.path.join(out, "index.json")))
        assert idx["seq_json"] == os.path.abspath(a.seq_json) and idx["S"] == a.S and idx["O"] == a.O and \
            idx.get("resample", "cv2") == a.resample and \
            idx.get("exclude") == (os.path.abspath(a.exclude) if a.exclude else None), \
            "existing index built with other settings; use another --root/--split or delete it"
        log(f"resuming: index N={idx['N']} M_pad={idx['M_pad']} resample={idx.get('resample', 'cv2')}")
        idx = refresh_ok(a, out, idx)
    else:
        idx = build_index(a, out)
    if a.index_only:
        return
    order = np.load(os.path.join(out, "meta.npz"))["order"]
    ch = idx["chunk"]
    jobs = [(c, order[k:k + ch].tolist()) for c, k in enumerate(range(0, len(order), ch))]
    ok = np.load(os.path.join(out, "meta.npz"))["ok"]
    todo = [j for j in jobs if not chunk_done(out, j, ok)]
    for p, shape in array_specs(out, idx).values():
        npy_create(p, shape)
    log(f"crops: {len(todo)}/{len(jobs)} chunks to do, {a.workers} workers (cgroup {cg_gib():.1f} GiB)")
    t0 = time.time()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    n_seq = 0
    with ctx.Pool(a.workers, initializer=_init, initargs=(out,)) as pool:
        for k, (cid, n, dt) in enumerate(pool.imap_unordered(_work, todo)):
            n_seq += n
            if (k + 1) % 25 == 0 or k + 1 == len(todo):
                el = time.time() - t0
                log(f"chunk {k + 1}/{len(todo)}  {n_seq} seqs  {n_seq / el:.2f} seq/s  ETA "
                    f"{(len(todo) - k - 1) * el / (k + 1) / 60:.1f} min  cgroup {cg_gib():.1f} GiB")
    complete = all(chunk_done(out, j, ok) for j in jobs)
    sizes = {f: os.path.getsize(os.path.join(out, f)) for f in sorted(os.listdir(out)) if
             os.path.isfile(os.path.join(out, f))}
    st = dict(complete=complete, t_crops_this_run=time.time() - t0, seqs_this_run=n_seq, workers=a.workers,
              file_bytes=sizes, total_bytes=sum(sizes.values()))
    json.dump(st, open(os.path.join(out, "build_stats.json"), "w"), indent=1)
    for f in sizes:
        drop(os.path.join(out, f), sync=True)
    log(f"{'COMPLETE' if complete else 'INCOMPLETE'}: {st['total_bytes'] / 1e9:.1f} GB in {out}")


if __name__ == "__main__":
    main()
