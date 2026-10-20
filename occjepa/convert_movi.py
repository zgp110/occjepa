"""Stream EoRaS's MOVi-D / MOVi-B release (data.dgl.ai/dataset/EoRas/MOVi-{D,B}.tar.gz) into compact arrays.
  python -m occjepa.convert_movi --tar MOVi-D.tar.gz --out $OCCJEPA_DATA/movi/movid

Only what OccJEPA needs is kept (no depth / flow / BEV renders):
  <out>/<split>/frames.npy   uint8 [V, T, H, W, 3]   rgba_full_*.png (alpha dropped)
  <out>/<split>/segvis.npy   uint8 [V, T, H, W]      segmentation_full_*.png  (visible instance id + 1, 0 = background)
  <out>/<split>/amodal.npy   uint8 [I, T, H, W/8]    segmentation_<obj>_*.png == obj+1, np.packbits along W
  <out>/<split>/index.json   videos, instances [[video_idx, obj]], EoRaS instance_list (if readable without pickle),
                             per-video metadata subset (num_frames, size, bbox_frames per instance)
Nothing from the archive is executed or unpickled; members are read as bytes and decoded with PIL / json only.
"""
import argparse
import io
import json
import os
import re
import tarfile
import time

import numpy as np
from PIL import Image

from .npyio import NpyWriter

PAT = re.compile(r"^movi[a-z]_bev/(train|test)/(\d+)/(.+)$")
RGB = re.compile(r"^rgba_full_(\d{5})\.png$")
SEG = re.compile(r"^segmentation_full_(\d{5})\.png$")
AMO = re.compile(r"^segmentation_(\d+)_(\d{5})\.png$")


def png(b):
    return np.array(Image.open(io.BytesIO(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--partial", action="store_true", help="tolerate a truncated archive (smoke test)")
    a = ap.parse_args()
    vids = {}   # (split, vid) -> dict(rgb={f: arr}, seg={f: arr}, amo={(obj, f): packed}, meta=dict)
    lists = {}
    t0 = time.time()
    n = 0
    with open(a.tar, "rb") as fh:
        tf = tarfile.open(fileobj=fh, mode="r|gz")
        it = iter(tf)
        while True:
            try:
                m = next(it)
            except StopIteration:
                break
            except (EOFError, tarfile.ReadError, OSError) as e:
                if not a.partial:
                    raise
                print("truncated archive:", e)
                break
            n += 1
            if n % 20000 == 0:
                print(f"{n} members {time.time() - t0:.0f}s videos {len(vids)}", flush=True)
                os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)  # archive page cache (memory rule)
            if not m.isfile():
                continue
            name = m.name
            if name.startswith("/") or ".." in name.split("/"):
                print("skip suspicious", name)
                continue
            mm = PAT.match(name)
            if mm is None:
                base = os.path.basename(name)
                if base == "instance_list.npy":
                    split = name.split("/")[-2]
                    b = tf.extractfile(m).read()
                    try:
                        arr = np.load(io.BytesIO(b), allow_pickle=False)
                        lists[split] = [str(x) for x in arr.tolist()]
                    except Exception as e:  # object arrays need pickle: never unpickle untrusted data
                        print("instance_list not readable without pickle:", split, e)
                continue
            split, vid, fn = mm.group(1), int(mm.group(2)), mm.group(3)
            if not (RGB.match(fn) or SEG.match(fn) or AMO.match(fn) or fn == "metadata.json"):
                continue
            b = tf.extractfile(m).read()
            d = vids.setdefault((split, vid), dict(rgb={}, seg={}, amo={}, meta=None))
            if fn == "metadata.json":
                md = json.loads(b)
                d["meta"] = dict(num_frames=md["metadata"]["num_frames"], height=md["metadata"]["height"],
                                 width=md["metadata"]["width"],
                                 bbox_frames=[ins["bbox_frames"] for ins in md["instances"]],
                                 n_inst=len(md["instances"]))
            elif RGB.match(fn):
                d["rgb"][int(RGB.match(fn).group(1))] = png(b)[..., :3].astype(np.uint8)
            elif SEG.match(fn):
                d["seg"][int(SEG.match(fn).group(1))] = png(b)
            else:
                g = AMO.match(fn)
                obj, f = int(g.group(1)), int(g.group(2))
                d["amo"][(obj, f)] = np.packbits(png(b) == obj + 1, axis=-1)
    print(f"stream done: {n} members, {len(vids)} videos, {time.time() - t0:.0f}s", flush=True)

    for split in ("train", "test"):
        keys = sorted(k for k in vids if k[0] == split and vids[k]["meta"] is not None
                      and len(vids[k]["rgb"]) == vids[k]["meta"]["num_frames"]
                      and len(vids[k]["seg"]) == vids[k]["meta"]["num_frames"])
        print(split, "complete videos", len(keys), "of", sum(1 for k in vids if k[0] == split), flush=True)
        if not keys:
            continue
        od = os.path.join(a.out, split)
        os.makedirs(od, exist_ok=True)
        d0 = vids[keys[0]]
        T = d0["meta"]["num_frames"]
        H, W = d0["meta"]["height"], d0["meta"]["width"]
        insts = []
        for vi, k in enumerate(keys):
            d = vids[k]
            assert d["meta"]["num_frames"] == T and (d["meta"]["height"], d["meta"]["width"]) == (H, W), k
            assert len(d["rgb"]) == T and len(d["seg"]) == T, (k, len(d["rgb"]), len(d["seg"]))
            for obj in range(d["meta"]["n_inst"]):
                if all((obj, f) in d["amo"] for f in range(T)):
                    insts.append([vi, obj])
                else:
                    print("missing amodal frames", k, obj)
        fw = NpyWriter(os.path.join(od, "frames.npy"), (len(keys), T, H, W, 3), np.uint8)
        sw = NpyWriter(os.path.join(od, "segvis.npy"), (len(keys), T, H, W), np.uint8)
        aw = NpyWriter(os.path.join(od, "amodal.npy"), (len(insts), T, H, W // 8), np.uint8)
        for vi, k in enumerate(keys):
            d = vids[k]
            fw.write_rows([vi], np.stack([d["rgb"][f] for f in range(T)])[None])
            seg = np.stack([d["seg"][f] for f in range(T)])
            assert seg.max() < 256
            sw.write_rows([vi], seg.astype(np.uint8)[None])
        for ii, (vi, obj) in enumerate(insts):
            d = vids[keys[vi]]
            aw.write_rows([ii], np.stack([d["amo"][(obj, f)] for f in range(T)])[None])
        for w in (fw, sw, aw):
            w.close()
        index = dict(split=split, T=T, H=H, W=W, videos=[k[1] for k in keys], instances=insts,
                     eoras_instance_list=lists.get(split),
                     meta={str(k[1]): vids[k]["meta"] for k in keys})
        json.dump(index, open(os.path.join(od, "index.json"), "w"))
        print(split, "videos", len(keys), "instances", len(insts), "T,H,W", T, H, W, flush=True)
    print("DONE", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
