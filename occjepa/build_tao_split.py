"""Build Diffusion-VAS-protocol TAO-Amodal sequences for a split (the held-out test split of the paper: --split test).
Rule (reconstructed from Diffusion-VAS's released val json; on val it recovers all 1,389 of their tracks):
  videos  BURST sequences with fps == 30;
  tracks  every BURST track of the video; BURST local id k <-> TAO-Amodal track = k-th smallest track id of the video;
  window  the first 25 consecutive annotated frames in which the track has a BURST mask AND a TAO-Amodal annotation;
          tracks without such a run are skipped; one sequence per track;
  fields  file_names '<split>/<dataset>/<seq_name>/<frame>', rles = BURST rle, amodal_bboxes = TAO-Amodal amodal_bbox,
          visibilities = TAO-Amodal visibility, category_id = TAO-Amodal track category, frame_indices = all_image_paths
          index (unused by every evaluation), abs2rel[video] = smallest track id of the video (DVAS's key convention).
python -m occjepa.build_tao_split --split val|test --annot_root $OCCJEPA_DATA/tao_amodal --out_dir $OCCJEPA_DATA/tao_amodal
    (annot_root holds TAO-Amodal's BURST_annotations/ and amodal_annotations/ from HF chengyenhsieh/TAO-Amodal;
     val also compares with the Diffusion-VAS json in --dvas_dir, if given)
"""
import argparse
import collections
import json
import os

SRC = {"val": ("BURST_annotations/val/all_classes_visibility.json", "amodal_annotations/validation.json"),
       "test": ("BURST_annotations/test/all_classes_visibility.json", "amodal_annotations/test.json")}
L = 25


def first_run(ok):
    run = 0
    for i, x in enumerate(ok):
        run = run + 1 if x else 0
        if run == L:
            return i - L + 1
    return None


def build(split, root):
    bu = json.load(open(os.path.join(root, SRC[split][0])))
    am = json.load(open(os.path.join(root, SRC[split][1])))
    vtracks = collections.defaultdict(list)
    tcat = {}
    for t in am["tracks"]:
        vtracks[t["video_id"]].append(t["id"])
        tcat[t["id"]] = t["category_id"]
    img = {i["id"]: i for i in am["images"]}
    ann_by = collections.defaultdict(dict)
    for a in am["annotations"]:
        ann_by[a["track_id"]][img[a["image_id"]]["file_name"].split("/")[-1]] = a
    seqs, abs2rel, st = [], {}, collections.Counter()
    for b in bu["sequences"]:
        v = b["id"]
        tids = sorted(vtracks[v])
        if tids:
            abs2rel[str(v)] = tids[0]
            st["noncontiguous_track_ids"] += tids != list(range(tids[0], tids[0] + len(tids)))
        st["videos"] += 1
        if b["fps"] != 30:
            st["videos_fps_not_30"] += 1
            continue
        ap = b["annotated_image_paths"]
        for k in sorted(b["track_category_ids"], key=int):
            if int(k) > len(tids):
                st["burst_track_without_amodal_track"] += 1
                continue
            t = tids[int(k) - 1]
            ok = [k in b["segmentations"][i] and ap[i] in ann_by[t] for i in range(len(ap))]
            s0 = first_run(ok)
            if s0 is None:
                st["tracks_without_window"] += 1
                continue
            w = ap[s0:s0 + L]
            an = [ann_by[t][f] for f in w]
            seqs.append(dict(track_id=t, category_id=tcat[t], video_id=v,
                             frame_indices=[b["all_image_paths"].index(f) for f in w],
                             file_names=[f"{split}/{b['dataset']}/{b['seq_name']}/{f}" for f in w],
                             visibilities=[a["visibility"] for a in an], amodal_bboxes=[a["amodal_bbox"] for a in an],
                             width=b["width"], height=b["height"],
                             rles=[b["segmentations"][s0 + j][k]["rle"] for j in range(L)]))
    st["sequences"] = len(seqs)
    st["videos_with_sequences"] = len({s["video_id"] for s in seqs})
    st["dataset_counts"] = dict(collections.Counter(s["file_names"][0].split("/")[1] for s in seqs))
    return seqs, abs2rel, dict(st)


def check_val(seqs, abs2rel, D):
    dv = json.load(open(f"{D}/diffusion_vas_tao_amodal_val.json"))
    a2r = json.load(open(f"{D}/tao_amodal_track_ids_abs2rel_val.json"))
    ours = {(s["video_id"], s["track_id"]): s for s in seqs}
    theirs = collections.defaultdict(list)
    for s in dv:
        theirs[(s["video_id"], s["track_id"])].append(s)
    both = set(ours) & set(theirs)
    same_start = [k for k in both if any(o["file_names"][0] == ours[k]["file_names"][0] for o in theirs[k])]
    fld = collections.Counter()
    for k in same_start:
        o = next(x for x in theirs[k] if x["file_names"][0] == ours[k]["file_names"][0])
        for f in ("file_names", "rles", "amodal_bboxes", "visibilities", "category_id", "width", "height"):
            fld[f] += ours[k][f] == o[f]
    return dict(dvas_sequences=len(dv), dvas_tracks=len(theirs), ours=len(ours), both=len(both),
                dvas_only=len(set(theirs) - set(ours)), ours_only=len(set(ours) - set(theirs)), same_start=len(same_start),
                fields_identical_on_same_start=dict(fld),
                abs2rel_identical=sum(a2r.get(k) == v for k, v in abs2rel.items()), abs2rel_n=len(a2r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=sorted(SRC), required=True)
    ap.add_argument("--annot_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dvas_dir", default=None)
    a = ap.parse_args()
    seqs, abs2rel, st = build(a.split, a.annot_root)
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(seqs, open(f"{a.out_dir}/{a.split}_seqs.json", "w"))
    json.dump(abs2rel, open(f"{a.out_dir}/{a.split}_abs2rel.json", "w"))
    out = dict(split=a.split, stats=st)
    if a.split == "val" and a.dvas_dir:
        out["check_vs_dvas"] = check_val(seqs, abs2rel, a.dvas_dir)
    json.dump(out, open(f"{a.out_dir}/{a.split}_build.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
