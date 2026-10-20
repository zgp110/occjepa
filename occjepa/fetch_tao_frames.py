"""Download only the TAO-Amodal frames used by a sequence json (Diffusion-VAS val: 1,392 sequences x 25 frames; our
test split: 1,947 x 25) from the HF-gated zips (chengyenhsieh/TAO-Amodal frames/<split>/<Source>.zip), via HTTP range
requests on the zip central directory + the needed members. Writes <out>/<split>/<Source>/<video>/frameNNNN.jpg
(the layout of the sequences' file_names).
python -m occjepa.fetch_tao_frames --annot <sequence json> --out $OCCJEPA_DATA/tao_amodal/frames
(TAO-Amodal is gated on HF: accept the terms, then put your token in ~/.cache/huggingface/token or $HF_TOKEN_PATH)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import struct
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .remote_zip import RemoteFile

REPO = "chengyenhsieh/TAO-Amodal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--sources", default="")
    c = ap.parse_args()
    seqs = json.load(open(c.annot))
    need = collections.defaultdict(set)
    for s in seqs:
        for f in s["file_names"]:
            sp, src, rest = f.split("/", 2)
            need[(sp, src)].add(rest)
    print({k: len(v) for k, v in need.items()}, flush=True)
    todo_src = [k for k in sorted(need) if not c.sources or k[1] in c.sources.split(",")]
    tot_bytes = 0
    for sp, src in todo_src:
        t0 = time.time()
        rf = RemoteFile(REPO, f"frames/{sp}/{src}.zip")
        z = zipfile.ZipFile(rf)
        by_name = {i.filename: i for i in z.infolist()}
        # tolerate a top-level folder inside the zip
        alt = {}
        for n in by_name:
            parts = n.split("/")
            if len(parts) >= 2:
                alt["/".join(parts[-2:])] = n
        print(f"{src}: zip {rf.size/1e9:.2f} GB, {len(by_name)} members, CD read {rf.nbytes/1e6:.1f} MB "
              f"in {time.time()-t0:.0f}s", flush=True)
        lock = threading.Lock()
        url = [rf.final_url]

        def refresh():
            with lock:
                r = rf.s.head(rf.url, allow_redirects=True, timeout=60)
                url[0] = r.url

        def fetch(rest):
            dst = os.path.join(c.out, sp, src, rest)
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                return 0, "exists"
            zi = by_name.get(rest) or by_name.get(alt.get(rest, ""))
            if zi is None:
                return 0, "missing"
            a = zi.header_offset
            b = a + 30 + len(zi.orig_filename.encode()) + 1024 + zi.compress_size
            for attempt in range(10):
                try:
                    r = requests.get(url[0], headers={"Range": f"bytes={a}-{min(b, rf.size - 1)}"}, timeout=120)
                    if r.status_code in (401, 403):
                        refresh()
                        continue
                    r.raise_for_status()
                    buf = r.content
                    sig, = struct.unpack("<I", buf[:4])
                    assert sig == 0x04034B50, "bad local header"
                    n, m = struct.unpack("<HH", buf[26:30])
                    data = buf[30 + n + m: 30 + n + m + zi.compress_size]
                    assert len(data) == zi.compress_size, "short read"
                    if zi.compress_type == 8:
                        data = zlib.decompressobj(-15).decompress(data)
                    elif zi.compress_type != 0:
                        raise RuntimeError(f"compress_type {zi.compress_type}")
                    assert zlib.crc32(data) & 0xFFFFFFFF == zi.CRC, "crc"
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    tmp = dst + ".part"
                    with open(tmp, "wb") as fh:
                        fh.write(data)
                        fh.flush()
                        os.fsync(fh.fileno())
                        try:
                            os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                        except OSError:
                            pass
                    os.replace(tmp, dst)
                    return len(buf), "ok"
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {str(e)[:80]}"
                    time.sleep(1 + attempt)
            return 0, "fail " + err

        cnt = collections.Counter()
        with ThreadPoolExecutor(c.threads) as ex:
            futs = [ex.submit(fetch, r) for r in sorted(need[(sp, src)])]
            for k, fu in enumerate(as_completed(futs)):
                nb, st = fu.result()
                tot_bytes += nb
                cnt[st.split(":")[0]] += 1
                if st.startswith("fail") or st == "missing":
                    print("  ", st, flush=True)
                if (k + 1) % 500 == 0:
                    print(f"  {src} {k+1}/{len(futs)} {dict(cnt)} {tot_bytes/1e9:.2f} GB {time.time()-t0:.0f}s", flush=True)
        print(f"{src} done {dict(cnt)} in {time.time()-t0:.0f}s, total {tot_bytes/1e9:.2f} GB", flush=True)
        del z, by_name, alt, rf
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
