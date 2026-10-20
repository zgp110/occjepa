"""Read members of a remote (HF-hosted) zip via HTTP range requests, so only the needed frames are downloaded.
The HF token is read from $HF_TOKEN_PATH and only ever placed in the request header (never printed or logged)."""
from __future__ import annotations

import io
import os
import time

import requests

HF_BASE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def _token():
    p = os.environ.get("HF_TOKEN_PATH", os.path.expanduser("~/.cache/huggingface/token"))
    return open(p).read().strip()


class RemoteFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests with a small block cache."""

    def __init__(self, repo, path, block=1 << 20, max_blocks=64):
        self.url = HF_BASE.format(repo=repo, path=path)
        self.s = requests.Session()
        self.s.headers["Authorization"] = "Bearer " + _token()
        r = self.s.head(self.url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])
        self.final_url = r.url          # signed CDN URL (no auth header needed, but harmless)
        self.pos = 0
        self.block, self.max_blocks, self.cache = block, max_blocks, {}
        self.nbytes = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _get(self, a, b):  # inclusive range [a, b]
        for attempt in range(8):
            try:
                url = self.final_url if attempt < 4 else self.url
                hdr = {"Range": f"bytes={a}-{b}"}
                r = (requests.get(url, headers=hdr, timeout=120) if url != self.url
                     else self.s.get(url, headers=hdr, timeout=120, allow_redirects=True))
                if r.status_code in (403, 401) and url != self.url:   # signed URL expired: refresh
                    r2 = self.s.head(self.url, allow_redirects=True, timeout=60)
                    self.final_url = r2.url
                    continue
                r.raise_for_status()
                assert r.status_code == 206, r.status_code
                self.nbytes += len(r.content)
                return r.content
            except Exception as e:  # noqa: BLE001
                err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"range get failed {a}-{b}: {type(err).__name__}")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if n > 4 * self.block:   # big read: fetch directly
            out = self._get(self.pos, self.pos + n - 1)
            self.pos += n
            return out
        out = bytearray()
        while n > 0:
            bi = self.pos // self.block
            if bi not in self.cache:
                a = bi * self.block
                self.cache[bi] = self._get(a, min(self.size, a + self.block) - 1)
                if len(self.cache) > self.max_blocks:
                    self.cache.pop(next(iter(self.cache)))
            blk = self.cache[bi]
            o = self.pos - bi * self.block
            take = blk[o:o + n]
            out += take
            self.pos += len(take)
            n -= len(take)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)
