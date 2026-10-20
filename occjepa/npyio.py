"""Memory-lean row I/O for the big .npy token caches (hundreds of GB).

Why: np.memmap keeps every touched page MAPPED in the reading process; mapped pages count against the process'
memory and cannot be dropped with posix_fadvise, which matters for random-row training over a 150 GB cache.
Here rows are read with os.pread into fresh buffers and the page cache of the read range is dropped immediately
(POSIX_FADV_DONTNEED); writes use pwrite + fdatasync + DONTNEED.  Nothing stays mapped or cached.
"""
from __future__ import annotations

import os
import threading
from queue import Queue
from typing import Optional, Sequence, Tuple

import numpy as np


def _header(path: str):
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        rd = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = rd(f)
        off = f.tell()
    assert not fortran
    return tuple(shape), np.dtype(dtype), off


class NpyRows:
    """Read rows (first axis) of a C-ordered .npy without mmap.  ``sub`` = (start, stop) on the 2nd axis selects a
    contiguous slab inside each row (e.g. a range of tubelets)."""

    def __init__(self, path: str):
        self.path = path
        self.shape, self.dtype, self.off = _header(path)
        self.row_elems = int(np.prod(self.shape[1:])) if len(self.shape) > 1 else 1
        self.row_bytes = self.row_elems * self.dtype.itemsize
        self.fd = os.open(path, os.O_RDONLY)

    def __len__(self):
        return self.shape[0]

    def _read(self, start: int, nbytes: int) -> bytes:
        buf = bytearray(nbytes)
        mv = memoryview(buf)
        got = 0
        while got < nbytes:
            n = os.preadv(self.fd, [mv[got:]], start + got)
            if n <= 0:
                raise IOError(f"short read {self.path} @ {start + got}")
            got += n
        os.posix_fadvise(self.fd, start, nbytes, os.POSIX_FADV_DONTNEED)
        return buf

    def row(self, i: int, sub: Optional[Tuple[int, int]] = None) -> np.ndarray:
        if sub is None:
            b = self._read(self.off + i * self.row_bytes, self.row_bytes)
            return np.frombuffer(b, self.dtype).reshape(self.shape[1:])
        inner = int(np.prod(self.shape[2:])) * self.dtype.itemsize
        s0, s1 = sub
        b = self._read(self.off + i * self.row_bytes + s0 * inner, (s1 - s0) * inner)
        return np.frombuffer(b, self.dtype).reshape((s1 - s0,) + tuple(self.shape[2:]))

    def rows(self, idx: Sequence[int], sub=None) -> np.ndarray:
        return np.stack([self.row(int(i), sub) for i in idx], 0)

    def chunk(self, start: int, stop: int) -> np.ndarray:
        """Contiguous rows [start, stop) in one read."""
        stop = min(stop, self.shape[0])
        b = self._read(self.off + start * self.row_bytes, (stop - start) * self.row_bytes)
        return np.frombuffer(b, self.dtype).reshape((stop - start,) + tuple(self.shape[1:]))

    def close(self):
        os.close(self.fd)


class NpyWriter:
    """Create a .npy of a given shape/dtype and fill rows with pwrite; every ``sync_every`` bytes the file is
    fdatasync'ed and its page cache dropped, so writing a 150 GB cache never holds more than ~sync_every in RAM."""

    def __init__(self, path: str, shape, dtype=np.float16, sync_every: int = 2 << 30):
        self.path, self.shape, self.dtype = path, tuple(shape), np.dtype(dtype)
        hdr = {"descr": np.lib.format.dtype_to_descr(self.dtype), "fortran_order": False, "shape": self.shape}
        with open(path, "wb") as f:
            np.lib.format.write_array_header_1_0(f, hdr)
            self.off = f.tell()
            f.truncate(self.off + int(np.prod(self.shape)) * self.dtype.itemsize)
        self.row_bytes = int(np.prod(self.shape[1:])) * self.dtype.itemsize
        self.fd = os.open(path, os.O_WRONLY)
        self.sync_every, self.pending = sync_every, 0

    def write_rows(self, idx: Sequence[int], arr: np.ndarray):
        arr = np.ascontiguousarray(arr, dtype=self.dtype)
        for k, i in enumerate(idx):
            os.pwrite(self.fd, arr[k].tobytes(), self.off + int(i) * self.row_bytes)
            self.pending += self.row_bytes
        if self.pending >= self.sync_every:
            self.flush()

    def flush(self):
        os.fdatasync(self.fd)
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)
        self.pending = 0

    def close(self):
        self.flush()
        os.close(self.fd)


def prefetch(gen, depth: int = 2):
    """Run a batch generator in a background thread with a bounded queue (no worker processes, no shared memory)."""
    q: Queue = Queue(maxsize=depth)
    stop = object()

    def run():
        try:
            for item in gen:
                q.put(item)
        finally:
            q.put(stop)

    threading.Thread(target=run, daemon=True).start()
    while True:
        item = q.get()
        if item is stop:
            return
        yield item
