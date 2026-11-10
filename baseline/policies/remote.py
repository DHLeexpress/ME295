"""Minimal client for openpi's WebsocketPolicyServer (same wire format as openpi_client, without its numpy<2 pin).

Messages are msgpack with numpy arrays packed as {__ndarray__, data, dtype, shape}; the server first sends its
metadata, then answers each observation dict with an output dict (e.g. {"actions": (H, D)}).
"""
from __future__ import annotations

import time

import msgpack
import numpy as np
import websockets.sync.client


def _pack(obj):
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"unsupported dtype {obj.dtype}")
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class RemotePolicy:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000, wait: float = 600.0):
        self.uri = f"ws://{host}:{port}"
        t_end = time.time() + wait
        while True:
            try:
                self.ws = websockets.sync.client.connect(self.uri, compression=None, max_size=None,
                                                         open_timeout=30)
                self.metadata = msgpack.unpackb(self.ws.recv(), object_hook=_unpack)
                break
            except (ConnectionRefusedError, OSError, TimeoutError):
                if time.time() > t_end:
                    raise
                time.sleep(5)
        self.n_calls, self.latency = 0, 0.0

    def infer(self, obs: dict) -> dict:
        t0 = time.time()
        self.ws.send(msgpack.packb(obs, default=_pack))
        resp = self.ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"policy server error:\n{resp}")
        self.n_calls += 1
        self.latency += time.time() - t0
        return msgpack.unpackb(resp, object_hook=_unpack)

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass
