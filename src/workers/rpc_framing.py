"""Length-prefixed message framing over a stream socket, used by the one
remaining Unix-socket RPC bridge under `workers/` — global_track_rpc.py's
GlobalTrackManager bridge — and any future one. A second such bridge
(gpu_worker_rpc.py, for GPUInferenceWorker) existed during the Celery
migration and was deleted once GPU inference moved into its own Celery
workers directly (docs/LSO67_FOLLOWUP_QUEUE_DESIGN.md).

Extracted here rather than left in global_track_rpc.py so a second, unrelated
bridge doesn't need to import a GlobalTrackManager-specific module's private
names to get the wire format — this is the only part of that module with no
domain coupling.
"""

import socket
import struct

HEADER = struct.Struct("!I")  # 4-byte big-endian length prefix


def send_framed(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(HEADER.pack(len(payload)) + payload)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed before expected bytes arrived")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_framed(sock: socket.socket) -> bytes:
    (length,) = HEADER.unpack(recv_exact(sock, HEADER.size))
    return recv_exact(sock, length)
