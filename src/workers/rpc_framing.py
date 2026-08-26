"""Length-prefixed message framing over a stream socket, shared by every
Unix-socket RPC bridge under `workers/` (gpu_rpc.py's GlobalTrackManager
bridge, gpu_worker_rpc.py's GPUInferenceWorker bridge, and any future one).

Extracted here rather than left in gpu_rpc.py so a second, unrelated bridge
doesn't need to import a GlobalTrackManager-specific module's private names
to get the wire format — this is the only part of that module with no
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
