"""TCP frame I/O for the WorkflowSim <-> PPO sidecar bridge.

Wire format: 4-byte big-endian length prefix followed by a UTF-8 JSON body.
The Java client (RLBridge.java) emits frames shaped like
``{"type": "OBSERVE", ...}``; this module reads/writes them as plain ``dict``.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Optional


HEADER = struct.Struct(">I")


class FrameError(RuntimeError):
    """Raised when the peer disconnects or sends a malformed frame."""


def recv_frame(sock: socket.socket) -> Optional[dict]:
    """Read a single length-prefixed JSON frame.

    Returns ``None`` if the peer closed the connection cleanly. Raises
    :class:`FrameError` on protocol violations.
    """
    header = _recv_exact(sock, HEADER.size)
    if header is None:
        return None
    (length,) = HEADER.unpack(header)
    if length < 0 or length > 16 * 1024 * 1024:
        raise FrameError(f"invalid frame length {length}")
    body = _recv_exact(sock, length)
    if body is None:
        raise FrameError("short read on frame body")
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise FrameError(f"bad JSON frame: {exc}") from exc


def send_frame(sock: socket.socket, frame: dict) -> None:
    body = json.dumps(frame, separators=(",", ":")).encode("utf-8")
    sock.sendall(HEADER.pack(len(body)) + body)


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None if not buf else bytes(buf)
        buf.extend(chunk)
    return bytes(buf)
