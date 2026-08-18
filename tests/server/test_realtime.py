"""Tests for the device realtime channel (Engine.IO v3 / Socket.IO v2).

A real Supernote device opens its app-data channel with::

    GET /socket.io/?sign=..&random=..&EIO=3&transport=websocket&type=<deviceId>&token=<jwt>

i.e. **Engine.IO protocol v3 / Socket.IO protocol v2**, going straight to
``transport=websocket`` with no polling-first upgrade. ``python-socketio`` 5.x /
``python-engineio`` 4.x speak Engine.IO v4 / Socket.IO v5 only and reject ``EIO=3`` at
version negotiation, so the device needs a dedicated legacy handler dual-stacked
alongside the modern server used by this project's own Python client.

The two protocols differ in ways that matter on the wire:

* **CONNECT direction** — in Socket.IO v2 the *server* initiates the namespace CONNECT
  (``40``) and the device idles until it arrives; in v5 the client sends it first.
* **Heartbeat direction** — in Engine.IO v3 the *client* pings (``2``) and the server
  pongs (``3``); v4 reverses this.

These frame expectations are not read off the spec alone: they match a hand-rolled
implementation previously validated against real hardware (a Supernote Nomad A6 X2 on
stock Private Cloud sync), which completed its app-data sync with no failure banners.
"""

import asyncio
import json
import logging

import pytest
from aiohttp import WSCloseCode, WSServerHandshakeError
from aiohttp.test_utils import TestClient, make_mocked_request

from supernote.server.realtime import is_device_protocol_request

# Frames must arrive promptly; without a bound a missing reply hangs the suite.
_RECV_TIMEOUT = 5.0

_DEVICE_QUERY = "EIO=3&transport=websocket&type=SN000X00000000"


async def test_device_handshake_sends_open_then_server_initiated_connect(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    """An EIO=3 device gets the Engine.IO OPEN frame plus an unprompted Socket.IO CONNECT.

    The device will not proceed until the server initiates the default-namespace
    CONNECT, so ``40`` must be sent without the device asking for it.
    """
    token = auth_headers["x-access-token"]
    ws = await client.ws_connect(f"/socket.io/?{_DEVICE_QUERY}&token={token}")
    try:
        open_frame = await ws.receive_str(timeout=_RECV_TIMEOUT)
        assert open_frame.startswith("0"), (
            f"expected Engine.IO OPEN, got {open_frame!r}"
        )
        payload = json.loads(open_frame[1:])
        assert payload["sid"]
        # The device is already on websocket; advertising upgrades would be a lie.
        assert payload["upgrades"] == []
        assert payload["pingInterval"] == 25000
        assert payload["pingTimeout"] == 60000

        assert await ws.receive_str(timeout=_RECV_TIMEOUT) == "40"
    finally:
        await ws.close()


async def test_device_client_ping_is_answered_with_pong(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    """Engine.IO v3 reverses the heartbeat: the client pings and the server must pong.

    Under the v4 server this frame is an unknown packet and the connection is torn
    down, so the device's keepalive never completes.
    """
    token = auth_headers["x-access-token"]
    ws = await client.ws_connect(f"/socket.io/?{_DEVICE_QUERY}&token={token}")
    try:
        await ws.receive_str(timeout=_RECV_TIMEOUT)  # OPEN
        await ws.receive_str(timeout=_RECV_TIMEOUT)  # server CONNECT

        await ws.send_str("2")
        assert await ws.receive_str(timeout=_RECV_TIMEOUT) == "3"
    finally:
        await ws.close()


async def test_device_client_namespace_connect_is_echoed(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    """A client-initiated namespace CONNECT is echoed so the client fires 'connect'."""
    token = auth_headers["x-access-token"]
    ws = await client.ws_connect(f"/socket.io/?{_DEVICE_QUERY}&token={token}")
    try:
        await ws.receive_str(timeout=_RECV_TIMEOUT)  # OPEN
        await ws.receive_str(timeout=_RECV_TIMEOUT)  # server CONNECT

        await ws.send_str("40/device,")
        assert await ws.receive_str(timeout=_RECV_TIMEOUT) == "40/device,"
    finally:
        await ws.close()


async def test_device_app_event_does_not_break_the_channel(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    """An app-level event (the device's 'ratta_ping') leaves the channel usable.

    The device needs no server reply to its app events — only that the channel stays
    connected — so the keepalive must still work afterwards.
    """
    token = auth_headers["x-access-token"]
    ws = await client.ws_connect(f"/socket.io/?{_DEVICE_QUERY}&token={token}")
    try:
        await ws.receive_str(timeout=_RECV_TIMEOUT)  # OPEN
        await ws.receive_str(timeout=_RECV_TIMEOUT)  # server CONNECT

        await ws.send_str('42["ratta_ping","{}"]')
        await ws.send_str("2")
        assert await ws.receive_str(timeout=_RECV_TIMEOUT) == "3"
        assert not ws.closed
    finally:
        await ws.close()


@pytest.mark.parametrize(
    ("path", "query", "is_device"),
    [
        ("/socket.io/", "EIO=3&transport=websocket", True),
        ("/socket.io", "EIO=3&transport=websocket", True),
        # The modern client declares EIO=4 and belongs to python-socketio.
        ("/socket.io/", "EIO=4&transport=websocket", False),
        ("/socket.io/", "transport=websocket", False),
        # A near-miss path must not be handed to an entirely different protocol stack.
        ("/socket.iox", "EIO=3&transport=websocket", False),
        ("/socket.io.bak", "EIO=3&transport=websocket", False),
    ],
)
def test_device_protocol_request_matches_only_the_socketio_endpoint(
    path: str,
    query: str,
    is_device: bool,
) -> None:
    """Dispatch is on the endpoint and the declared version, not on a path prefix.

    This predicate decides which of two incompatible protocol implementations serves a
    request, so a prefix match is too loose: it would route paths that merely start with
    ``/socket.io`` into the legacy handler.
    """
    request = make_mocked_request("GET", f"{path}?{query}")
    assert is_device_protocol_request(request) is is_device


async def test_device_channel_rejects_missing_token(client: TestClient) -> None:
    """The channel is authenticated; an upgrade with no token is refused."""
    resp = await client.get(
        f"/socket.io/?{_DEVICE_QUERY}",
        headers={
            "Upgrade": "websocket",
            "Connection": "Upgrade",
            "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
            "Sec-WebSocket-Version": "13",
        },
    )
    assert resp.status == 401


async def test_device_channel_rejects_invalid_token(client: TestClient) -> None:
    """A bogus token is refused at the handshake, before the websocket opens."""
    with pytest.raises(WSServerHandshakeError) as exc:
        await client.ws_connect(f"/socket.io/?{_DEVICE_QUERY}&token=not-a-real-token")
    assert exc.value.status == 401


async def test_device_channel_requires_websocket_transport(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    """A plain GET is rejected — the device only ever uses transport=websocket."""
    token = auth_headers["x-access-token"]
    resp = await client.get(f"/socket.io/?EIO=3&transport=polling&token={token}")
    assert resp.status == 400


async def test_device_frames_are_logged_for_diagnosis(
    client: TestClient,
    auth_headers: dict[str, str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every inbound frame is logged, so a live channel can be read off the journal.

    The device's side of this protocol can only be observed on real hardware, and a
    channel that logged nothing between 'open' and 'closed' left every question about
    it — cadence, event surface, who hung up — unanswerable from a deployment.
    """
    caplog.set_level(logging.DEBUG, logger="supernote.server.realtime")
    token = auth_headers["x-access-token"]
    ws = await client.ws_connect(f"/socket.io/?{_DEVICE_QUERY}&token={token}")
    try:
        await ws.receive_str(timeout=_RECV_TIMEOUT)  # OPEN
        await ws.receive_str(timeout=_RECV_TIMEOUT)  # server CONNECT

        await ws.send_str("2")
        assert await ws.receive_str(timeout=_RECV_TIMEOUT) == "3"
        await ws.send_str('42["ratta_ping"]')
        # Round-trip a second ping so the app event is known to have been consumed.
        await ws.send_str("2")
        assert await ws.receive_str(timeout=_RECV_TIMEOUT) == "3"
    finally:
        await ws.close()

    frames = [r.getMessage() for r in caplog.records if "frame" in r.getMessage()]
    assert any('"2"' in m or "'2'" in m for m in frames), frames
    assert any("ratta_ping" in m for m in frames), frames


async def test_channel_close_logs_the_close_code(
    client: TestClient,
    auth_headers: dict[str, str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The close log carries the websocket close code and the inbound frame count.

    Distinguishing a clean client hang-up from a dropped connection is the whole
    question when a device reopens its channel, and only the close code answers it.
    """
    caplog.set_level(logging.DEBUG, logger="supernote.server.realtime")
    token = auth_headers["x-access-token"]
    ws = await client.ws_connect(f"/socket.io/?{_DEVICE_QUERY}&token={token}")
    await ws.receive_str(timeout=_RECV_TIMEOUT)  # OPEN
    await ws.receive_str(timeout=_RECV_TIMEOUT)  # server CONNECT
    await ws.send_str("2")
    assert await ws.receive_str(timeout=_RECV_TIMEOUT) == "3"
    await ws.close(code=WSCloseCode.GOING_AWAY)

    closed = await _await_log(caplog, "channel closed")
    assert "close_code=1001" in closed, closed
    assert "frames_in=1" in closed, closed


async def _await_log(caplog: pytest.LogCaptureFixture, needle: str) -> str:
    """Wait for a log line containing ``needle``; the handler finishes after the close."""
    for _ in range(int(_RECV_TIMEOUT / 0.05)):
        for record in caplog.records:
            message = record.getMessage()
            if needle in message:
                return message
        await asyncio.sleep(0.05)
    raise AssertionError(f"no log line containing {needle!r}: {caplog.text}")
