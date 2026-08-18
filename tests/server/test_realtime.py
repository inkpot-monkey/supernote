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
stock Private Cloud sync), which completed its app-data sync with no failure banners,
and the frame capture taken from that device in
``.scratch/realtime-frame-capture-2026-08-18.md``.

Raw protocol literals (``"40"``, ``"2"``, ``"41"``) appear throughout. The style guide's
"No Wire Token Leaks" rule targets callers and domain-level tests; this suite *is* the
protocol conformance suite, and the literals are the contract under test.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator

import pytest
from aiohttp import (
    ClientWebSocketResponse,
    WSCloseCode,
    WSMsgType,
    WSServerHandshakeError,
)
from aiohttp.test_utils import TestClient, make_mocked_request

from supernote.server import realtime
from supernote.server.realtime import is_device_protocol_request

# Frames must arrive promptly; without a bound a missing reply hangs the suite.
_RECV_TIMEOUT = 5.0

_DEVICE_QUERY = "EIO=3&transport=websocket&type=SN000X00000000"

# The capture recorded nine consecutive unanswered `ratta_ping`s on one 240s channel.
# Reproduce that count, which is what the channel has to survive; the device's real 25s
# cadence is not reproduced, since the handler holds no timers of its own
# (`heartbeat=None` on the WebSocketResponse) and so cannot be sensitive to it.
_OBSERVED_UNANSWERED_APP_EVENTS = 9

# Stands in for the handler's 85s silence bound, which no suite can wait out.
_SHORTENED_TIMEOUT_S = 0.2


@pytest.fixture
async def device_ws(
    client: TestClient,
    auth_headers: dict[str, str],
) -> AsyncIterator[ClientWebSocketResponse]:
    """A device channel past the handshake, with OPEN and the server CONNECT consumed."""
    token = auth_headers["x-access-token"]
    ws = await client.ws_connect(f"/socket.io/?{_DEVICE_QUERY}&token={token}")
    await ws.receive_str(timeout=_RECV_TIMEOUT)  # Engine.IO OPEN
    await ws.receive_str(timeout=_RECV_TIMEOUT)  # server-initiated Socket.IO CONNECT
    try:
        yield ws
    finally:
        await ws.close()


async def test_device_handshake_sends_open_then_server_initiated_connect(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    """An EIO=3 device gets the Engine.IO OPEN frame plus an unprompted Socket.IO CONNECT.

    The device will not proceed until the server initiates the default-namespace
    CONNECT, so ``40`` must be sent without the device asking for it.

    This test opens its own channel rather than taking the ``device_ws`` fixture: the
    two frames the fixture consumes are the ones under test here.
    """
    token = auth_headers["x-access-token"]
    ws = await client.ws_connect(f"/socket.io/?{_DEVICE_QUERY}&token={token}")
    try:
        open_frame = await ws.receive_str(timeout=_RECV_TIMEOUT)
        assert open_frame.startswith("0"), (
            f"expected Engine.IO OPEN, got {open_frame!r}"
        )
        payload = json.loads(open_frame[1:])
        # The sid is random per channel, so it is checked for presence and removed;
        # everything the device negotiates against is then asserted whole.
        assert payload.pop("sid")
        # `upgrades` is empty because the device is already on websocket; advertising
        # an upgrade would be a lie. The timings are what the device paces its ping by.
        assert payload == {
            "upgrades": [],
            "pingInterval": 25000,
            "pingTimeout": 60000,
        }

        assert await ws.receive_str(timeout=_RECV_TIMEOUT) == "40"
    finally:
        await ws.close()


async def test_device_client_ping_is_answered_with_pong(
    device_ws: ClientWebSocketResponse,
) -> None:
    """Engine.IO v3 reverses the heartbeat: the client pings and the server must pong.

    Under the v4 server this frame is an unknown packet and the connection is torn
    down, so the device's keepalive never completes.
    """
    await device_ws.send_str("2")
    assert await device_ws.receive_str(timeout=_RECV_TIMEOUT) == "3"


async def test_device_app_event_does_not_break_the_channel(
    device_ws: ClientWebSocketResponse,
) -> None:
    """An app-level event (the device's 'ratta_ping') leaves the channel usable.

    The device needs no server reply to its app events — only that the channel stays
    connected — so the keepalive must still work afterwards.
    """
    await device_ws.send_str('42["ratta_ping","{}"]')
    await device_ws.send_str("2")
    assert await device_ws.receive_str(timeout=_RECV_TIMEOUT) == "3"
    assert not device_ws.closed


async def test_channel_survives_repeated_heartbeat_cycles(
    device_ws: ClientWebSocketResponse,
) -> None:
    """The channel holds across many heartbeat cycles with app events going unanswered.

    This is the regression test for the churn that was once blamed on the unanswered
    ``ratta_ping``: the device was thought to treat it as liveness and hang up when no
    reply came. The frame capture falsified that — one 240s channel carried nine
    consecutive unanswered ``ratta_ping``s and closed only on the device's own
    schedule. Answering them is therefore *not* required, and this test pins the
    property that matters: repeated cycles neither close the channel nor desynchronise
    the pong stream.
    """
    for _ in range(_OBSERVED_UNANSWERED_APP_EVENTS):
        await device_ws.send_str("2")
        assert await device_ws.receive_str(timeout=_RECV_TIMEOUT) == "3"
        await device_ws.send_str('42["ratta_ping"]')

    assert not device_ws.closed
    # One more round trip after the last unanswered event: the channel is still live,
    # and the pongs still line up one-to-one with the pings rather than lagging behind.
    await device_ws.send_str("2")
    assert await device_ws.receive_str(timeout=_RECV_TIMEOUT) == "3"


async def test_device_disconnect_frame_draws_no_reply(
    device_ws: ClientWebSocketResponse,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``41`` (Socket.IO v2 DISCONNECT) is how every real channel ends, and needs no reply.

    Every close in the frame capture is the device sending ``41`` and then closing the
    websocket with code 1000 — a deliberate hang-up before its next sync, not a failure.
    The server must not answer it (a reply to a disconnect is a protocol error) and must
    record the close code, which is the only thing separating that clean hang-up from a
    connection that died.
    """
    caplog.set_level(logging.DEBUG, logger="supernote.server.realtime")

    await device_ws.send_str("41")
    # The pong to the ping that follows is the assertion: had the DISCONNECT drawn a
    # reply of its own, that reply — not "3" — would arrive here.
    await device_ws.send_str("2")
    assert await device_ws.receive_str(timeout=_RECV_TIMEOUT) == "3"

    await device_ws.close(code=WSCloseCode.OK)

    closed = await _await_log(caplog, "channel closed")
    assert "close_code=1000" in closed, closed


async def test_device_close_frame_ends_the_channel(
    device_ws: ClientWebSocketResponse,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``1`` (Engine.IO CLOSE) tears the transport down rather than being ignored.

    An ignored CLOSE is not fatal — the device drops the TCP connection immediately
    afterwards — but it leaves the handler blocked on a socket the client has already
    finished with until the silence bound lapses. Honouring it closes at once.
    """
    caplog.set_level(logging.DEBUG, logger="supernote.server.realtime")

    await device_ws.send_str("1")

    msg = await device_ws.receive(timeout=_RECV_TIMEOUT)
    assert msg.type is WSMsgType.CLOSE, msg
    closed = await _await_log(caplog, "channel closed")
    assert "frames_in=1" in closed, closed


async def test_silent_device_is_closed_once_the_ping_timeout_lapses(
    client: TestClient,
    auth_headers: dict[str, str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device that stops speaking is hung up on, not held open indefinitely.

    The handshake advertises ``pingTimeout``, and a v3 server is expected to enforce it
    at ``pingInterval + pingTimeout``. Nothing in the frame capture exercises this — every
    channel there ended in a clean ``41`` and a 1000 close — but a device that vanishes
    without a TCP close is the ordinary end of a mobile connection, and each one that did
    would hold a handler and a socket for the life of the process.

    The real bound is 85s, which no suite can wait out, so it is shortened here; what is
    under test is that a bound exists and that lapsing it closes the channel.
    """
    caplog.set_level(logging.DEBUG, logger="supernote.server.realtime")
    monkeypatch.setattr(realtime, "_RECEIVE_TIMEOUT_S", _SHORTENED_TIMEOUT_S)

    token = auth_headers["x-access-token"]
    ws = await client.ws_connect(f"/socket.io/?{_DEVICE_QUERY}&token={token}")
    try:
        await ws.receive_str(timeout=_RECV_TIMEOUT)  # Engine.IO OPEN
        await ws.receive_str(
            timeout=_RECV_TIMEOUT
        )  # server-initiated Socket.IO CONNECT

        # Say nothing at all: no ping, no close frame — exactly what a device that has
        # dropped off the network looks like from here.
        msg = await ws.receive(timeout=_RECV_TIMEOUT)
        assert msg.type is WSMsgType.CLOSE, msg
        assert msg.data == WSCloseCode.GOING_AWAY
    finally:
        await ws.close()

    assert "timed out" in await _await_log(caplog, "timed out")


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
    device_ws: ClientWebSocketResponse,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every inbound frame is logged, so a live channel can be read off the journal.

    The device's side of this protocol can only be observed on real hardware, and a
    channel that logged nothing between 'open' and 'closed' left every question about
    it — cadence, event surface, who hung up — unanswerable from a deployment.
    """
    caplog.set_level(logging.DEBUG, logger="supernote.server.realtime")

    await device_ws.send_str("2")
    assert await device_ws.receive_str(timeout=_RECV_TIMEOUT) == "3"
    await device_ws.send_str('42["ratta_ping"]')
    # Round-trip a second ping so the app event is known to have been consumed.
    await device_ws.send_str("2")
    assert await device_ws.receive_str(timeout=_RECV_TIMEOUT) == "3"

    frames = [r.getMessage() for r in caplog.records if "frame" in r.getMessage()]
    assert any('"2"' in m or "'2'" in m for m in frames), frames
    assert any("ratta_ping" in m for m in frames), frames


async def test_channel_close_logs_the_close_code(
    device_ws: ClientWebSocketResponse,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The close log carries the websocket close code and the inbound frame count.

    Distinguishing a clean client hang-up from a dropped connection is the whole
    question when a device reopens its channel, and only the close code answers it.
    """
    caplog.set_level(logging.DEBUG, logger="supernote.server.realtime")

    await device_ws.send_str("2")
    assert await device_ws.receive_str(timeout=_RECV_TIMEOUT) == "3"
    await device_ws.close(code=WSCloseCode.GOING_AWAY)

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
