"""Realtime channel for the Supernote device's app-data sync (Engine.IO v3 / Socket.IO v2).

The device opens a realtime channel on every sync with::

    GET /socket.io/?sign=..&random=..&EIO=3&transport=websocket&type=<deviceId>&token=<jwt>

i.e. **Engine.IO protocol v3 / Socket.IO protocol v2**, going straight to
``transport=websocket`` with no polling-first upgrade.

``python-socketio`` 5.x / ``python-engineio`` 4.x — the pair serving
:mod:`supernote.server.socket` — implement Socket.IO v5 / Engine.IO v4 only. They refuse
``EIO=3`` at version negotiation with a 400 before authentication is ever considered, and
the v3 encoding paths have been removed outright, so the device cannot be served there at
any option setting. Pinning the libraries back to the 2020-era versions that do speak v3
would drag the whole codebase back to restore one channel. This module hand-rolls the
small amount of v3/v2 the device actually needs, over aiohttp's native
``WebSocketResponse``, and is dispatched to on the ``EIO`` query parameter so the modern
server is left untouched.

Wire format (text frames over the single websocket)::

    packet = <engineio_type><payload>
    engine.io types : 0 open  1 close  2 ping  3 pong  4 message  5 upgrade  6 noop
    a Socket.IO packet rides inside an Engine.IO MESSAGE(4):
        socket.io types : 0 CONNECT  1 DISCONNECT  2 EVENT  3 ACK  4 ERROR ...
    so on the wire:
        "0{...}"   Engine.IO OPEN     (handshake, server -> client)
        "40"       MESSAGE + CONNECT  (default-namespace connect)
        "2" / "3"  Engine.IO ping / pong

Two directional differences from v4 are what make a v4 server unusable here:

* **The server initiates CONNECT.** In Socket.IO v2 the server sends ``40`` unprompted
  and the client fires its ``connect`` event on receipt; v3+ flips this, with the client
  connecting first. The device idles until the server speaks.
* **The client drives the heartbeat.** In Engine.IO v3 the client sends PING (``2``) and
  the server replies PONG (``3``); v4 reverses it. A v4 server treats the device's PING
  as an unknown packet and tears the connection down.

Connecting is sufficient — the device needs no server-initiated application events to
complete its app-data sync. Established live against a Supernote Nomad A6 X2 on stock
Private Cloud sync, which completed with no failure banners.
"""

from __future__ import annotations

import json
import logging
import secrets

from aiohttp import WSCloseCode, WSMsgType, web

from supernote.server.services.user import UserService

logger = logging.getLogger(__name__)

# Engine.IO v3 handshake timings advertised to the device (milliseconds).
_PING_INTERVAL_MS = 25000
_PING_TIMEOUT_MS = 60000

#: Longest silence tolerated before the device is presumed gone. Engine.IO v3 servers
#: declare a client dead once ``pingInterval + pingTimeout`` passes with nothing
#: received, so the handshake's advertised timings set this rather than a value of
#: their own.
_RECEIVE_TIMEOUT_S = (_PING_INTERVAL_MS + _PING_TIMEOUT_MS) / 1000

#: Query-parameter value identifying an Engine.IO v3 client.
ENGINEIO_V3 = "3"


def is_device_protocol_request(request: web.Request) -> bool:
    """Whether this request speaks Engine.IO v3 and belongs to the device channel."""
    # Match the endpoint itself rather than a prefix: this predicate hands the request
    # to a different protocol stack, so a near-miss path like `/socket.iox` must not
    # qualify. python-socketio mounts at `/socket.io/`; the device omits no slash.
    return (
        request.path.rstrip("/") == "/socket.io"
        and request.query.get("EIO") == ENGINEIO_V3
    )


async def handle_device_socket(request: web.Request) -> web.StreamResponse:
    """Serve the device's Engine.IO v3 / Socket.IO v2 channel (connect-and-keepalive)."""
    if request.headers.get("Upgrade", "").lower() != "websocket":
        # The device only ever uses transport=websocket; the v3 polling payload
        # encoding is deliberately not implemented.
        return web.json_response({"error": "websocket transport required"}, status=400)

    # This channel authenticates with the JWT as a query parameter, not the
    # x-access-token header the REST middleware reads, so verify it here. Reject before
    # the upgrade so a bad token surfaces as an HTTP status rather than a dead socket.
    token = request.query.get("token")
    if not token:
        return web.json_response({"error": "token required"}, status=401)
    user_service: UserService = request.app["user_service"]
    session = await user_service.verify_token(token)
    if not session:
        return web.json_response({"error": "invalid token"}, status=401)

    # `heartbeat=None` because the device drives the keepalive itself; `receive_timeout`
    # because nothing else would. A device that vanishes without a TCP close -- the
    # normal end of a mobile connection -- would otherwise hold this handler and its
    # socket for as long as the process lives, while the handshake below advertises a
    # timeout the server never applies.
    ws = web.WebSocketResponse(heartbeat=None, receive_timeout=_RECEIVE_TIMEOUT_S)
    await ws.prepare(request)

    sid = secrets.token_hex(12)
    device = request.query.get("type")
    logger.info(
        "device realtime channel open: user=%s device=%s sid=%s",
        session.email,
        device,
        sid,
    )

    # Engine.IO OPEN handshake. `upgrades` is empty: the device is already on
    # websocket, so there is nothing to upgrade to.
    await ws.send_str(
        "0"
        + json.dumps(
            {
                "sid": sid,
                "upgrades": [],
                "pingInterval": _PING_INTERVAL_MS,
                "pingTimeout": _PING_TIMEOUT_MS,
            }
        )
    )

    # Server-initiated Socket.IO v2 CONNECT for the default namespace. The device
    # waits on this and will not proceed until it arrives.
    await ws.send_str("40")

    frames_in = 0
    try:
        async for msg in ws:
            if msg.type is not WSMsgType.TEXT:
                logger.debug("device realtime frame: sid=%s <%s>", sid, msg.type.name)
                if msg.type is WSMsgType.ERROR:
                    logger.warning(
                        "device realtime channel error sid=%s: %s", sid, ws.exception()
                    )
                break

            data = msg.data
            frames_in += 1
            # The device's side of this protocol can only be observed on hardware, so
            # log it: this is the only record of what a real device sends and when.
            logger.debug("device realtime frame: sid=%s %r", sid, _elide(data))
            if data[:1] == "2":  # Engine.IO PING -> PONG keepalive (v3 direction)
                await ws.send_str("3" + data[1:])
            elif data[:1] == "1":
                # The client is tearing the transport down; stop reading rather than
                # wait out `receive_timeout` for a socket that is already finished.
                break
            # Everything else is a Socket.IO packet inside an Engine.IO MESSAGE -- the
            # device's `ratta_ping` event and its closing DISCONNECT among them. None
            # of them takes a reply; the device needs only that the channel stays up.
    except TimeoutError:
        # Nothing for `pingInterval + pingTimeout`: the device is gone without saying so.
        logger.info(
            "device realtime channel timed out: sid=%s after %.0fs silent",
            sid,
            _RECEIVE_TIMEOUT_S,
        )
        await ws.close(code=WSCloseCode.GOING_AWAY)
    finally:
        # The close code separates a clean client hang-up -- the device opens a fresh
        # channel for each sync and drops the old one -- from a connection that died.
        logger.debug(
            "device realtime channel closed: sid=%s close_code=%s frames_in=%d",
            sid,
            ws.close_code,
            frames_in,
        )
    return ws


def _elide(data: str, limit: int = 240) -> str:
    """Bound a logged frame; app payloads are small but nothing guarantees it."""
    return data if len(data) <= limit else f"{data[:limit]}...(+{len(data) - limit})"
