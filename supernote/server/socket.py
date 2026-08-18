"""Socket.IO server manager and event handling for Supernote server.

This is the **modern** channel: ``python-socketio`` 5.x over ``python-engineio`` 4.x,
which speak Socket.IO v5 / Engine.IO v4 and nothing older. It serves this project's own
client (:mod:`supernote.client.socket`).

Supernote *devices* are not v4 clients — they connect with ``EIO=3`` (Engine.IO v3 /
Socket.IO v2) and are refused here at version negotiation. Their channel is served
separately by :mod:`supernote.server.realtime`; see that module for the wire format.
"""

import inspect
import logging
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs

import engineio
import socketio
from aiohttp import web

from supernote.models.socket import (
    SocketHandshakeParams,
    SocketIoClientMessage,
    SocketIoEvent,
    SocketMessageData,
)
from supernote.server.config import ServerConfig
from supernote.server.socket_auth import (
    verify_handshake_signature,
    verify_handshake_token,
)

logger = logging.getLogger(__name__)


def _supported_server_options() -> frozenset[str]:
    """Collect every option name the Socket.IO / Engine.IO server pair really accepts."""

    def _params(func: Any) -> set[str]:
        return {
            name
            for name, param in inspect.signature(func).parameters.items()
            if name != "self"
            and param.kind
            not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        }

    return frozenset(
        _params(socketio.AsyncServer.__init__)
        | _params(engineio.AsyncServer.__init__)
        # socketio pops this one and forwards it as engineio's `logger`.
        | {"engineio_logger"}
    )


def build_async_server(**options: Any) -> socketio.AsyncServer:
    """Construct an ``AsyncServer``, rejecting options neither library implements.

    Both ``socketio.AsyncServer.__init__`` and ``engineio.AsyncServer.__init__`` end in
    ``**kwargs``, and socketio forwards everything it does not recognise straight to
    engineio, which drops the remainder on the floor. An unknown option therefore
    produces no error, no warning, and a clean startup — the server simply does not do
    the thing the option names.

    That failure mode is not hypothetical: this module long passed ``allow_eio3=True``,
    which is an option of the *JavaScript* socket.io server (``allowEIO3``) and has never
    existed in either Python library. The code advertised Engine.IO v3 support that was
    never present. Validate explicitly so a typo or a ported-from-JS option fails loudly.
    """
    if unknown := sorted(set(options) - _supported_server_options()):
        raise TypeError(
            f"unsupported Socket.IO server option(s): {', '.join(unknown)}. "
            "Neither socketio.AsyncServer nor engineio.AsyncServer accepts these; "
            "they would be silently ignored."
        )
    return socketio.AsyncServer(**options)


def _extract_handshake_params(environ: dict) -> SocketHandshakeParams:
    """Extract handshake authentication parameters from the WSGI/ASGI connection environment.

    Socket.IO passes the request environment dictionary to the connection handler,
    containing the raw QUERY_STRING sent during transport establishment.
    """
    query_string = environ.get("QUERY_STRING", "")
    parsed_query = parse_qs(query_string)

    token = parsed_query.get("token", [""])[0]
    conn_type = parsed_query.get("type", ["file"])[0]
    random_val = parsed_query.get("random", [""])[0]
    sign = parsed_query.get("sign", [""])[0]

    return SocketHandshakeParams(
        token=token,
        type=conn_type,
        random=random_val,
        sign=sign,
    )


class SocketIOServerManager:
    """Manages the Socket.IO server lifecycle, authentication, session rooms, and message delivery."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self._sio = build_async_server(
            async_mode="aiohttp",
            cors_allowed_origins="*",
            logger=False,
            engineio_logger=False,
        )
        self._sid_to_user: dict[str, str] = {}
        self._setup_handlers()

    @property
    def sio(self) -> socketio.AsyncServer:
        """Access the underlying AsyncServer instance."""
        return self._sio

    def attach_to_app(self, app: web.Application) -> None:
        """Attach the Socket.IO server instance to an aiohttp web Application."""
        self._sio.attach(app)
        app["socketio_manager"] = self

    async def _send_error(self, sid: str, status: HTTPStatus, message: str) -> None:
        """Send a SocketMessageData error response to a specific session ID."""
        err_data = SocketMessageData(
            code=str(status.value),
            msg=message,
        )
        await self._sio.emit(
            SocketIoEvent.SERVER_MESSAGE.value,
            err_data.to_json(),
            to=sid,
        )

    def _setup_handlers(self) -> None:
        @self._sio.event
        async def connect(sid: str, environ: dict) -> bool:
            params = _extract_handshake_params(environ)
            secret_key = self.config.auth.secret_key

            # Verify handshake signature
            if not verify_handshake_signature(params):
                logger.error(
                    "Socket.IO connect rejected: invalid signature (sid=%s)", sid
                )
                await self._send_error(
                    sid, HTTPStatus.FORBIDDEN, "sign verification failed"
                )
                return False

            # Verify JWT authentication token
            user_id = verify_handshake_token(params.token, secret_key)
            if not user_id:
                logger.error("Socket.IO connect rejected: invalid token (sid=%s)", sid)
                await self._send_error(
                    sid, HTTPStatus.FORBIDDEN, "token verification failed"
                )
                return False

            # Associate session ID with user_id and join user room
            self._sid_to_user[sid] = user_id
            user_room = f"user_{user_id}"
            await self._sio.enter_room(sid, user_room)
            logger.info(
                "Socket.IO connection established for user=%s (sid=%s, room=%s)",
                user_id,
                sid,
                user_room,
            )
            return True

        @self._sio.event
        async def disconnect(sid: str) -> None:
            user_id = self._sid_to_user.pop(sid, None)
            logger.info("Socket.IO disconnected for user=%s (sid=%s)", user_id, sid)

        @self._sio.on(SocketIoEvent.CLIENT_MESSAGE.value)
        async def on_client_message(sid: str, data: str) -> None:
            user_id = self._sid_to_user.get(sid)
            logger.debug(
                "Received ClientMessage from user=%s (sid=%s): %s", user_id, sid, data
            )

            if data == SocketIoClientMessage.STATUS.value:
                # Heartbeat check reply
                await self._sio.emit(
                    SocketIoEvent.SERVER_MESSAGE.value,
                    "true",
                    to=sid,
                )
            elif data == SocketIoClientMessage.RECEIVED.value:
                # Client delivery acknowledgment
                logger.debug("Client ACK received for user=%s (sid=%s)", user_id, sid)

        @self._sio.on(SocketIoEvent.RATTA_PING.value)
        async def on_ratta_ping(sid: str, data: str) -> None:
            user_id = self._sid_to_user.get(sid)
            logger.debug(
                "Received ratta_ping from user=%s (sid=%s): %s", user_id, sid, data
            )
            await self._sio.emit(
                SocketIoEvent.RATTA_PING.value,
                SocketIoClientMessage.RECEIVED.value,
                to=sid,
            )

    async def send_message(self, user_id: str, message: SocketMessageData) -> None:
        """Send a SocketMessageData event to all connected sessions for a given user.

        Args:
            user_id: Target user email or identifier.
            message: SocketMessageData payload to deliver.
        """
        user_room = f"user_{user_id}"
        payload_str = message.to_json()
        logger.debug("Emitting ServerMessage to room=%s: %s", user_room, payload_str)
        await self._sio.emit(
            SocketIoEvent.SERVER_MESSAGE.value,
            payload_str,
            room=user_room,
        )


def setup_socketio(app: web.Application, config: ServerConfig) -> SocketIOServerManager:
    """Initialize and attach Socket.IO server manager to an aiohttp Application.

    Args:
        app: The aiohttp web Application.
        config: The server configuration instance.

    Returns:
        The configured SocketIOServerManager instance.
    """
    manager = SocketIOServerManager(config)
    manager.attach_to_app(app)
    return manager
