from __future__ import annotations

from typing import Callable
from pathlib import Path

from telethon import TelegramClient
from telethon.errors import (
    ChannelPrivateError,
    RPCError,
    SessionPasswordNeededError,
    UsernameInvalidError,
    UsernameNotOccupiedError,
)
from telethon.sessions import StringSession
from telethon.tl.custom.message import Message
from telethon.tl.types import MessageMediaDocument, MessageMediaPhoto

from telegram_media.downloader import ChannelMessage
from telegram_media.session import (
    load_api_credentials,
    load_download_session,
    prompt_code,
    prompt_password,
    prompt_phone,
)


def _resolve_extension(message: Message, default: str) -> str:
    file_wrapper = getattr(message, "file", None)
    extension = getattr(file_wrapper, "ext", None)
    if extension:
        return extension if extension.startswith(".") else f".{extension}"
    return default


class TelethonMediaApi:
    def __init__(self, client: TelegramClient) -> None:
        self.client = client

    async def iter_channel_messages(
        self,
        channel_id: int | str,
        *,
        min_message_id: int,
    ):
        try:
            entity = await self.client.get_entity(channel_id)
            async for raw_message in self.client.iter_messages(
                entity,
                min_id=min_message_id,
                reverse=True,
            ):
                yield self._to_channel_message(channel_id, raw_message)
        except (ChannelPrivateError, UsernameInvalidError, UsernameNotOccupiedError) as exc:
            raise RuntimeError(
                "The authenticated account cannot access this channel. "
                "Check TELEGRAM_STRING_SESSION and channel visibility."
            ) from exc
        except ValueError as exc:
            raise RuntimeError(f"Could not resolve Telegram channel {channel_id!r}.") from exc
        except RPCError as exc:
            raise RuntimeError(f"Telegram request failed: {exc}") from exc

    async def download_media(
        self,
        message: ChannelMessage,
        destination: Path,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> str | bytes | None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        return await self.client.download_media(
            message.source,
            file=destination,
            progress_callback=progress_callback,
        )

    def _to_channel_message(self, channel_id: int | str, message: Message) -> ChannelMessage:
        if isinstance(message.media, MessageMediaPhoto) and message.photo is not None:
            return ChannelMessage(
                channel_id=channel_id,
                message_id=message.id,
                media_type="image",
                file_id=str(message.photo.id),
                extension=_resolve_extension(message, ".jpg"),
                source=message,
            )

        if isinstance(message.media, MessageMediaDocument) and message.document is not None:
            mime_type = message.document.mime_type or ""
            if mime_type.startswith("video/"):
                return ChannelMessage(
                    channel_id=channel_id,
                    message_id=message.id,
                    media_type="video",
                    file_id=str(message.document.id),
                    extension=_resolve_extension(message, ".mp4"),
                    source=message,
                )
            if mime_type.startswith("image/"):
                return ChannelMessage(
                    channel_id=channel_id,
                    message_id=message.id,
                    media_type="image",
                    file_id=str(message.document.id),
                    extension=_resolve_extension(message, ".jpg"),
                    source=message,
                )

        return ChannelMessage(
            channel_id=channel_id,
            message_id=message.id,
            media_type=None,
            file_id=None,
            extension=None,
            source=message,
        )


async def create_download_client() -> TelegramClient:
    api_id, api_hash = load_api_credentials()
    session = load_download_session()
    client = TelegramClient(StringSession(session), api_id, api_hash)
    await client.connect()
    if not await client.is_user_authorized():
        await client.disconnect()
        raise RuntimeError(
            "TELEGRAM_STRING_SESSION is invalid or expired. "
            "Generate a fresh one with `uv run telegram-media generate-session`."
        )
    return client


async def generate_string_session(phone: str | None = None) -> str:
    api_id, api_hash = load_api_credentials()
    client = TelegramClient(StringSession(), api_id, api_hash)
    await client.connect()
    try:
        login_phone = phone or prompt_phone()
        await client.send_code_request(login_phone)
        try:
            await client.sign_in(phone=login_phone, code=prompt_code())
        except SessionPasswordNeededError:
            await client.sign_in(password=prompt_password())
        session = client.session.save()
    finally:
        await client.disconnect()

    if not session:
        raise RuntimeError("Telethon did not return a string session.")
    return session
