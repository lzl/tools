from __future__ import annotations

import getpass
import os
from pathlib import Path


def load_dotenv_if_present(start_dir: Path | None = None) -> Path | None:
    env_path = _find_dotenv(start_dir or Path.cwd())
    if env_path is None:
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[7:].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ[key] = value

    return env_path


def _find_dotenv(start_dir: Path) -> Path | None:
    current = start_dir.resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_api_credentials() -> tuple[int, str]:
    api_id_text = os.environ.get("TELEGRAM_API_ID")
    api_hash = os.environ.get("TELEGRAM_API_HASH")

    if not api_id_text:
        raise RuntimeError("TELEGRAM_API_ID is required")
    if not api_hash:
        raise RuntimeError("TELEGRAM_API_HASH is required")

    try:
        api_id = int(api_id_text)
    except ValueError as exc:
        raise RuntimeError("TELEGRAM_API_ID must be an integer") from exc

    return api_id, api_hash


def load_download_session() -> str:
    session = os.environ.get("TELEGRAM_STRING_SESSION")
    if not session:
        raise RuntimeError(
            "TELEGRAM_STRING_SESSION is required for download runs. "
            "Generate one with `uv run telegram-media generate-session`."
        )
    return session


def prompt_phone() -> str:
    return input("Telegram phone number: ").strip()


def prompt_code() -> str:
    return input("Telegram login code: ").strip()


def prompt_password() -> str:
    return getpass.getpass("Telegram 2FA password: ")
