"""Environment loading. Secrets live outside the repo in ~/.config/pred_el_prices/.env."""

import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path.home() / ".config" / "pred_el_prices" / ".env"


def entsoe_api_key() -> str:
    # utf-8-sig tolerates the BOM that Windows PowerShell writes by default;
    # real environment variables take precedence
    load_dotenv(ENV_PATH, encoding="utf-8-sig")
    key = os.environ.get("ENTSOE_API_KEY", "")
    if not key:
        raise RuntimeError(f"ENTSOE_API_KEY not set; expected it in {ENV_PATH}")
    return key


def energyforecast_token() -> str:
    load_dotenv(ENV_PATH, encoding="utf-8-sig")
    token = os.environ.get("ENERGYFORECAST_TOKEN", "")
    if not token:
        raise RuntimeError(f"ENERGYFORECAST_TOKEN not set; expected it in {ENV_PATH}")
    return token


def netztransparenz_credentials() -> tuple[str, str]:
    # the names carry the "TRANZ" spelling as registered in .env and GitHub secrets
    load_dotenv(ENV_PATH, encoding="utf-8-sig")
    cid = os.environ.get("NETZTRANZPARENZ_CLIENT_ID", "")
    secret = os.environ.get("NETZTRANZPARENZ_CLIENT_SECRET", "")
    if not (cid and secret):
        raise RuntimeError(
            f"NETZTRANZPARENZ_CLIENT_ID/_SECRET not set; expected them in {ENV_PATH}"
        )
    return cid, secret
