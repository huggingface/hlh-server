from __future__ import annotations

import os
import time

import httpx


DEFAULT_POPCORN_API_URL = "https://site--bot--dxfjds728w5v.code.run"
_valid_ids: dict[str, float] = {}


def _cache_ttl() -> int:
    return int(os.environ.get("POPCORN_AUTH_TTL", "300"))


async def validate_cli_id(cli_id: str) -> bool:
    api_url = os.environ.get("POPCORN_API_URL", DEFAULT_POPCORN_API_URL)
    now = time.monotonic()
    expiry = _valid_ids.get(cli_id)
    if expiry is not None and now < expiry:
        return True
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{api_url}/user/submissions",
                params={"limit": "1"},
                headers={"X-Popcorn-Cli-Id": cli_id},
            )
        if response.status_code == 200:
            _valid_ids[cli_id] = now + _cache_ttl()
            return True
        return False
    except Exception:
        return False
