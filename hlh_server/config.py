from __future__ import annotations

import os
from dataclasses import dataclass


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _profile_runner_env() -> str:
    raw = os.environ.get("HLH_PROFILE_RUNNER", "service").strip().lower()
    if raw in {"service", "github"}:
        return raw
    raise ValueError(f"invalid HLH_PROFILE_RUNNER: {raw!r}. expected one of: service, github")


@dataclass(frozen=True)
class Settings:
    service_host: str = os.environ.get("HLH_SERVICE_HOST", "127.0.0.1")
    service_port: int = _int_env("HLH_SERVICE_PORT", 8788)
    profile_runner: str = _profile_runner_env()
    profile_workflow_file: str = os.environ.get("HLH_PROFILE_WORKFLOW_FILE", "main.yml")
    profile_workflow_ref: str = os.environ.get("HLH_PROFILE_WORKFLOW_REF", "main")
    profile_poll_seconds: int = _int_env("HLH_PROFILE_POLL_SECONDS", 5)
    profile_timeout_seconds: int = _int_env("HLH_PROFILE_TIMEOUT_SECONDS", 600)
    upload_top_submission_bundle: bool = _bool_env("HLH_UPLOAD_TOP_SUBMISSION_BUNDLE", False)
    submit_bundle_enabled: bool = _bool_env("HLH_SUBMIT_BUNDLE_ENABLED", True)
    submit_bundle_bucket_template: str = os.environ.get(
        "HLH_SUBMIT_BUNDLE_BUCKET_TEMPLATE",
        "{hf_user}/humanitys-last-hackathon",
    )
    submit_bundle_hf_token_env: str = os.environ.get(
        "HLH_SUBMIT_BUNDLE_HF_TOKEN_ENV",
        "HF_TOKEN",
    )


def get_settings() -> Settings:
    return Settings()
