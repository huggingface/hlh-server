from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from hlh_server import auth
from hlh_server.artifact_store import HFBucketArtifactStore
from hlh_server.config import Settings, get_settings
from hlh_server.metal_profiler import profile_file


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    artifact_store = HFBucketArtifactStore(settings.submit_bundle_bucket_template)

    app = FastAPI(title="Humanitys Last Hackathon Server")
    app.state.settings = settings
    app.state.artifact_store = artifact_store

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/hlh/profile")
    async def hlh_profile(
        kernel: UploadFile = File(...),
        cli_id: str | None = Header(default=None, alias="X-Popcorn-Cli-Id"),
    ) -> dict[str, Any]:
        await _require_popcorn_cli_id(cli_id)
        kernel_content, kernel_filename = await _read_uploaded_kernel(kernel)
        try:
            if settings.profile_runner == "github":
                return await asyncio.to_thread(
                    _profile_via_github_workflow,
                    kernel_content,
                    kernel_filename,
                    settings,
                )
            return await _profile_on_service_host(kernel_content, kernel_filename)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/v1/hlh/submit")
    async def hlh_submit(
        leaderboard: str = Form(...),
        gpu: str = Form(...),
        kernel: UploadFile = File(...),
        cli_id: str | None = Header(default=None, alias="X-Popcorn-Cli-Id"),
    ) -> dict[str, Any]:
        await _require_popcorn_cli_id(cli_id)
        kernel_content, kernel_filename = await _read_uploaded_kernel(kernel)
        if not leaderboard.strip():
            raise HTTPException(status_code=422, detail="leaderboard must not be empty")
        if not gpu.strip():
            raise HTTPException(status_code=422, detail="gpu must not be empty")

        with tempfile.TemporaryDirectory(prefix="hlh-submit-") as tmp_dir:
            kernel_path = Path(tmp_dir) / Path(kernel_filename or "kernel.py").name
            kernel_path.write_text(kernel_content)
            output_path = Path(tmp_dir) / "popcorn-submit.json"
            command = [
                "popcorn",
                "submit",
                str(kernel_path),
                "--mode",
                "leaderboard",
                "--no-tui",
                "--output",
                str(output_path),
                "--leaderboard",
                leaderboard,
                "--gpu",
                gpu,
            ]
            try:
                completed = subprocess.run(command, capture_output=True, text=True, check=False)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=500, detail="popcorn command not found on server") from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            parsed_output = _load_popcorn_submit_output(output_path, completed.stdout, completed.stderr)

        submission_id = _resolve_hlh_submission_id(parsed_output) or uuid.uuid4().hex
        should_upload_submission_bundle = settings.upload_top_submission_bundle and completed.returncode == 0
        return {
            "tool": "popcorn",
            "mode": "leaderboard",
            "command": command,
            "leaderboard": leaderboard,
            "gpu": gpu,
            "exit_code": completed.returncode,
            "ok": completed.returncode == 0,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "parsed_output": parsed_output,
            "score_summary": _summarize_score_payload(parsed_output),
            "submission_id": submission_id,
            "should_upload_submission_bundle": should_upload_submission_bundle,
        }

    @app.post("/v1/hlh/submit/bundle")
    async def hlh_submit_bundle(
        submission_id: str = Form(...),
        kernel_filename: str = Form(...),
        bundle: UploadFile = File(...),
        cli_id: str | None = Header(default=None, alias="X-Popcorn-Cli-Id"),
    ) -> dict[str, Any]:
        await _require_popcorn_cli_id(cli_id)
        if not settings.submit_bundle_enabled:
            raise HTTPException(status_code=503, detail="bundle upload is disabled")
        if not submission_id.strip():
            raise HTTPException(status_code=422, detail="submission_id must not be empty")
        if not kernel_filename.strip():
            raise HTTPException(status_code=422, detail="kernel_filename must not be empty")

        token_env = settings.submit_bundle_hf_token_env
        hf_token = os.environ.get(token_env, "").strip()
        if not hf_token:
            raise HTTPException(status_code=500, detail=f"missing Hugging Face token in env var: {token_env}")

        context = artifact_store.create_context(
            submission_id=submission_id,
            hf_user="service",
            hf_token=hf_token,
        )

        with tempfile.TemporaryDirectory(prefix="hlh-submit-bundle-") as tmp_dir:
            tar_path = Path(tmp_dir) / "submission_bundle.tar.gz"
            tar_path.write_bytes(await bundle.read())
            extract_root = Path(tmp_dir) / "extracted"
            extract_root.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_path, mode="r:*") as archive:
                _safe_extract_tar(archive, extract_root)
            for extracted_path in sorted(path for path in extract_root.rglob("*") if path.is_file()):
                relative_path = extracted_path.relative_to(extract_root).as_posix()
                upload_target = artifact_store.make_upload_target(context, relative_path)
                artifact_store.upload_file(upload_target, extracted_path, hf_token)
        return {"received": True, "bucket_uri": context.bucket_uri}

    @app.exception_handler(KeyError)
    async def key_error_handler(_request: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": f"{exc} not found"})

    return app


async def _require_popcorn_cli_id(cli_id: str | None) -> str:
    value = str(cli_id or "").strip()
    if not value:
        raise HTTPException(status_code=401, detail="missing X-Popcorn-Cli-Id")
    if not await auth.validate_cli_id(value):
        raise HTTPException(status_code=401, detail="invalid Popcorn cli_id")
    return value


async def _read_uploaded_kernel(kernel: UploadFile) -> tuple[str, str]:
    content = (await kernel.read()).decode("utf-8", errors="replace")
    if not content.strip():
        raise HTTPException(status_code=422, detail="kernel_content must not be empty")
    filename = Path(kernel.filename or "kernel.py").name
    return content, filename


async def _profile_on_service_host(kernel_content: str, kernel_filename: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="hlh-profile-") as tmp_dir:
        kernel_path = Path(tmp_dir) / Path(kernel_filename or "kernel.py").name
        kernel_path.write_text(kernel_content)
        return await profile_file(str(kernel_path))


def _profile_via_github_workflow(
    kernel_content: str,
    kernel_filename: str,
    settings: Settings,
) -> dict[str, Any]:
    safe_filename = Path(kernel_filename or "kernel.py").name
    request_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="hlh-gh-profile-") as tmp_dir:
        content_path = Path(tmp_dir) / "kernel_content.txt"
        content_path.write_text(kernel_content)
        dispatch_command = [
            "gh",
            "workflow",
            "run",
            settings.profile_workflow_file,
            "--ref",
            settings.profile_workflow_ref,
            "-F",
            f"file_content=@{content_path}",
            "-f",
            f"filename={safe_filename}",
            "-f",
            f"request_id={request_id}",
        ]
        _run_process(dispatch_command, error_prefix="failed to dispatch profile workflow")

    run_id = _find_dispatched_run_id_by_request_id(
        workflow_file=settings.profile_workflow_file,
        branch=settings.profile_workflow_ref,
        request_id=request_id,
    )
    deadline = time.monotonic() + settings.profile_timeout_seconds
    run_url = ""
    while True:
        run_payload = _run_process_json(
            ["gh", "run", "view", str(run_id), "--json", "status,conclusion,url"],
            error_prefix=f"failed to query workflow run {run_id}",
        )
        status = str(run_payload.get("status", "")).strip().lower()
        run_url = str(run_payload.get("url", "")).strip()
        if status == "completed":
            conclusion = str(run_payload.get("conclusion", "")).strip().lower()
            if conclusion != "success":
                raise RuntimeError(
                    f"profile workflow run {run_id} did not succeed "
                    f"(conclusion={conclusion or 'unknown'}, url={run_url})"
                )
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"timed out waiting for workflow run {run_id} to complete")
        time.sleep(settings.profile_poll_seconds)

    logs = _run_process(
        ["gh", "run", "view", str(run_id), "--log"],
        error_prefix=f"failed to fetch logs for workflow run {run_id}",
    ).stdout
    profile_payload = _extract_profile_from_workflow_logs(logs)
    if not isinstance(profile_payload, dict):
        raise RuntimeError(f"profile workflow run {run_id} returned non-object JSON payload")
    return profile_payload


def _load_popcorn_submit_output(
    output_path: Path,
    stdout: str,
    stderr: str,
) -> dict[str, Any] | list[Any] | None:
    if output_path.exists():
        try:
            return json.loads(output_path.read_text())
        except json.JSONDecodeError:
            pass
    for source in (stdout, stderr):
        parsed = _maybe_parse_json_output(source)
        if parsed is not None:
            return parsed
    return None


def _maybe_parse_json_output(text: str) -> Any:
    for line in reversed(text.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _find_dispatched_run_id_by_request_id(
    *,
    workflow_file: str,
    branch: str,
    request_id: str,
) -> int:
    deadline = time.monotonic() + 60
    while True:
        runs = _run_process_json(
            [
                "gh",
                "run",
                "list",
                "--workflow",
                workflow_file,
                "--branch",
                branch,
                "--event",
                "workflow_dispatch",
                "--limit",
                "100",
                "--json",
                "databaseId,displayTitle,name",
            ],
            error_prefix="failed to list workflow runs",
        )
        if isinstance(runs, list):
            for run in runs:
                if not isinstance(run, dict):
                    continue
                marker = f"profile-{request_id}"
                run_title = str(run.get("displayTitle", "")).strip()
                run_name = str(run.get("name", "")).strip()
                if marker not in run_title and marker not in run_name:
                    continue
                run_id = run.get("databaseId")
                if isinstance(run_id, int):
                    return run_id
                if isinstance(run_id, str) and run_id.isdigit():
                    return int(run_id)
        if time.monotonic() >= deadline:
            raise RuntimeError(f"unable to discover dispatched workflow run id for request_id={request_id}")
        time.sleep(2)


def _extract_profile_from_workflow_logs(logs: str) -> dict[str, Any] | list[Any] | None:
    profiler_text = _extract_run_profiler_text(logs)
    for source in (profiler_text, logs):
        if not source:
            continue
        for chunk in reversed(_extract_balanced_json_chunks(source)):
            try:
                parsed_chunk = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed_chunk, dict):
                return parsed_chunk
        parsed_output = _maybe_parse_json_output(source)
        if isinstance(parsed_output, dict):
            return parsed_output
    return None


def _extract_balanced_json_chunks(text: str) -> list[str]:
    chunks: list[str] = []
    stack: list[str] = []
    start_idx: int | None = None
    in_string = False
    escaped = False
    for idx, ch in enumerate(text):
        if start_idx is None:
            if ch in "{[":
                start_idx = idx
                stack = [ch]
                in_string = False
                escaped = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            stack.append(ch)
            continue
        if ch == "}":
            if not stack or stack[-1] != "{":
                start_idx = None
                stack = []
                continue
            stack.pop()
        elif ch == "]":
            if not stack or stack[-1] != "[":
                start_idx = None
                stack = []
                continue
            stack.pop()
        if start_idx is not None and not stack:
            chunks.append(text[start_idx : idx + 1])
            start_idx = None
    return chunks


_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_LOG_TIMESTAMP_RE = re.compile(r"^\ufeff?\d{4}-\d{2}-\d{2}T[0-9:\.\+\-]+Z\s*")


def _extract_run_profiler_text(logs: str) -> str:
    lines: list[str] = []
    for raw_line in logs.splitlines():
        line = _ANSI_ESCAPE_RE.sub("", raw_line)
        if "\tRun profiler\t" not in line:
            continue
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue
        message = _LOG_TIMESTAMP_RE.sub("", parts[2]).rstrip()
        if not message:
            continue
        lines.append(message)
    return "\n".join(lines)


def _run_process(command: list[str], *, error_prefix: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"{error_prefix}: {stderr or f'exit code {completed.returncode}'}")
    return completed


def _run_process_json(command: list[str], *, error_prefix: str) -> Any:
    completed = _run_process(command, error_prefix=error_prefix)
    try:
        return json.loads(completed.stdout or "null")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{error_prefix}: invalid JSON output") from exc


def _resolve_hlh_submission_id(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    for candidate in (payload.get("submission_id"), payload.get("id")):
        value = str(candidate or "").strip()
        if value:
            return value
    return ""


def _summarize_score_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    summary: dict[str, Any] = {}
    for key in ("score", "status", "leaderboard", "gpu", "submission_id"):
        if key in payload:
            summary[key] = payload[key]
    result = payload.get("result")
    if isinstance(result, dict):
        for key in ("score", "status", "submission_id"):
            if key in result and key not in summary:
                summary[key] = result[key]
    return summary


def _safe_extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    for member in archive.getmembers():
        member_path = destination / member.name
        resolved = member_path.resolve()
        if destination.resolve() not in (resolved, *resolved.parents):
            raise HTTPException(status_code=400, detail=f"unsafe tar member: {member.name}")
    archive.extractall(destination, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Humanitys Last Hackathon server")
    _ = parser.parse_args()
    settings = get_settings()
    uvicorn.run(
        "hlh_server.service:create_app",
        host=settings.service_host,
        port=settings.service_port,
        factory=True,
    )


if __name__ == "__main__":
    main()
