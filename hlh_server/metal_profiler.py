from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path


def _helper_program() -> str:
    return textwrap.dedent(
        """
        import contextlib
        import io
        import json
        import os
        import platform
        import runpy
        import sys
        import time
        import traceback

        def safe_call(fn, default=None):
            try:
                return fn()
            except Exception:
                return default

        file_path = sys.argv[1]
        capture_path = sys.argv[2] if len(sys.argv) > 2 else ""
        result = {
            "file_path": os.path.basename(file_path) if file_path else "unknown",
            "profile_tool": "mps",
            "backend": "metal",
            "gpu": "Apple Silicon (Metal)",
            "driver_version": "",
            "runtime_version": "",
            "kernels": [],
            "summary": {
                "total_kernels": 0,
                "total_duration_us": 0.0,
                "avg_compute_throughput_pct": 0.0,
                "avg_memory_throughput_pct": 0.0,
                "bottleneck": "unknown"
            },
            "raw_artifacts": {}
        }

        try:
            import torch

            result["runtime_version"] = getattr(torch, "__version__", "")
            result["gpu"] = f"Apple Silicon ({platform.machine()})"

            mps = getattr(torch, "mps", None)
            mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
            profiler = getattr(mps, "profiler", None) if mps is not None else None
            result["mps"] = {
                "is_built": bool(getattr(mps_backend, "is_built", lambda: False)()),
                "is_available": bool(getattr(mps_backend, "is_available", lambda: False)()),
                "device_count": safe_call(getattr(mps, "device_count", lambda: 0), 0),
            }
            if not result["mps"]["is_available"]:
                result["error"] = "MPS backend is not available on this host"
            else:
                memory = {
                    "current_allocated_memory_bytes": safe_call(mps.current_allocated_memory, 0),
                    "driver_allocated_memory_bytes": safe_call(mps.driver_allocated_memory, 0),
                    "recommended_max_memory_bytes": safe_call(mps.recommended_max_memory, 0),
                }
                script_stdout = io.StringIO()
                script_stderr = io.StringIO()
                contexts = []
                if profiler is not None and hasattr(profiler, "profile"):
                    try:
                        contexts.append(profiler.profile())
                    except Exception as exc:
                        result["raw_artifacts"]["signpost_error"] = str(exc)
                if capture_path and profiler is not None and hasattr(profiler, "metal_capture"):
                    try:
                        contexts.append(profiler.metal_capture(capture_path))
                    except Exception as exc:
                        result["raw_artifacts"]["gputrace_error"] = str(exc)

                start = time.perf_counter()
                old_argv = sys.argv[:]
                try:
                    with contextlib.ExitStack() as stack:
                        for ctx in contexts:
                            stack.enter_context(ctx)
                        with contextlib.redirect_stdout(script_stdout), contextlib.redirect_stderr(script_stderr):
                            sys.argv = [file_path]
                            runpy.run_path(file_path, run_name="__main__")
                        if hasattr(mps, "synchronize"):
                            mps.synchronize()
                finally:
                    sys.argv = old_argv

                duration_us = round((time.perf_counter() - start) * 1_000_000, 2)
                memory.update(
                    {
                        "current_allocated_memory_bytes": safe_call(mps.current_allocated_memory, memory["current_allocated_memory_bytes"]),
                        "driver_allocated_memory_bytes": safe_call(mps.driver_allocated_memory, memory["driver_allocated_memory_bytes"]),
                        "recommended_max_memory_bytes": safe_call(mps.recommended_max_memory, memory["recommended_max_memory_bytes"]),
                    }
                )
                result["memory"] = memory
                result["summary"]["total_duration_us"] = duration_us
                result["script_stdout"] = script_stdout.getvalue()
                result["script_stderr"] = script_stderr.getvalue()
                if capture_path and os.path.exists(capture_path):
                    result["raw_artifacts"]["gputrace"] = capture_path
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["stderr"] = traceback.format_exc()[-4000:]

        print(json.dumps(result, sort_keys=True))
        """
    )


def _capture_enabled() -> bool:
    return os.environ.get("METAL_CAPTURE_ENABLED", "") == "1"


def _xctrace_available() -> bool:
    return bool(shutil.which("xctrace"))


def _build_error_payload(file_path: str, error: str, stderr: str = "") -> dict:
    payload = {
        "file_path": os.path.basename(file_path) if file_path else "unknown",
        "profile_tool": "mps",
        "backend": "metal",
        "gpu": "Apple Silicon (Metal)",
        "kernels": [],
        "summary": {
            "total_kernels": 0,
            "total_duration_us": 0.0,
            "avg_compute_throughput_pct": 0.0,
            "avg_memory_throughput_pct": 0.0,
            "bottleneck": "unknown",
        },
        "error": error,
    }
    if stderr:
        payload["stderr"] = stderr
    return payload


def _parse_json_payload(stdout_text: str) -> dict | None:
    for line in reversed(stdout_text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


async def _run_process(cmd: list[str], *, env: dict[str, str]) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    stdout_bytes, stderr_bytes = await proc.communicate()
    return (
        proc.returncode,
        stdout_bytes.decode("utf-8", errors="replace"),
        stderr_bytes.decode("utf-8", errors="replace"),
    )


async def _run_helper(file_path: str, *, capture_path: str | None = None) -> tuple[int, str, str]:
    cmd = [sys.executable, "-c", _helper_program(), file_path]
    if capture_path:
        cmd.append(capture_path)
    env = dict(os.environ)
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    if _capture_enabled():
        env.setdefault("MTL_CAPTURE_ENABLED", "1")
    return await _run_process(cmd, env=env)


async def profile_file(file_path: str) -> dict:
    trace_path = None
    capture_path = None
    if _capture_enabled() and _xctrace_available():
        trace_dir = tempfile.mkdtemp(prefix="hlh-metal-capture-")
        trace_path = Path(trace_dir) / "capture.gputrace"
        capture_path = str(trace_path)
    returncode, stdout_text, stderr_text = await _run_helper(file_path, capture_path=capture_path)
    payload = _parse_json_payload(stdout_text)
    if payload is not None:
        return payload
    if returncode != 0:
        return _build_error_payload(file_path, f"profiler exited with code {returncode}", stderr_text)
    return _build_error_payload(file_path, "profiler returned no JSON payload", stderr_text)
