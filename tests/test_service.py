from __future__ import annotations

import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from hlh_server.config import Settings
from hlh_server import service as service_mod
from hlh_server.service import create_app


class ServiceTests(unittest.TestCase):
    def test_extract_profile_from_workflow_logs_prefers_run_profiler_payload(self) -> None:
        logs = """profile\tRun profiler\t2026-04-22T23:21:20.0000000Z {
profile\tRun profiler\t2026-04-22T23:21:20.0000000Z   "backend": "metal",
profile\tRun profiler\t2026-04-22T23:21:20.0000000Z   "summary": {"total_duration_us": 1.23}
profile\tRun profiler\t2026-04-22T23:21:20.0000000Z }"""

        payload = service_mod._extract_profile_from_workflow_logs(logs)
        self.assertIsInstance(payload, dict)
        assert isinstance(payload, dict)
        self.assertEqual(payload["backend"], "metal")
        self.assertEqual(payload["summary"]["total_duration_us"], 1.23)

    def test_extract_profile_from_workflow_logs_ignores_ansi_wrapped_lines(self) -> None:
        logs = """profile\tRun profiler\t2026-04-22T23:21:20.0000000Z \u001b[36;1m{\u001b[0m
profile\tRun profiler\t2026-04-22T23:21:20.0000000Z \u001b[36;1m  "backend": "metal"\u001b[0m
profile\tRun profiler\t2026-04-22T23:21:20.0000000Z \u001b[36;1m}\u001b[0m"""

        payload = service_mod._extract_profile_from_workflow_logs(logs)
        self.assertEqual(payload, {"backend": "metal"})

    def test_hlh_profile_endpoint_profiles_kernel_content(self) -> None:
        app = create_app()
        client = TestClient(app)
        expected_profile = {"backend": "metal", "summary": {"total_duration_us": 123.4}}
        with patch("hlh_server.service.profile_file", new=AsyncMock(return_value=expected_profile)), patch(
            "hlh_server.service.auth.validate_cli_id",
            new=AsyncMock(return_value=True),
        ):
            response = client.post(
                "/v1/hlh/profile",
                files={"kernel": ("my_kernel.py", b"print('hello')\n", "text/x-python")},
                headers={"X-Popcorn-Cli-Id": "cli-test"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected_profile)

    def test_hlh_profile_endpoint_runs_via_github_workflow(self) -> None:
        settings = Settings(profile_runner="github")
        app = create_app(settings=settings)
        client = TestClient(app)
        expected_profile = {"backend": "metal", "summary": {"total_duration_us": 321.0}}
        with patch(
            "hlh_server.service._profile_via_github_workflow",
            return_value=expected_profile,
        ) as workflow_mock, patch(
            "hlh_server.service.auth.validate_cli_id",
            new=AsyncMock(return_value=True),
        ):
            response = client.post(
                "/v1/hlh/profile",
                files={"kernel": ("my_kernel.py", b"print('hello')\n", "text/x-python")},
                headers={"X-Popcorn-Cli-Id": "cli-test"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected_profile)
        workflow_mock.assert_called_once()

    def test_find_dispatched_run_id_by_request_id_matches_run_name(self) -> None:
        runs = [
            {"databaseId": 1001, "name": "profile-wrong", "displayTitle": "Profiling Job"},
            {"databaseId": 1002, "name": "profile-req-123", "displayTitle": "Profiling Job"},
        ]
        with patch("hlh_server.service._run_process_json", return_value=runs):
            run_id = service_mod._find_dispatched_run_id_by_request_id(
                workflow_file="main.yml",
                branch="main",
                request_id="req-123",
            )
        self.assertEqual(run_id, 1002)

    def test_profile_via_github_workflow_dispatches_request_id(self) -> None:
        expected_profile = {"backend": "metal", "summary": {"total_duration_us": 123.0}}
        completed_dispatch = subprocess.CompletedProcess(args=["gh"], returncode=0, stdout="", stderr="")
        completed_logs = subprocess.CompletedProcess(
            args=["gh"],
            returncode=0,
            stdout='{"backend":"metal","summary":{"total_duration_us":123.0}}',
            stderr="",
        )
        settings = Settings(profile_runner="github")
        with patch(
            "hlh_server.service.uuid.uuid4",
            return_value=SimpleNamespace(hex="req-xyz"),
        ), patch(
            "hlh_server.service._run_process",
            side_effect=[completed_dispatch, completed_logs],
        ) as run_process_mock, patch(
            "hlh_server.service._find_dispatched_run_id_by_request_id",
            return_value=4242,
        ) as find_mock, patch(
            "hlh_server.service._run_process_json",
            return_value={"status": "completed", "conclusion": "success", "url": "https://example"},
        ), patch(
            "hlh_server.service._extract_profile_from_workflow_logs",
            return_value=expected_profile,
        ):
            profile = service_mod._profile_via_github_workflow("print('hello')\n", "my_kernel.py", settings)

        self.assertEqual(profile, expected_profile)
        dispatch_command = run_process_mock.call_args_list[0].args[0]
        self.assertIn("request_id=req-xyz", dispatch_command)
        find_mock.assert_called_once_with(
            workflow_file="main.yml",
            branch="main",
            request_id="req-xyz",
        )

    def test_hlh_profile_endpoint_rejects_blank_content(self) -> None:
        app = create_app()
        client = TestClient(app)
        with patch("hlh_server.service.auth.validate_cli_id", new=AsyncMock(return_value=True)):
            response = client.post(
                "/v1/hlh/profile",
                files={"kernel": ("kernel.py", b"   ", "text/x-python")},
                headers={"X-Popcorn-Cli-Id": "cli-test"},
            )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "kernel_content must not be empty")

    def test_hlh_profile_endpoint_requires_cli_id(self) -> None:
        app = create_app()
        client = TestClient(app)
        response = client.post(
            "/v1/hlh/profile",
            files={"kernel": ("kernel.py", b"print('hello')\n", "text/x-python")},
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "missing X-Popcorn-Cli-Id")

    def test_hlh_profile_endpoint_rejects_invalid_cli_id(self) -> None:
        app = create_app()
        client = TestClient(app)
        with patch("hlh_server.service.auth.validate_cli_id", new=AsyncMock(return_value=False)):
            response = client.post(
                "/v1/hlh/profile",
                files={"kernel": ("kernel.py", b"print('hello')\n", "text/x-python")},
                headers={"X-Popcorn-Cli-Id": "cli-bad"},
            )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "invalid Popcorn cli_id")

    def test_hlh_submit_runs_server_side_submit(self) -> None:
        app = create_app()
        client = TestClient(app)
        completed = subprocess.CompletedProcess(
            args=["popcorn", "submit", "kernel.py"],
            returncode=0,
            stdout="human-readable output\n",
            stderr="",
        )

        def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            output_path = Path(command[command.index("--output") + 1])
            output_path.write_text(json.dumps({"status": "ok", "score": 99.1, "submission_id": "sub-123"}))
            return completed

        with patch("hlh_server.service.auth.validate_cli_id", new=AsyncMock(return_value=True)), patch(
            "hlh_server.service.subprocess.run",
            side_effect=fake_run,
        ):
            response = client.post(
                "/v1/hlh/submit",
                data={"leaderboard": "demo", "gpu": "H100"},
                files={"kernel": ("kernel.py", b"print('hello')\n", "text/x-python")},
                headers={"X-Popcorn-Cli-Id": "cli-test"},
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["submission_id"], "sub-123")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["score_summary"]["score"], 99.1)

    def test_hlh_submit_bundle_rejects_when_disabled(self) -> None:
        settings = Settings(submit_bundle_enabled=False)
        app = create_app(settings=settings)
        client = TestClient(app)
        with patch("hlh_server.service.auth.validate_cli_id", new=AsyncMock(return_value=True)):
            response = client.post(
                "/v1/hlh/submit/bundle",
                data={"submission_id": "sub-123", "kernel_filename": "kernel.py"},
                files={"bundle": ("bundle.tar.gz", b"abc", "application/gzip")},
                headers={"X-Popcorn-Cli-Id": "cli-test"},
            )
        self.assertEqual(response.status_code, 503)

    def test_hlh_submit_bundle_requires_hf_token(self) -> None:
        app = create_app()
        client = TestClient(app)
        with patch.dict("os.environ", {}, clear=True), patch(
            "hlh_server.service.auth.validate_cli_id",
            new=AsyncMock(return_value=True),
        ):
            response = client.post(
                "/v1/hlh/submit/bundle",
                data={"submission_id": "sub-123", "kernel_filename": "kernel.py"},
                files={"bundle": ("bundle.tar.gz", b"abc", "application/gzip")},
                headers={"X-Popcorn-Cli-Id": "cli-test"},
            )
        self.assertEqual(response.status_code, 500)
        self.assertIn("missing Hugging Face token", response.json()["detail"])

    def test_hlh_submit_bundle_extracts_and_uploads_files(self) -> None:
        app = create_app()
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tar_path = Path(tmp_dir) / "bundle.tar.gz"
            payload_path = Path(tmp_dir) / "artifact.txt"
            payload_path.write_text("hello\n")
            with tarfile.open(tar_path, mode="w:gz") as archive:
                archive.add(payload_path, arcname="nested/artifact.txt")
            with patch.dict("os.environ", {"HF_TOKEN": "hf-test"}), patch(
                "hlh_server.service.auth.validate_cli_id",
                new=AsyncMock(return_value=True),
            ), patch(
                "hlh_server.service.HFBucketArtifactStore.create_context",
                return_value=type("Ctx", (), {"bucket_uri": "hf://buckets/demo/submissions/sub-123", "prefix": "submissions/sub-123", "bucket_id": "demo/repo"})(),
            ), patch(
                "hlh_server.service.HFBucketArtifactStore.make_upload_target",
                side_effect=lambda context, relative_path: f"hf://buckets/demo/repo/{relative_path}",
            ), patch(
                "hlh_server.service.HFBucketArtifactStore.upload_file",
            ) as upload_mock:
                response = client.post(
                    "/v1/hlh/submit/bundle",
                    data={"submission_id": "sub-123", "kernel_filename": "kernel.py"},
                    files={"bundle": ("bundle.tar.gz", tar_path.read_bytes(), "application/gzip")},
                    headers={"X-Popcorn-Cli-Id": "cli-test"},
                )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["received"], True)
        upload_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
