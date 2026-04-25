from __future__ import annotations

import posixpath
from dataclasses import dataclass
from pathlib import Path


HF_PREFIX = "hf://buckets/"


@dataclass(frozen=True)
class UploadContext:
    bucket_uri: str
    prefix: str
    bucket_id: str | None = None


class HFBucketArtifactStore:
    def __init__(self, bucket_template: str):
        self.bucket_template = bucket_template

    def create_context(
        self,
        *,
        submission_id: str,
        hf_user: str,
        hf_token: str,
    ) -> UploadContext:
        bucket_id = self.bucket_template.format(hf_user=hf_user, namespace=hf_user)
        self._api(hf_token).create_bucket(bucket_id, private=True, exist_ok=True)
        prefix = posixpath.join("submissions", submission_id)
        bucket_uri = f"{HF_PREFIX}{bucket_id}/{prefix}"
        return UploadContext(bucket_uri=bucket_uri, prefix=prefix, bucket_id=bucket_id)

    def make_upload_target(self, context: UploadContext, relative_path: str) -> str:
        if not context.bucket_id:
            raise ValueError("missing bucket id")
        return f"{HF_PREFIX}{context.bucket_id}/{posixpath.join(context.prefix, relative_path)}"

    def upload_file(self, upload_target: str, local_path: Path, token: str) -> None:
        bucket_id, remote_path = parse_hf_target(upload_target)
        self._api(token).batch_bucket_files(bucket_id, add=[(str(local_path), remote_path)])

    @staticmethod
    def _api(token: str):
        from huggingface_hub import HfApi

        return HfApi(token=token)


def parse_hf_target(upload_target: str) -> tuple[str, str]:
    if not upload_target.startswith(HF_PREFIX):
        raise ValueError(f"Not an HF bucket target: {upload_target}")
    suffix = upload_target.removeprefix(HF_PREFIX)
    namespace, bucket_name, *remainder = suffix.split("/")
    return f"{namespace}/{bucket_name}", "/".join(remainder)
