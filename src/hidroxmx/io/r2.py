"""Minimal client for Cloudflare R2 (S3-compatible).

Credentials are read from the process environment (``R2_ENDPOINT_URL``,
``R2_BUCKET``, ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``). This module
does not depend on ``s3fs`` to keep the base install lean; ``s3fs`` remains an
optional extra for callers that want a filesystem-like view.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(slots=True)
class R2Client:
    """Thin wrapper around ``boto3`` for the R2 bucket used by the project."""

    endpoint_url: str
    bucket: str
    access_key_id: str
    secret_access_key: str
    region: str = "auto"

    def _client(self):
        import boto3  # imported lazily

        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region,
        )

    # -------- object-level -----------------------------------------------------
    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._client().head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                return False
            raise

    def put_bytes(self, key: str, data: bytes) -> None:
        self._client().put_object(Bucket=self.bucket, Key=key, Body=data)

    def get_bytes(self, key: str) -> bytes:
        obj = self._client().get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read()

    def upload_file(self, key: str, path: Path) -> None:
        self._client().upload_file(str(path), self.bucket, key)

    def download_file(self, key: str, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._client().download_file(self.bucket, key, str(path))
        return path

    def delete(self, key: str) -> None:
        self._client().delete_object(Bucket=self.bucket, Key=key)

    def list_prefix(self, prefix: str) -> Iterator[str]:
        paginator = self._client().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for entry in page.get("Contents", []):
                yield entry["Key"]

    # -------- convenience for JSON -------------------------------------------
    def put_json(self, key: str, obj) -> None:
        import json

        self.put_bytes(key, json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8"))

    def get_json(self, key: str):
        import json

        return json.loads(self.get_bytes(key).decode("utf-8"))

    # -------- convenience for streaming Parquet ------------------------------
    def open_parquet_dataset(self, prefix: str):
        """Return a ``pyarrow.dataset.Dataset`` that scans ``s3://bucket/prefix``."""
        import pyarrow.fs as pafs
        import pyarrow.dataset as pads

        fs = pafs.S3FileSystem(
            endpoint_override=self.endpoint_url.replace("https://", ""),
            access_key=self.access_key_id,
            secret_key=self.secret_access_key,
            region=self.region,
            scheme="https",
        )
        return pads.dataset(f"{self.bucket}/{prefix}", filesystem=fs, format="parquet",
                            partitioning="hive")


def r2_from_env() -> R2Client:
    """Build an :class:`R2Client` from environment variables (see .env.example)."""
    missing = [k for k in ("R2_ENDPOINT_URL", "R2_BUCKET",
                           "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
               if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"missing environment variables for R2: {', '.join(missing)}. "
            f"See .env.example and load it with python-dotenv."
        )
    return R2Client(
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        bucket=os.environ["R2_BUCKET"],
        access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region=os.environ.get("AWS_DEFAULT_REGION", "auto"),
    )
