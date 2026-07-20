"""Minimal client for Cloudflare R2 (S3-compatible).

Credentials are read from the process environment (``R2_ENDPOINT_URL``,
``R2_BUCKET``, ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``). This module
does not depend on ``s3fs`` to keep the base install lean; ``s3fs`` remains an
optional extra for callers that want a filesystem-like view.
"""
from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


def _retry(fn: Callable[[], Any], *, attempts: int = 4, base_delay: float = 1.5) -> Any:
    """Retry an idempotent call on transient Cloudflare TLS / network errors."""
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            message = str(exc).lower()
            transient = any(
                marker in message
                for marker in (
                    "ssl", "eof", "reset", "timed out", "temporarily",
                    "connection", "handshake",
                )
            )
            if attempt >= attempts or not transient:
                break
            time.sleep(base_delay * (2 ** (attempt - 1)))
    assert last_exc is not None
    raise last_exc


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
        from botocore.config import Config

        # Aggressive internal retries. Cloudflare R2 sometimes drops the TLS
        # handshake on the first request from a fresh process on Windows +
        # OpenSSL 3.0.x; we belt-and-braces this with an outer ``_retry``
        # wrapper on every entry point.
        cfg = Config(
            retries={"max_attempts": 10, "mode": "adaptive"},
            connect_timeout=15,
            read_timeout=60,
        )
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region,
            config=cfg,
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
        _retry(lambda: self._client().put_object(Bucket=self.bucket, Key=key, Body=data))

    def get_bytes(self, key: str) -> bytes:
        obj = _retry(lambda: self._client().get_object(Bucket=self.bucket, Key=key))
        return obj["Body"].read()

    def upload_file(self, key: str, path: Path) -> None:
        try:
            _retry(lambda: self._client().upload_file(str(path), self.bucket, key), attempts=2)
        except Exception:
            # Cloudflare R2 sometimes refuses the botocore TLS handshake on
            # Windows / Python 3.12 with OpenSSL 3.x. curl works with the same
            # credentials, so we fall back to a presigned PUT + curl transfer.
            self._curl_put(key, path)

    def download_file(self, key: str, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _retry(lambda: self._client().download_file(self.bucket, key, str(path)), attempts=2)
        except Exception:
            self._curl_get(key, path)
        return path

    def _curl_put(self, key: str, path: Path) -> None:
        import shutil
        import subprocess

        if shutil.which("curl") is None:
            raise RuntimeError("curl is not on PATH; cannot use it as an R2 upload fallback")
        url = self._client().generate_presigned_url(
            "put_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=600,
        )
        cmd = ["curl", "-fsSL", "--retry", "3", "--retry-delay", "2",
               "-X", "PUT", "-T", str(path), url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"curl PUT to R2 failed (exit {result.returncode}): {result.stderr[-400:]}"
            )

    def _curl_get(self, key: str, path: Path) -> None:
        import shutil
        import subprocess

        if shutil.which("curl") is None:
            raise RuntimeError("curl is not on PATH; cannot use it as an R2 download fallback")
        url = self._client().generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=600,
        )
        tmp = path.with_suffix(path.suffix + ".part")
        cmd = ["curl", "-fsSL", "--retry", "3", "--retry-delay", "2", "-o", str(tmp), url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"curl GET from R2 failed (exit {result.returncode}): {result.stderr[-400:]}"
            )
        tmp.replace(path)

    def delete(self, key: str) -> None:
        _retry(lambda: self._client().delete_object(Bucket=self.bucket, Key=key))

    def list_prefix(self, prefix: str) -> Iterator[str]:
        client = self._client()
        paginator = client.get_paginator("list_objects_v2")

        def _pages():
            return list(paginator.paginate(Bucket=self.bucket, Prefix=prefix))

        for page in _retry(_pages):
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
