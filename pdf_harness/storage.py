"""Artifact storage adapters and page-first PDF rendering."""

from __future__ import annotations

import hashlib
import io
import tempfile
from pathlib import Path
from typing import Protocol


class ArtifactStore(Protocol):
    def health(self) -> tuple[bool, str]: ...
    def put(self, key: str, data: bytes, content_type: str) -> str: ...
    def get(self, uri: str) -> bytes: ...


def document_source_key(project_id: str, document_id: str, digest: str) -> str:
    return f"projects/{project_id}/documents/{document_id}/{digest}/source.pdf"


def page_key(project_id: str, document_id: str, digest: str, page_number: int) -> str:
    return f"projects/{project_id}/documents/{document_id}/{digest}/pages/{page_number:05d}.png"


class LocalArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def health(self) -> tuple[bool, str]:
        return True, f"local artifacts at {self.root}"

    def _path(self, value: str) -> Path:
        key = value.removeprefix("local://").lstrip("/")
        path = (self.root / key).resolve()
        if self.root not in path.parents and path != self.root:
            raise ValueError("artifact key escapes storage root")
        return path

    def put(self, key: str, data: bytes, content_type: str) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return f"local://{key}"

    def get(self, uri: str) -> bytes:
        return self._path(uri).read_bytes()


class GCSArtifactStore:
    def __init__(self, bucket: str) -> None:
        from google.cloud import storage

        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket)

    def health(self) -> tuple[bool, str]:
        try:
            self.bucket.reload()
            return True, f"GCS bucket gs://{self.bucket.name} reachable"
        except Exception as exc:  # noqa: BLE001
            return False, f"GCS unavailable: {type(exc).__name__}"

    def put(self, key: str, data: bytes, content_type: str) -> str:
        blob = self.bucket.blob(key)
        try:
            blob.upload_from_string(data, content_type=content_type, if_generation_match=0)
        except Exception as exc:  # the immutable object may already exist on an idempotent retry
            if type(exc).__name__ != "PreconditionFailed":
                raise
        return f"gs://{self.bucket.name}/{key}"

    def get(self, uri: str) -> bytes:
        prefix = f"gs://{self.bucket.name}/"
        if not uri.startswith(prefix):
            raise ValueError(f"artifact URI is outside configured bucket: {uri}")
        return self.bucket.blob(uri.removeprefix(prefix)).download_as_bytes()


def render_pdf(pdf: bytes, grayscale: bool = True, dpi: int = 200) -> list[bytes]:
    """Render a PDF into ordered PNG bytes. Poppler is required by pdf2image."""
    from pdf2image import convert_from_bytes

    images = convert_from_bytes(pdf, dpi=dpi, grayscale=grayscale, fmt="png")
    rendered: list[bytes] = []
    for image in images:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        rendered.append(buf.getvalue())
    if not rendered:
        raise ValueError("PDF rendered zero pages")
    return rendered


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
