from pathlib import Path

import pytest

from pdf_harness.bootstrap import build_context
from pdf_harness.config import Settings
from pdf_harness.repository import InMemoryRepository


def test_build_context_supports_disposable_memory_preview(tmp_path: Path) -> None:
    context = build_context(Settings(
        repository_backend="memory",
        environment="development",
        local_data_dir=tmp_path,
    ))

    assert isinstance(context.repo, InMemoryRepository)
    assert context.repo.health() == (True, "in-memory repository available")


def test_memory_repository_is_rejected_in_production(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only available in development"):
        build_context(Settings(
            repository_backend="memory",
            environment="production",
            local_data_dir=tmp_path,
        ))
