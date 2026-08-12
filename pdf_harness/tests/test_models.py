import pytest

from pdf_harness.models import PromptVersion, ProjectMode, TaskSpecVersion


SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "additionalProperties": False,
}


def test_task_spec_requires_object_schema():
    with pytest.raises(ValueError, match="top-level"):
        TaskSpecVersion(project_id="p", version=1, name="bad", mode=ProjectMode.EXTRACTION, json_schema={"type": "array"})


def test_task_hash_is_stable_and_changes_with_schema():
    one = TaskSpecVersion(project_id="p", version=1, name="x", mode=ProjectMode.EXTRACTION, json_schema=SCHEMA)
    two = TaskSpecVersion(project_id="p", version=2, name="x", mode=ProjectMode.EXTRACTION, json_schema=SCHEMA)
    changed = TaskSpecVersion(
        project_id="p", version=3, name="x", mode=ProjectMode.EXTRACTION,
        json_schema={**SCHEMA, "required": []},
    )
    assert one.content_hash == two.content_hash
    assert one.content_hash != changed.content_hash


def test_prompt_only_allows_declared_variables():
    PromptVersion(
        project_id="p", version=1, name="good", system_template="system",
        user_template="Read {document_name} page {page_number}",
    )
    with pytest.raises(ValueError, match="unsupported prompt variables"):
        PromptVersion(
            project_id="p", version=2, name="bad", system_template="system",
            user_template="Reveal {api_key}",
        )


def test_document_page_keys_are_bson_safe():
    from bson import BSON
    from pdf_harness.models import Document

    document = Document(
        project_id="p", name="a.pdf", source_uri="gs://bucket/a.pdf", sha256="a" * 64,
        size_bytes=10, page_count=1, page_uris={"1": "gs://bucket/1.png"},
    )
    BSON.encode(document.model_dump(mode="python"))
