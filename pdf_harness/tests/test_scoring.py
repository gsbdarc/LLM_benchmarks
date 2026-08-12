import pytest

from pdf_harness.models import FieldRule
from pdf_harness.scoring import score_field, score_record


def test_typed_metrics_cover_text_number_date_boolean_and_list():
    assert score_field("  HELLO ", "hello", {"type": "string"})["score"] == 1
    assert score_field("$10.05", 10, {"type": "number"}, FieldRule(numeric_tolerance=.1))["score"] == 1
    assert score_field("Jan 2, 2024", "2024-01-02", {"type": "string", "format": "date"})["score"] == 1
    assert score_field("yes", True, {"type": "boolean"})["score"] == 1
    assert score_field(["A", "B"], ["b", "c"], {"type": "array"})["f1"] == pytest.approx(.5)


def test_record_reports_quality_and_required_coverage():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["name", "count"],
    }
    result = score_record({"name": "Alice", "count": None}, {"name": "alice", "count": 3}, schema)
    assert result["overall_score"] == .5
    assert result["required_field_coverage"] == .5
