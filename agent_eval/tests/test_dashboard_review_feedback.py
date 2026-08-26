"""Regression tests for the reader-facing GitHub dashboard review feedback."""

from analysis import build_dashboard, build_date_fix_demo
from agent_eval import config


def _template(path):
    return path.read_text()


def test_stanford_gateway_price_snapshot_and_provenance():
    assert config.model_price("playground", "gpt-5-mini") == (0.125, 1.0)
    assert config.model_price("playground", "claude-sonnet-4-6") == (1.5, 7.5)
    assert config.model_price("playground", "gemini-2.5-pro") == (2.5, 15)
    assert config.model_price("playground", "gemini-2.5-flash") == (0.3, 2.5)
    assert config.model_price("playground", "DeepSeek-V3.2") == (None, None)
    assert config.model_price("nim") == (0, 0)
    assert config.model_price("qwen") == (0, 0)
    assert config.model_price("missing-backend") == (None, None)

    pricing = config.BACKENDS["playground"]["pricing"]
    assert pricing["unit"] == "USD per 1M tokens"
    assert pricing["source_url"] == "https://uit.stanford.edu/service/ai-api-gateway/rates"
    assert pricing["verified_at"] == "2026-08-18"
    assert "snapshot" in pricing["note"].casefold()
    assert "verify" in pricing["note"].casefold()


def test_version_metadata_explains_dirty_code_without_losing_identity():
    versions = build_dashboard._version_metadata({"f11e283-dirty"})
    meta = versions["f11e283-dirty"]
    assert meta["short_sha"] == "f11e283"
    assert meta["dirty"] is True
    assert "modified" in meta["label"].lower()
    assert "f11e283" in meta["label"]


def test_glossary_has_stable_ids_and_is_alphabetized():
    terms = [item["term"] for item in build_dashboard.GLOSSARY]
    assert terms == sorted(terms, key=str.casefold)
    assert all(item.get("id") for item in build_dashboard.GLOSSARY)
    assert len({item["id"] for item in build_dashboard.GLOSSARY}) == len(terms)


def test_glossary_definitions_use_plain_language():
    definitions = " ".join(item["def"] for item in build_dashboard.GLOSSARY).casefold()
    jargon = (
        "composite", "levenshtein", "char_f1", "set_f1", "sequence_lcs", "word-iou",
        "type-tool", "field_type", "save_evaluation", "prompt_name", "max_steps",
        "backend", "endpoint", "telemetry", "latency", " null",
    )
    assert not [word for word in jargon if word in definitions]


def test_date_fix_demo_uses_distinct_decision_and_review_language():
    labels = {item["key"]: item["label"] for item in build_date_fix_demo.OUTCOMES}
    assert labels["flagged"] == "Abstained — no single date"

    html = _template(build_date_fix_demo.TEMPLATE)
    assert "Can an AI reviewer fix a TV guide date without guessing?" in html
    assert "Rows requiring human review" in html
    assert "Why included" in html
    assert "Nothing here is reconstructed" not in html
    assert "Fixing the wording is a one-line prompt change" not in html


def test_date_fix_about_card_is_scannable_and_keeps_detail_secondary():
    html = _template(build_date_fix_demo.TEMPLATE)
    about = html.split('<section class="card" id="about">', 1)[1].split("</section>", 1)[0]
    assert 'class="about-grid"' in about
    assert "What it checks" in about
    assert "What it can do" in about
    assert "What matters most" in about
    assert '<details class="provenance-details">' in about
    assert about.index("How these results were checked") < about.index('id="provenance"')
    assert "derived" not in about.casefold()
    assert "abstention" not in about.casefold()


def test_date_fix_demo_prioritizes_context_and_preserves_full_text():
    html = _template(build_date_fix_demo.TEMPLATE)
    assert html.index('id="about"') < html.index('id="kpis"')
    assert html.index('id="promptCard"') < html.index('id="kpis"')
    assert "Ground truth" in html
    assert "Original model output" in html
    assert "Agent’s final value" in html
    assert "data-full-text" in html
    assert "esc(String(s.thinking_api))" in html


def test_date_fix_prompt_card_explains_both_instruction_parts_plainly():
    html = _template(build_date_fix_demo.TEMPLATE)
    prompt = html.split('<section class="card" id="promptCard">', 1)[1].split("</section>", 1)[0]
    assert 'class="instruction-grid"' in prompt
    assert "Same rules for every answer" in prompt
    assert "One row at a time" in prompt
    assert "The rules stay the same. The row details change." in prompt
    assert '<details id="promptBox" class="prompt-details">' in prompt
    assert '<details id="promptBox" open>' not in prompt
    assert "See the exact instructions" in prompt
    assert 'id="promptText" class="prompt-text-grid"' in prompt
    assert ".prompt-text-grid { display: grid;" in html
    assert "align-items: start" in html
    assert "System prompt —" not in html
    assert "Per-answer" not in html


def test_date_fix_section_introductions_use_the_full_card_width():
    html = _template(build_date_fix_demo.TEMPLATE)
    assert html.count('class="section-lead"') == 5
    assert ".section-lead {" in html
    assert "width: 100%" in html
    assert "max-width: 76ch" not in html
    assert "A complete review has three steps" in html
    assert "silent errors visible" in html


def test_date_fix_queue_highlights_wrong_rows_that_requested_review():
    html = _template(build_date_fix_demo.TEMPLATE)
    assert ".queue-priority > td" in html
    assert "queue-priority-image" in html
    assert 'var priority = r.display_outcome === "wrong" && r.needs_review;' in html
    assert 'priority ? " queue-priority" : ""' in html
    assert "⚑ image " in html


def test_every_date_fix_card_and_reviewer_filter_are_collapsible():
    html = _template(build_date_fix_demo.TEMPLATE)
    card_count = html.count('<section class="card"')
    assert card_count > 1
    assert html.count('<details class="section-details"') == card_count
    assert html.count('<summary class="section-toggle">') == card_count
    assert '<summary class="section-toggle"><h2>What this tests</h2></summary>' in html
    assert '<details class="filters filter-details" open>' in html
    assert '<summary class="filter-toggle">Reviewer filter</summary>' in html


def test_metric_dashboard_has_readable_labels_and_missing_data_states():
    html = _template(build_dashboard.TEMPLATE)
    assert "Average LLM time (s)" in html
    assert "Average overhead time (s)" in html
    assert 'header("Save success"' in html
    assert 'header("Save %"' not in html
    assert "Unpriced" in html
    assert "Partial" in html
    assert "N/A — remote API" in html
    assert "Missing telemetry" in html


def test_metric_dashboard_tables_paths_and_glossary_are_reviewable():
    html = _template(build_dashboard.TEMPLATE)
    assert "wide-table" in html
    assert "run-table" in html
    assert "Line width shows run count; hover for the exact number" in html
    assert "✓ Clean" in html and "✗ Misrouted" in html and "⌀ Unscored" in html
    assert "Full tool name:" in html
    assert '<details class="section-details">' in html
    assert "glossaryLink" in html


def test_metric_dashboard_definitions_open_in_place_without_version_helper():
    html = _template(build_dashboard.TEMPLATE)
    assert '“modified” means' not in html
    assert 'class="glossary-term"' in html
    assert 'data-definition="' in html
    assert 'id="definitionTooltip"' in html
    assert 'href="#glossary-' not in html


def test_every_metric_dashboard_card_is_collapsible_including_filters():
    html = _template(build_dashboard.TEMPLATE)
    card_count = html.count('<section class="card">')
    assert card_count > 1
    assert html.count('<details class="section-details"') == card_count
    assert html.count('<summary class="section-toggle">') == card_count
    assert '<summary class="section-toggle"><h2>Filters</h2></summary>' in html
