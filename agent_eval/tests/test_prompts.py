"""Tests for the prompt registry — index resolution + the back-compat default API."""

import pytest

from agent_eval import prompts


def test_default_is_lowest_index():
    name, system, user, key = prompts.resolve_prompt()
    assert key == prompts.DEFAULT_KEY == min(prompts._REGISTRY)
    assert (name, system, user, key) == (
        prompts.PROMPT_NAME, prompts.METRIC_EVAL_SYSTEM, prompts._DEFAULT_USER, prompts.PROMPT_KEY)


def test_resolve_by_index_name_and_intstring_agree():
    by_index = prompts.resolve_prompt(1)
    assert by_index == prompts.resolve_prompt("1")            # int-like string
    assert by_index == prompts.resolve_prompt("composite_v1")  # by name
    assert by_index[0] == "composite_v1" and by_index[3] == 1


def test_resolve_unknown_raises():
    with pytest.raises(KeyError):
        prompts.resolve_prompt(999)
    with pytest.raises(KeyError):
        prompts.resolve_prompt("does-not-exist")


def test_eval_user_prompt_interpolates_and_selects_variant():
    assert "task_id=2450" in prompts.eval_user_prompt("2450", 3)
    assert "run_id=None" in prompts.eval_user_prompt("2450", None)
    # explicit selector yields the same text as the default (composite_v1, lowest index)
    assert prompts.eval_user_prompt("2450", 3, prompt="composite_v1") == \
        prompts.eval_user_prompt("2450", 3)


def test_composite_v2_loads_at_index_2_and_v1_stays_default():
    name, system, user, key = prompts.resolve_prompt("composite_v2")
    assert (name, key) == ("composite_v2", 2)
    assert prompts.resolve_prompt(2)[0] == "composite_v2"       # by index too
    assert "{task_id}" in user and "{run_id}" in user and system
    assert prompts.PROMPT_NAME == "composite_v1"                # lowest index remains default
