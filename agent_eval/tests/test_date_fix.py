"""Tests for the date-fix task: the date rule, the parser, and the scorer.

The headline test is `test_rule_against_every_ground_truth`, which pins the rule's
behaviour on the real corpus (30 matched / 3 ambiguous / 2 known misses) using a
frozen copy of the 35 ground-truth rows — the same numbers the demo quotes. If a
change to the rule or the parser moves them, this fails and tells you which image.
"""

import json
from types import SimpleNamespace

from agent_eval.reporting import scorers
from agent_eval.tools import compute_guide_date, parse_date

# The 35 ground_truths rows (image_id, newspaper_date, day_of_week, tv_guide_date),
# frozen here so the suite stays offline. Regenerate with:
#   db.ground_truths.find({}, {newspaper_date:1, day_of_week:1, tv_guide_date:1})
GROUND_TRUTH = [
    ("0", "Dec 17 2000", "Wednesday", "December 20 2000"),
    ("1", "Aug 3 2014", "Saturday", "August 9 2014"),
    ("2", "Feb 22 2015", "Sunday", "February 22 2015"),
    ("3", "Jan 12 2014", "Saturday", "January 18 2014"),
    ("4", "Nov 23 2014", "Saturday", "November 29 2014"),
    ("5", "May 28 1995", "Saturday", "June 3 1995"),
    ("6", "Oct 30 1994", "Saturday", "November 5 1994"),
    ("7", "Apr 13 2008", "Monday", "April 14 2008"),
    ("8", "Apr 25 2010", "Thursday", "April 29 2010"),
    ("9", "Jun 17 2012", "Monday", "June 18 2012"),
    ("10", "Jun 9 2013", "Sunday", "June 9 2013"),
    ("11", "Mar 11 2012", "Friday", "March 16 2012"),
    ("12", "Mar 30 2014", "Monday", "March 31 2014"),
    ("13", "May 15 2016", "Sunday", "May 22 2016"),
    ("14", "Oct 14 2007", "Monday", "October 15 2007"),
    ("15", "Oct 18 2009", "Saturday", "October 24 2009"),
    ("16", "Oct 9 2016", "Tuesday", "October 11 2016"),
    ("17", "Oct 12 1997", "Monday", "October 13 1997"),
    ("18", "Jan 17 1999", "Thursday", "January 21 1999"),
    ("19", "Nov 8 1998", "Friday", "November 13 1998"),
    ("20", "Sep 5 1999", "Wednesday", "September 8 1999"),
    ("21", "Mar 5 2000", "Saturday", "March 11 2000"),
    ("22", "May 31 1998", "Tuesday", "June 2 1998"),
    ("23", "Feb 9 1997", "Thursday", "February 13 1997"),
    ("24", "Mar 21 2004", "Friday", "March 26 2004"),
    ("25", "Mar 9 2003", "Monday", "March 10 2003"),
    ("26", "May 26 2002", "Sunday", "May 26 2002"),
    ("27", "Sep 28 1997", "Thursday", "October 2 1997"),
    ("28", "Apr 3 2005", "Monday-Friday", "April 4-8 2005"),
    ("29", "Sep 15 2002", "Sunday", "September 15 2002"),
    ("30", "Nov 6 1994", "Tuesday", "November 8 1994"),
    ("31", "Mar 27 2011", "Saturday", "April 2 2011"),
    ("32", "May 22 1994", "Sunday", "May 29 1994"),
    ("33", "Jun 29 1997", "Monday-Friday", "June 30-4 1997"),
    ("34", "Nov 16 1997", "Monday-Friday", "November 17 to 21 1997"),
]
# The rule's two known misses: published on a Sunday for a Sunday guide, where the
# answer is the FOLLOWING Sunday. Images 2, 10, 26 and 29 are the same shape but have
# the guide date EQUAL to the publication date, so the corpus is genuinely
# inconsistent here and no single rule can satisfy both groups. 30/32 is the ceiling.
KNOWN_MISSES = {"13", "32"}
# Ground truth is a span of days, so no single date can be right.
RANGE_TRUTH = {"28", "33", "34"}


def test_rule_against_every_ground_truth():
    matched, ambiguous, missed = [], [], []
    for image_id, published, dow, truth in GROUND_TRUTH:
        result = compute_guide_date(published, dow)
        if "ambiguous" in result:
            ambiguous.append(image_id)
        elif result["date"] == parse_date(truth).strftime("%Y-%m-%d"):
            matched.append(image_id)
        else:
            missed.append(image_id)
    assert set(ambiguous) == RANGE_TRUTH
    assert set(missed) == KNOWN_MISSES
    # The numbers the demo quotes: 30 of the 32 single-day images.
    assert len(matched) == 30
    assert (len(matched), len(ambiguous), len(missed)) == (30, 3, 2)


def test_rule_is_on_or_after_not_strictly_after():
    # Image 2: published Sunday for a Sunday guide, and the answer is the same day.
    # Switching to "strictly after" would fix images 13/32 but break this and 3 others.
    assert compute_guide_date("Feb 22 2015", "Sunday")["date"] == "2015-02-22"


def test_rule_counts_forward_to_the_named_weekday():
    res = compute_guide_date("Dec 17 2000", "Wednesday")
    assert res["date"] == "2000-12-20"
    assert res["publication_weekday"] == "Sunday"
    assert res["days_after_publication"] == 3


def test_weekday_range_is_ambiguous_not_guessed():
    res = compute_guide_date("Apr 3 2005", "Monday-Friday")
    assert "date" not in res and "spans multiple days" in res["ambiguous"]


def test_unparseable_inputs_are_ambiguous():
    assert "ambiguous" in compute_guide_date("not a date", "Monday")
    assert "ambiguous" in compute_guide_date("Dec 17 2000", "")
    assert "ambiguous" in compute_guide_date("Dec 17 2000", "someday")


class TestParseDate:
    def test_formats_models_actually_emit(self):
        expected = "2015-02-22"
        for value in ("Sun, Feb 22, 2015", "Sunday, February 22, 2015", "2015-02-22",
                      "Feb 22 2015", "02/22/2015", "Sun, Feb. 22, 2015"):
            got = parse_date(value)
            assert got is not None and got.strftime("%Y-%m-%d") == expected, value

    def test_no_year_is_unparseable_rather_than_guessed(self):
        # Resolving these would mean inventing a year.
        assert parse_date("Monday, Oct. 15") is None
        assert parse_date("April 2") is None

    def test_range_is_unparseable(self):
        assert parse_date("Oct. 30-Nov. 5, 1994") is None
        assert parse_date("April 4-8 2005") is None

    def test_empty_and_non_strings(self):
        assert parse_date(None) is None
        assert parse_date("") is None
        assert parse_date("   ") is None


# ── scorer ──────────────────────────────────────────────────────────────────

def correction_run(action, final_value, tool="save_correction"):
    """A fake run whose only tool call is one save_correction."""
    args = json.dumps({"action": action, "final_value": final_value, "reason": "because"})
    msg = SimpleNamespace(tool_calls=[
        SimpleNamespace(id="c1", function=SimpleNamespace(name=tool, arguments=args))
    ])
    return {"messages": [msg], "stopped_reason": "answered"}


TRUTH = "April 29 2010"


def test_metric_eval_runs_are_skipped():
    # No expected_date -> not a date-fix row -> the scorer must stay out of the way.
    assert scorers.date_fix_scorer(correction_run("corrected", "2010-04-29")) is None


def test_fixed_correct_and_fixed_wrong():
    good = scorers.date_fix_scorer(correction_run("corrected", "2010-04-29"),
                                   expected_date=TRUTH, original_value="April 25, 2010")
    assert good["fix_outcome"] == "fixed_correct" and good["regression"] is False
    bad = scorers.date_fix_scorer(correction_run("corrected", "2010-05-06"),
                                  expected_date=TRUTH, original_value="April 25, 2010")
    assert bad["fix_outcome"] == "fixed_wrong"


def test_confirmed_right_and_confirmed_wrong_are_distinguished():
    """A blanket 'confirmed' must not look like success — otherwise an agent that
    rubber-stamps every row would score perfectly."""
    ok = scorers.date_fix_scorer(correction_run("confirmed", "April 29, 2010"),
                                 expected_date=TRUTH, original_value="April 29, 2010")
    assert ok["fix_outcome"] == "confirmed_correct"
    rubber_stamp = scorers.date_fix_scorer(correction_run("confirmed", "April 25, 2010"),
                                           expected_date=TRUTH, original_value="April 25, 2010")
    assert rubber_stamp["fix_outcome"] == "confirmed_wrong"


def test_regression_is_flagged_only_when_it_started_right():
    broke_it = scorers.date_fix_scorer(correction_run("corrected", "2010-05-06"),
                                       expected_date=TRUTH, original_value="April 29, 2010")
    assert broke_it["fix_outcome"] == "fixed_wrong" and broke_it["regression"] is True
    # started wrong, still wrong: bad, but not a regression
    already_wrong = scorers.date_fix_scorer(correction_run("corrected", "2010-05-06"),
                                            expected_date=TRUTH, original_value="April 25, 2010")
    assert already_wrong["regression"] is False


def test_abstained_and_unparseable_truth():
    res = scorers.date_fix_scorer(correction_run("abstained", "April 4-8 2005"),
                                  expected_date="April 4-8 2005",
                                  original_value="April 4-8 2005")
    assert res["fix_outcome"] == "abstained"
    assert res["truth_parseable"] is False
    assert res["regression"] is False


def test_abstaining_is_never_a_regression():
    res = scorers.date_fix_scorer(correction_run("abstained", "April 29, 2010"),
                                  expected_date=TRUTH, original_value="April 29, 2010")
    assert res["fix_outcome"] == "abstained" and res["regression"] is False


def test_no_save_is_no_action():
    res = scorers.date_fix_scorer({"messages": [], "stopped_reason": "answered"},
                                  expected_date=TRUTH, original_value="April 25, 2010")
    assert res["fix_outcome"] == "no_action" and res["regression"] is False


def test_save_evaluation_calls_do_not_count_as_corrections():
    run = correction_run("corrected", "2010-04-29", tool="save_evaluation")
    res = scorers.date_fix_scorer(run, expected_date=TRUTH, original_value="April 25, 2010")
    assert res["fix_outcome"] == "no_action"


def test_corrupt_publication_year_is_ambiguous_not_computed():
    """A model reported a 2002 paper's date as '1902-05-26'. Deriving from that produced
    a confident wrong date that overwrote a CORRECT answer, so an implausible year now
    abstains instead of computing."""
    res = compute_guide_date("1902-05-26", "Sunday")
    assert "date" not in res
    assert "implausible year 1902" in res["ambiguous"]
    # the same inputs with a plausible year still compute normally
    assert compute_guide_date("2002-05-26", "Sunday")["date"] == "2002-05-26"


def test_plausible_boundary_years_still_compute():
    for year in ("1994", "2016"):
        assert "date" in compute_guide_date(f"May 26 {year}", "Sunday")


def test_case_warns_on_corrupt_input_without_using_ground_truth(monkeypatch):
    """The agent is told an implausible publication year is untrustworthy at READ time.

    Truth-free on purpose: the answer key decides which rows enter a graded sample
    (registry/create_date_fix_mapping.py), but telling the agent mid-run that its
    inputs check out would be information it could not have in production.
    """
    from agent_eval import tools

    docs = {
        "1": {"task_id": "n1", "run_id": 0, "benchmark_id": "1", "model_id": "16",
              "image_id": "26", "output": "1902-05-26"},
        "2": {"task_id": "n2", "run_id": 0, "benchmark_id": "2", "model_id": "16",
              "image_id": "26", "output": "Sunday"},
        "3": {"task_id": "n3", "run_id": 0, "benchmark_id": "3", "model_id": "16",
              "image_id": "26", "output": "2002-05-26"},
    }

    class Coll:
        def find_one(self, q):
            return docs["3"]
        def find(self, q, proj=None):
            return list(docs.values())

    monkeypatch.setattr(tools, "get_db", lambda: {"llm_outputs": Coll()})
    case = tools.get_guide_date_case("n3", 0)
    assert "implausible year (1902)" in case["input_warning"]
    assert 'action="abstained"' in case["input_warning"]
    # the values themselves are still served, and no ground truth leaks in
    assert case["tv_guide_date"] == "2002-05-26"
    assert not any("expected" in k or "truth" in k for k in case)


def test_case_has_no_warning_for_a_plausible_year(monkeypatch):
    from agent_eval import tools

    docs = {
        "1": {"task_id": "n1", "run_id": 0, "benchmark_id": "1", "model_id": "16",
              "image_id": "26", "output": "2002-05-26"},
        "3": {"task_id": "n3", "run_id": 0, "benchmark_id": "3", "model_id": "16",
              "image_id": "26", "output": "2002-05-26"},
    }

    class Coll:
        def find_one(self, q):
            return docs["3"]
        def find(self, q, proj=None):
            return list(docs.values())

    monkeypatch.setattr(tools, "get_db", lambda: {"llm_outputs": Coll()})
    assert "input_warning" not in tools.get_guide_date_case("n3", 0)


def test_same_weekday_case_is_flagged_for_review_not_silently_guessed():
    """Offset 0 means the paper was published on the guide's own weekday, and the corpus
    is split on what that means (4 images same-day, 2 images a week later). The rule
    still answers, but says out loud that it is the likelier of two readings."""
    res = compute_guide_date("Feb 22 2015", "Sunday")
    assert res["date"] == "2015-02-22"
    assert res["needs_review"] is True
    assert "same weekday" in res["review_reason"]


def test_unambiguous_case_is_not_flagged():
    res = compute_guide_date("Dec 17 2000", "Wednesday")
    assert res["days_after_publication"] == 3
    assert "needs_review" not in res


def test_scorer_records_the_agents_review_flag():
    import json as _json
    args = _json.dumps({"action": "corrected", "final_value": "2010-04-29",
                        "reason": "r", "needs_review": True,
                        "review_reason": "published on the same weekday"})
    msg = SimpleNamespace(tool_calls=[
        SimpleNamespace(id="c1", function=SimpleNamespace(name="save_correction", arguments=args))])
    res = scorers.date_fix_scorer({"messages": [msg], "stopped_reason": "answered"},
                                  expected_date=TRUTH, original_value="April 25, 2010")
    assert res["fix_outcome"] == "fixed_correct" and res["needs_review"] is True
    # and an unflagged row reads False rather than None
    assert scorers.date_fix_scorer(correction_run("corrected", "2010-04-29"),
                                   expected_date=TRUTH,
                                   original_value="April 25, 2010")["needs_review"] is False
