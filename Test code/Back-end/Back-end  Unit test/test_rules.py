"""Unit tests for the current uma.rules module.

Designed for pytest on Python 3.11+. Use mocks only at the boundaries of
rule extraction, allowing matcher and spatial logic to run directly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

import uma.models as models
import uma.patterns as patterns
import uma.rules as rules
import uma.verify as verify


@dataclass(eq=False)
class FakeToken:
    """Small, geometry-compatible stand-in for an OCR token."""

    tid: int
    text: str
    x0: float = 0
    y0: float = 0
    w: float = 30
    h: float = 10
    agreement: float = 1.0
    conf: float = 0.95
    alt: list = field(default_factory=list)
    notes_hint: str | None = None

    @property
    def x1(self):
        return self.x0 + self.w

    @property
    def cy(self):
        return self.y0 + self.h / 2

    @property
    def cx(self):
        return self.x0 + self.w / 2

    def alternatives(self):
        return self.alt


def token(tid, text, x=0, y=0, w=30, h=10, **kwargs):
    return FakeToken(tid, text, x0=x, y0=y, w=w, h=h, **kwargs)


def cfg(threshold=0.80, architects=(), years=(1900, 2100), anchor_fuzz=85):
    return SimpleNamespace(
        lock_threshold=threshold,
        architects=list(architects),
        year_range=years,
        anchor_fuzz=anchor_fuzz,
    )


def alt(text, engine="fallback", conf=0.891234):
    return SimpleNamespace(text=text, engine=engine, conf=conf)


# ---------------------------------------------------------------------------
# Input filtering, normalisation and alternative OCR candidates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value, expected",
    [
        ("PROJECT:", True),
        ("SCALE 1:100", True),
        ("  BUILDING PLAN  ", True),
        ("ABCD", True),
        ("ABC", False),
        ("A" * 41, False),
        ("ROOM 123456789", False),
        ("mixed text", False),
        ("", False),
    ],
)
def test_keep_line(monkeypatch, value, expected):
    # Pin the two domain recognisers; other paths use actual uppercase/ratio logic.
    monkeypatch.setattr(patterns, "CAPTION_RE", re.compile(r"PROJECT:"))
    monkeypatch.setattr(rules, "SCALE_RE", re.compile(r"1:\d+"))
    assert rules.keep_line(SimpleNamespace(text=value)) is expected


@pytest.mark.parametrize(
    "value, expected",
    [("  a  b\n c ", "A B C"), ("", ""), (None, "")],
)
def test_norm(value, expected):
    assert rules._norm(value) == expected


def test_analyze_candidates_without_alternatives():
    assert rules.analyze_candidates(token(1, "ABC")) is None


def test_analyze_candidates_no_improvement(monkeypatch):
    monkeypatch.setattr(rules, "confusion_variants", lambda *_: [])
    t = token(1, "ABC", alt=[alt("ABC"), alt("XYZ")])
    assert rules.analyze_candidates(t, "scale") is None


def test_analyze_candidates_prefers_more_complete_alternative():
    t = token(1, "MAIN", alt=[alt("MAIN BUILDING", "engine_A", 0.83556)])
    answer = rules.analyze_candidates(t)
    assert answer["preferred"] == "MAIN BUILDING"
    assert answer["engine"] == "engine_A"
    assert answer["conf"] == 0.836
    assert any("superset" in message for message in answer["reasons"])


def test_analyze_candidates_expected_format(monkeypatch):
    monkeypatch.setitem(rules.FIELD_PATTERNS, "drawing_number", re.compile(r"D-\d{3}"))
    monkeypatch.setattr(rules, "confusion_variants", lambda *_: [])
    t = token(1, "unreadable", alt=[alt("D-123")])
    answer = rules.analyze_candidates(t, "drawing_number")
    assert answer["preferred"] == "D-123"
    assert any("expected drawing_number format" in r for r in answer["reasons"])


def test_analyze_candidates_ocr_confusions(monkeypatch):
    monkeypatch.setitem(rules.FIELD_PATTERNS, "scale", re.compile(r"1:\d+"))
    monkeypatch.setattr(rules, "confusion_variants", lambda value, name: ["1:100"])
    t = token(1, "?", alt=[alt("I:1OO")])
    assert "OCR confusions" in " ".join(rules.analyze_candidates(t, "scale")["reasons"])


def test_analyze_candidates_checks_multiple_confusion_variants(monkeypatch):
    monkeypatch.setitem(rules.FIELD_PATTERNS, "scale", re.compile(r"1:\d+"))
    monkeypatch.setattr(rules, "confusion_variants", lambda *_: ["INVALID", "1:100"])
    t = token(1, "unknown", alt=[alt("I:1OO")])
    answer = rules.analyze_candidates(t, "scale")
    assert answer["preferred"] == "I:1OO"
    assert "1:100" in " ".join(answer["reasons"])


def test_analyze_candidates_prefers_candidate_with_more_evidence(monkeypatch):
    monkeypatch.setitem(rules.FIELD_PATTERNS, "scale", re.compile(r"1:\d+"))
    monkeypatch.setattr(rules, "confusion_variants", lambda *_: [])
    t = token(1, "1:", alt=[alt("1:150", "best"), alt("1:125 unrelated", "other")])
    answer = rules.analyze_candidates(t, "scale")
    assert answer["engine"] == "best"  # Equal evidence keeps the first.
    assert len(answer["reasons"]) == 2


# ---------------------------------------------------------------------------
# Spatial grouping and primitive matchers
# ---------------------------------------------------------------------------

def test_extend_row_collects_consecutive_tokens_only():
    a, b, c = token(1, "A", 0), token(2, "B", 35), token(3, "C", 70)
    off_row = token(4, "D", 50, 40)
    assert rules.extend_row(a, [c, off_row, b, a]) == [a, b, c]
    assert rules.extend_row(b, [a, b, c]) == [b, c]  # Never extends left.


@pytest.mark.parametrize("stop", ["SCALE:", "SCALES", "DATE", "DRAWING", "JOB", "SHEET", "CHECKED"])
def test_extend_row_stops_at_labels_and_anchors(stop):
    seed, boundary, after = token(1, "hello", 0), token(2, stop, 35), token(3, "ignored", 70)
    assert rules.extend_row(seed, [seed, boundary, after]) == [seed]


def test_extend_row_stops_at_distant_token():
    seed, near, far = token(1, "A", 0), token(2, "B", 35), token(3, "C", 250)
    assert rules.extend_row(seed, [seed, near, far]) == [seed, near]


def test_extend_row_seed_not_in_tokens():
    seed = token(1, "A", 0)
    assert rules.extend_row(seed, [token(2, "B", 35)]) == [seed]


def test_extend_row_zero_height():
    a, b = token(1, "A", 0, h=0), token(2, "B", 31, h=0)
    assert rules.extend_row(a, [a, b]) == [a, b]


def test_match_scale_clean(monkeypatch):
    monkeypatch.setattr(rules, "SCALE_RE", re.compile(r"1:\d+"))
    label, item = token(1, "SCALE"), token(2, "1:100", 50)
    group, score, trimmed = rules.match_scale([label, item])
    assert group == [item]
    assert score == 0.94
    assert trimmed == "1:100"


def test_match_scale_noisy_extended_row(monkeypatch):
    monkeypatch.setattr(rules, "SCALE_RE", re.compile(r"1:\d+"))
    a, b = token(1, "1:200"), token(2, "NOT TO SCALE", 35, w=90)
    group, score, trimmed = rules.match_scale([a, b])
    assert group == [a, b]
    assert score == 0.72
    assert trimmed == "1:200"


def test_match_scale_not_found(monkeypatch):
    monkeypatch.setattr(rules, "SCALE_RE", re.compile(r"1:\d+"))
    assert rules.match_scale([token(1, "SCALE"), token(2, "hello")]) == ([], 0.0, None)


def test_match_scale_recogniser_not_matching_joined_group(monkeypatch):
    # Force the unusual fallback (pattern matches a token, but not joined text).
    class OneShotPattern:
        def __init__(self):
            self.count = 0

        def search(self, value):
            self.count += 1
            return re.search(r"1:100", value) if self.count == 1 else None

    monkeypatch.setattr(rules, "SCALE_RE", OneShotPattern())
    group, score, trimmed = rules.match_scale([token(1, "1:100")])
    assert len(group) == 1
    assert trimmed is None
    assert score == 0.94


def test_match_date_skips_unmatched_unparseable_and_outside_years(monkeypatch):
    monkeypatch.setattr(rules, "DATE_RE", re.compile(r"\d{2}/\d{2}(?:/\d{4})?"))
    monkeypatch.setattr(rules, "YEAR_RE", re.compile(r"(\d{4})"))
    good = token(4, "05/06/2020")
    items = [token(1, "N/A"), token(2, "05/06"), token(3, "05/06/1840"), good]
    assert rules.match_date(items, cfg(years=(1900, 2030))) == (good, 0.89)
    assert rules.match_date(items, cfg(years=(2022, 2030))) == (None, 0.0)


def test_match_architect_returns_best_matching_name():
    items = [token(1, "ACME"), token(2, "ARCHITECTS", 35)]
    hit, score, name = rules.match_architect(items, cfg(architects=["OTHER FIRM", "ACME ARCHITECTS"]))
    assert hit == [items[0]]
    assert score == 0.96
    assert name == "Acme Architects"


def test_match_architect_without_sufficient_match():
    assert rules.match_architect([token(1, "12345")], cfg(architects=["MELBOURNE ARCHITECTS"])) == ([], 0.0, None)
    assert rules.match_architect([], cfg()) == ([], 0.0, None)


def test_match_architect_without_first_word_hit(monkeypatch):
    monkeypatch.setattr(rules.fuzz, "partial_ratio", lambda *_: 93)
    hit, score, name = rules.match_architect([token(1, "A DIFFERENT NAME")], cfg(architects=["ACME ARCHITECTS"]))
    assert hit == []
    assert score == 0.93
    assert name == "Acme Architects"


def test_match_drawing_no_selects_agreement_then_confidence(monkeypatch):
    monkeypatch.setattr(rules, "DRAWING_NO_RE", re.compile(r"D-\d{3}"))
    low = token(1, "D-100", agreement=0.85, conf=0.99)
    high = token(2, "D-200", agreement=1.0, conf=0.80)
    higher_conf = token(3, "D-300", agreement=1.0, conf=0.95)
    assert rules.match_drawing_no([low, high, higher_conf]) == (higher_conf, 0.88)
    assert rules.match_drawing_no([low]) == (low, 0.52)


def test_match_drawing_no_uses_fallback_candidate(monkeypatch):
    monkeypatch.setattr(rules, "DRAWING_NO_RE", re.compile(r"D-\d{3}"))
    t = token(1, "D-I0O")
    monkeypatch.setattr(rules, "scan_candidates", lambda _: (alt("D-100", "engine2"), None))
    hit, score = rules.match_drawing_no([t])
    assert hit is t and score == 0.60
    assert "engine2" in t.notes_hint


def test_match_drawing_no_no_valid_candidates(monkeypatch):
    monkeypatch.setattr(rules, "DRAWING_NO_RE", re.compile(r"D-\d{3}"))
    a, b = token(1, "UNKNOWN"), token(2, "BAD")
    monkeypatch.setattr(rules, "scan_candidates", lambda t: ((None, None) if t is a else (alt("NO MATCH"), None)))
    assert rules.match_drawing_no([a, b]) == (None, 0.0)
    assert rules.match_drawing_no([]) == (None, 0.0)


@pytest.mark.parametrize(
    "value, expected",
    [(" ab 12. ", "AB12"), (" a\t b,;:", "AB"), ("", ""), (None, "")],
)
def test_drawing_no_key(value, expected):
    assert rules._drawing_no_key(value) == expected


def test_find_anchors_strips_suffix_and_uses_fuzzy_matching(monkeypatch):
    monkeypatch.setattr(rules, "ANCHORS", {"date": ["DATE"], "scale": ["SCALE"]})
    date, scale, off = token(1, "date:"), token(2, "SCALE."), token(3, "HELLO")
    out = rules.find_anchors([date, scale, off], fuzz_threshold=95)
    assert out == {"date": date, "scale": scale}
    assert rules.find_anchors([off], fuzz_threshold=95) == {}


def test_find_anchors_choose_higher_similarity(monkeypatch):
    monkeypatch.setattr(rules, "ANCHORS", {"date": ["DATE"]})
    almost, exact = token(1, "DAT"), token(2, "DATE")
    assert rules.find_anchors([almost, exact], 80)["date"] is exact
    assert rules.find_anchors([], 80) == {}


# ---------------------------------------------------------------------------
# Spatial scoring and Hungarian global assignment
# ---------------------------------------------------------------------------

def test_pair_score_same_object():
    item = token(1, "DATE")
    assert rules.pair_score(item, item) == 0.0


def test_pair_score_right_side_distance():
    anchor, value = token(1, "DATE", x=0, w=30), token(2, "2020", x=40)
    assert rules.pair_score(anchor, value) == pytest.approx(1 - 10 / 300)
    assert rules.pair_score(anchor, token(3, "far", x=500)) == 0.0


def test_pair_score_below_and_disallowed_positions():
    anchor = token(1, "DATE", x=0, y=0, w=30, h=10)
    assert rules.pair_score(anchor, token(2, "2020", x=0, y=12)) == pytest.approx(0.5)
    assert rules.pair_score(anchor, token(3, "2020", x=150, y=12)) == 0.0
    assert rules.pair_score(anchor, token(4, "2020", x=-50, y=0)) == 0.0
    assert rules.pair_score(anchor, token(5, "2020", x=0, y=50)) == 0.0


def test_assign_globally_no_anchors_or_candidates():
    a = token(1, "DATE")
    assert rules.assign_globally({}, [a]) == {}
    assert rules.assign_globally({"date": a}, [a]) == {}


def test_assign_globally_assigns_nearby_values_without_reuse():
    a1, a2 = token(1, "DATE", x=0, y=0), token(2, "SCALE", x=0, y=40)
    v1, v2 = token(3, "2020", x=35, y=0), token(4, "1:100", x=35, y=40)
    answer = rules.assign_globally({"date": a1, "scale": a2}, [a1, a2, v1, v2])
    assert answer["date"][0] is v1
    assert answer["scale"][0] is v2
    assert len({id(pair[0]) for pair in answer.values()}) == 2


def test_assign_globally_respects_field_specific_threshold(monkeypatch):
    monkeypatch.setattr(rules, "pair_score", lambda *_: 0.33)
    a, b = token(1, "DRAWING NUMBER"), token(2, "X")
    result = rules.assign_globally({"drawing_number": a}, [a, b])
    assert result["drawing_number"][0] is b
    assert rules.assign_globally({"project_name": a}, [a, b]) == {}


def test_assign_globally_score_on_boundary_is_not_enough(monkeypatch):
    monkeypatch.setattr(rules, "pair_score", lambda *_: 0.35)
    a, b = token(1, "PROJECT"), token(2, "ABC")
    assert rules.assign_globally({"project_name": a}, [a, b]) == {}


def test_assign_globally_optimises_joint_score(monkeypatch):
    a1, a2, v1, v2 = [token(i, str(i)) for i in range(4)]
    scores = {(id(a1), id(v1)): 0.90, (id(a1), id(v2)): 0.85,
              (id(a2), id(v1)): 0.83, (id(a2), id(v2)): 0.37}
    monkeypatch.setattr(rules, "pair_score", lambda a, b: scores.get((id(a), id(b)), 0.0))
    result = rules.assign_globally({"project_name": a1, "date": a2}, [a1, a2, v1, v2])
    assert result["project_name"][0] is v2
    assert result["date"][0] is v1


@pytest.mark.parametrize(
    "title, expected",
    [
        ("North Elevation", "elevation"),
        ("floor plan", "plan"),
        ("section A", "section"),
        ("Detail 02", "detail"),
        ("Perspective View", "perspective"),
        ("Plan and Section", "mixed"),
        ("Site Description", None),
        ("", None),
        (None, None),
    ],
)
def test_infer_drawing_type(title, expected):
    assert rules.infer_drawing_type(title) == expected


# ---------------------------------------------------------------------------
# extract(): orchestration of the matchers, validation, locks and deferrals
# ---------------------------------------------------------------------------
@pytest.fixture
def extraction_env(monkeypatch):
    """Isolate the orchestration while testing each matcher directly above."""

    class FakeFieldValue:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    # extract uses models.FieldValue for probes and rules.FieldValue for output.
    monkeypatch.setattr(models, "FieldValue", FakeFieldValue)
    monkeypatch.setattr(rules, "FieldValue", FakeFieldValue)
    monkeypatch.setattr(verify, "format_check", lambda _: None)
    monkeypatch.setattr(rules, "match_scale", lambda _: ([], 0.0, None))
    monkeypatch.setattr(rules, "match_date", lambda *_: (None, 0.0))
    monkeypatch.setattr(rules, "match_architect", lambda *_: ([], 0.0, None))
    monkeypatch.setattr(rules, "match_drawing_no", lambda _: (None, 0.0))
    monkeypatch.setattr(rules, "find_anchors", lambda *_: {})
    monkeypatch.setattr(rules, "assign_globally", lambda *_: {})
    return monkeypatch


def test_extract_no_matches_returns_all_pending(extraction_env):
    locked, pending, rejected = rules.extract([], cfg())
    assert locked == {}
    assert pending == rules.FIELDS
    assert rejected == {}


def test_extract_scale_clean_lock_with_token_evidence(extraction_env):
    a, b = token(1, "1:100", agreement=0.85, conf=0.75), token(2, "NOTE", agreement=0.93, conf=0.50)
    extraction_env.setattr(rules, "match_scale", lambda _: ([a, b], 0.94, "1:100"))
    locked, pending, rejected = rules.extract([a, b], cfg())
    field = locked["scale"]
    assert field.verbatim == "1:100 NOTE"
    assert field.corrected == "1:100"
    assert field.correction_basis is not None
    assert field.tokens == [1, 2]
    assert field.confidence == 0.94
    assert field.signals["agreement"] == 0.85
    assert field.signals["ocr_conf"] == 0.50
    assert field.source == "rule" and field.status == "locked"
    assert "scale" not in pending and rejected == {}


def test_extract_scale_noisy_value_is_deferred(extraction_env, caplog):
    a = token(1, "1:100 NOT TO SCALE")
    extraction_env.setattr(rules, "match_scale", lambda _: ([a], 0.72, "1:100"))
    with caplog.at_level("INFO", logger=rules.__name__):
        locked, pending, rejected = rules.extract([a], cfg())
    assert "scale" not in locked
    assert "scale" in pending
    assert "non-scale content" in rejected["scale"]
    assert "deferred scale" in caplog.text


def test_extract_date_accepts_validation_score_none(extraction_env):
    d = token(1, "12/05/2020")
    extraction_env.setattr(rules, "match_date", lambda *_: (d, 0.89))
    locked, pending, rejected = rules.extract([d], cfg())
    assert locked["date"].verbatim == "12/05/2020"
    assert locked["date"].corrected == "12/05/2020"
    assert locked["date"].correction_basis is None
    assert "date" not in pending and rejected == {}


def test_extract_date_accepts_perfect_validation_score(extraction_env):
    d = token(1, "12/05/2020")
    extraction_env.setattr(rules, "match_date", lambda *_: (d, 0.89))
    extraction_env.setattr(verify, "format_check", lambda _: 1.0)
    assert "date" in rules.extract([d], cfg())[0]


def test_extract_invalid_date_gate_rejects_and_logs(extraction_env, caplog):
    d = token(1, "bad-date")
    extraction_env.setattr(rules, "match_date", lambda *_: (d, 0.89))
    extraction_env.setattr(verify, "format_check", lambda _: 0.2)
    with caplog.at_level("INFO", logger=rules.__name__):
        locked, pending, rejected = rules.extract([d], cfg())
    assert "date" not in locked and "date" in pending
    assert "does not match" in rejected["date"]
    assert "deferred date" in caplog.text


def test_extract_date_below_lock_threshold(extraction_env):
    d = token(1, "2020")
    extraction_env.setattr(rules, "match_date", lambda *_: (d, 0.79))
    assert "date" not in rules.extract([d], cfg())[0]


def test_extract_architect_rejects_large_expansion(extraction_env):
    t = token(1, "ACME")
    extraction_env.setattr(rules, "match_architect", lambda *_: ([t], 0.96, "ACME VERY LONG ARCHITECTURE GROUP"))
    locked, pending, rejected = rules.extract([t], cfg())
    assert "architect" not in locked
    assert "architect" in pending
    assert "expanding" in rejected["architect"]


def test_extract_architect_caps_moderate_expansion(extraction_env):
    t = token(1, "ARCHITECTS")
    extraction_env.setattr(rules, "match_architect", lambda *_: ([t], 0.96, "ARCHITECTS PLUS"))
    locked, pending, rejected = rules.extract([t], cfg())
    assert locked["architect"].corrected == "ARCHITECTS PLUS"
    assert locked["architect"].confidence == 0.90
    assert locked["architect"].correction_basis is not None
    assert "architect" not in pending and rejected == {}


def test_extract_architect_full_read_and_below_threshold(extraction_env):
    t = token(1, "ACME ARCHITECTS")
    extraction_env.setattr(rules, "match_architect", lambda *_: ([t], 0.91, "Acme Architects"))
    assert rules.extract([t], cfg())[0]["architect"].confidence == 0.91
    assert "architect" not in rules.extract([t], cfg(threshold=0.96))[0]


def test_extract_value_only_drawing_number(extraction_env):
    d = token(1, "A-101")
    extraction_env.setattr(rules, "match_drawing_no", lambda _: (d, 0.88))
    locked, pending, rejected = rules.extract([d], cfg())
    assert locked["drawing_number"].tokens == [1]
    assert "drawing_number" not in pending and rejected == {}
    assert "drawing_number" not in rules.extract([d], cfg(threshold=0.90))[0]


def test_extract_anchor_only_drawing_number(extraction_env):
    d = token(1, "A-101")
    extraction_env.setattr(rules, "assign_globally", lambda *_: {"drawing_number": (d, 0.85)})
    locked, pending, rejected = rules.extract([d], cfg())
    assert locked["drawing_number"].verbatim == "A-101"
    assert locked["drawing_number"].confidence == round(0.80 + 0.15 * 0.85, 3)
    assert "drawing_number" not in pending and rejected == {}
    assert "drawing_number" not in rules.extract([d], cfg(threshold=0.95))[0]


def test_extract_matching_drawing_number_evidence(extraction_env):
    value = token(1, "a -101.")
    anchor_value = token(2, "A-101")
    extraction_env.setattr(rules, "match_drawing_no", lambda _: (value, 0.52))
    extraction_env.setattr(rules, "assign_globally", lambda *_: {"drawing_number": (anchor_value, 0.9)})
    locked, pending, rejected = rules.extract([value, anchor_value], cfg())
    assert locked["drawing_number"].tokens == [2]
    assert locked["drawing_number"].confidence == 0.935
    assert rejected == {}
    assert "drawing_number" not in pending
    assert "drawing_number" not in rules.extract([value, anchor_value], cfg(threshold=0.95))[0]


def test_extract_conflicting_drawing_number_evidence(extraction_env):
    a, b = token(1, "A-101"), token(2, "B-202")
    extraction_env.setattr(rules, "match_drawing_no", lambda _: (a, 0.88))
    extraction_env.setattr(rules, "assign_globally", lambda *_: {"drawing_number": (b, 0.8)})
    locked, pending, rejected = rules.extract([a, b], cfg())
    assert "drawing_number" not in locked
    assert "drawing_number" in pending
    assert "Evidence conflict" in rejected["drawing_number"]
    assert "A-101" in rejected["drawing_number"]
    assert "B-202" in rejected["drawing_number"]


def test_extract_paired_fields_skip_unknown_or_already_locked(extraction_env):
    date, other = token(1, "2020"), token(2, "NONE")
    extraction_env.setattr(rules, "match_date", lambda *_: (date, 0.89))
    extraction_env.setattr(rules, "assign_globally", lambda *_: {
        "date": (other, 0.99), "unknown_metadata": (other, 0.99)
    })
    locked, pending, rejected = rules.extract([date, other], cfg())
    assert locked["date"].tokens == [1]
    assert "unknown_metadata" not in locked
    assert rejected == {}


def test_extract_paired_multitoken_project(extraction_env):
    a, b = token(1, "BUILDING", x=0), token(2, "A", x=35)
    extraction_env.setattr(rules, "assign_globally", lambda *_: {"project_name": (a, 0.95)})
    locked, pending, rejected = rules.extract([a, b], cfg())
    assert locked["project_name"].tokens == [1, 2]
    assert locked["project_name"].verbatim == "BUILDING A"
    assert locked["project_name"].correction_basis is None
    assert locked["project_name"].corrected is None
    assert "project_name" not in pending and rejected == {}


def test_extract_paired_single_token_draughtsperson(extraction_env):
    a, b = token(1, "ALICE", x=0), token(2, "SMITH", x=35)
    extraction_env.setattr(rules, "assign_globally", lambda *_: {"draughtsperson": (a, 0.93)})
    locked, _, _ = rules.extract([a, b], cfg())
    assert locked["draughtsperson"].tokens == [1]
    assert locked["draughtsperson"].verbatim == "ALICE"
    assert "draughtsperson" not in rules.extract([a, b], cfg(threshold=0.99))[0]


@pytest.mark.parametrize("title, expected", [
    ("NORTH ELEVATION", "elevation"),
    ("PLAN AND SECTION", "mixed"),
    ("PROJECT OVERVIEW", None),
])
def test_extract_classifies_drawing_title(extraction_env, title, expected):
    d = token(1, title)
    extraction_env.setattr(rules, "assign_globally", lambda *_: {"drawing_title": (d, 0.99)})
    locked, pending, rejected = rules.extract([d], cfg())
    assert locked["drawing_title"].verbatim == title
    if expected is not None:
        field = locked["drawing_type"]
        assert field.corrected == expected
        assert field.confidence == 0.88
        assert field.tokens == [1]
        assert field.signals["inferred_from"] == "drawing_title"
        assert field.notes
        assert "drawing_type" not in pending
    else:
        assert "drawing_type" not in locked
        assert "drawing_type" in pending
    assert rejected == {}


def test_extract_rejected_scale_and_date_at_once(extraction_env):
    s, d = token(1, "1:100 and annotation"), token(2, "bad date")
    extraction_env.setattr(rules, "match_scale", lambda _: ([s], 0.72, "1:100"))
    extraction_env.setattr(rules, "match_date", lambda *_: (d, 0.89))
    extraction_env.setattr(verify, "format_check", lambda _: 0)
    locked, pending, rejected = rules.extract([s, d], cfg())
    assert locked == {}
    assert set(rejected) == {"scale", "date"}
    assert len(pending) == len(rules.FIELDS)
