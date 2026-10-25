import numpy as np

from uma import locate
from uma.models import Token
from uma.table_regions import (
    TableProposal,
    find_table,
    label_groups,
    merge_segments,
)


def test_reliable_table_proposal_flows_into_locate_result(monkeypatch):
    expected = (40, 50, 150, 110)

    def fake_find_table(plain, tokens):
        return TableProposal(
            bbox=expected,
            reliable=True,
            method="local_frame",
            detail={"groups": ["date", "drawn", "scale"]},
        )

    monkeypatch.setattr(locate, "find_table", fake_find_table)

    result = locate.locate(
        np.full((160, 220), 255, dtype=np.uint8),
        [Token("DATE", (50, 80, 80, 90), 0.9)],
    )

    assert result.bbox == expected
    assert result.method == "local_frame"
    assert result.confidence == 0.85
    assert result.detail["review_required"] is False


def test_localization_uses_lower_right_fallback_when_evidence_is_missing(
    monkeypatch,
):
    monkeypatch.setattr(
        locate,
        "find_table",
        lambda plain, tokens: TableProposal(
            bbox=None,
            reliable=False,
            method="none",
            detail={"reason": "no evidence"},
        ),
    )

    result = locate.locate(
        np.full((100, 160), 255, dtype=np.uint8),
        [],
    )

    assert result.method == "fallback"
    assert result.bbox == (96, 55, 160, 100)
    assert result.detail["degraded"] is True


def test_title_block_labels_are_grouped_by_metadata_role():
    tokens = [
        Token("DRAWN", (10, 10, 55, 25), 0.9),
        Token("CHECKED", (60, 10, 120, 25), 0.9),
        Token("SCALE", (10, 35, 55, 50), 0.9),
        Token("DATE", (60, 35, 100, 50), 0.9),
    ]

    groups = label_groups(tokens)

    assert {"drawn", "checked", "scale", "date"} <= groups


def test_nearby_horizontal_segments_are_merged():
    lines = [
        (10, 20, 80, 20),
        (84, 20, 150, 20),
    ]

    merged = merge_segments(lines, horizontal=True, max_gap=10)

    assert merged == [(10, 20, 150, 20)]


def test_table_detection_returns_a_safe_failure_without_evidence():
    image = np.full((120, 160), 255, dtype=np.uint8)

    proposal = find_table(image, [])

    assert isinstance(proposal, TableProposal)
    assert proposal.reliable is False
    assert proposal.bbox is None
