from types import SimpleNamespace

import pytest

from uma.unimelb_result_codec import (
    OCR_RESULT_FIELDS,
    OCR_REASON_CODES,
    note_values,
    split_reason_and_notes,
    evidence_regions_bbox,
    tokens_bbox,
    extract_ocr_result,
)


# ============================================================
# Helpers
# ============================================================

def make_field(
    *,
    tokens=None,
    verbatim=None,
    corrected=None,
    confidence=None,
    correction_basis=None,
    source=None,
    signals=None,
    reason=None,
    notes=None,
    status=None,
):
    return SimpleNamespace(
        tokens=[] if tokens is None else tokens,
        verbatim=verbatim,
        corrected=corrected,
        confidence=confidence,
        correction_basis=correction_basis,
        source=source,
        signals={} if signals is None else signals,
        reason=reason,
        notes=notes,
        status=status,
    )


def make_record(
    fields=None,
    token_map=None,
):
    return SimpleNamespace(
        fields={} if fields is None else fields,
        token_map={} if token_map is None else token_map,
    )


# ============================================================
# Constants
# ============================================================

def test_ocr_result_fields():
    assert OCR_RESULT_FIELDS == (
        "drawing_number",
        "project_name",
        "drawing_title",
        "architect",
        "draughtsperson",
        "date",
        "scale",
        "drawing_type",
    )


def test_reason_codes():
    assert OCR_REASON_CODES == frozenset({
        "illegible",
        "not_present",
        "ambiguous",
        "low_resolution",
        "requires_visual",
    })


# ============================================================
# note_values
# ============================================================

def test_note_values_empty():
    assert note_values() == []


def test_note_values_none():
    assert note_values(None) == []


def test_note_values_single_string():
    assert note_values(
        "hello"
    ) == ["hello"]


def test_note_values_strips_whitespace():
    assert note_values(
        "   hello world   "
    ) == ["hello world"]


def test_note_values_ignores_empty_string():
    assert note_values(
        "",
        "   ",
        None,
    ) == []


def test_note_values_multiple_values():
    assert note_values(
        "first",
        "second",
        "third",
    ) == [
        "first",
        "second",
        "third",
    ]


def test_note_values_list():
    assert note_values(
        ["first", "second"]
    ) == [
        "first",
        "second",
    ]


def test_note_values_tuple():
    assert note_values(
        ("first", "second")
    ) == [
        "first",
        "second",
    ]


def test_note_values_nested():
    result = note_values(
        [
            "first",
            [
                "second",
                (
                    "third",
                    None,
                ),
            ],
        ]
    )

    assert result == [
        "first",
        "second",
        "third",
    ]


def test_note_values_removes_duplicates():
    result = note_values(
        "hello",
        "hello",
        ["hello", "world"],
        ("world",),
    )

    assert result == [
        "hello",
        "world",
    ]


def test_note_values_converts_values_to_string():
    result = note_values(
        123,
        4.5,
        True,
    )

    assert result == [
        "123",
        "4.5",
        "True",
    ]


# ============================================================
# split_reason_and_notes
# ============================================================

def test_split_reason_none():
    reason, notes = split_reason_and_notes(
        None,
        None,
    )

    assert reason is None
    assert notes == []


def test_split_reason_scalar_known():
    reason, notes = split_reason_and_notes(
        "illegible",
        None,
    )

    assert reason == "illegible"
    assert notes == []


def test_split_reason_scalar_unknown_preserved():
    reason, notes = split_reason_and_notes(
        "custom_reason",
        None,
    )

    assert reason == "custom_reason"
    assert notes == []


def test_split_reason_scalar_stripped():
    reason, notes = split_reason_and_notes(
        "  ambiguous  ",
        None,
    )

    assert reason == "ambiguous"
    assert notes == []


def test_split_reason_empty_scalar():
    reason, notes = split_reason_and_notes(
        "   ",
        None,
    )

    assert reason is None
    assert notes == []


def test_split_reason_with_notes():
    reason, notes = split_reason_and_notes(
        "illegible",
        "needs review",
    )

    assert reason == "illegible"
    assert notes == ["needs review"]


def test_split_reason_collection_known_code():
    reason, notes = split_reason_and_notes(
        [
            "illegible",
            "extra explanation",
        ],
        None,
    )

    assert reason == "illegible"

    assert notes == [
        "extra explanation"
    ]


def test_split_reason_collection_first_known_code():
    reason, notes = split_reason_and_notes(
        [
            "not a code",
            "low_resolution",
            "ambiguous",
        ],
        None,
    )

    assert reason == "low_resolution"

    # Once a reason has been found,
    # remaining entries become notes.
    assert notes == [
        "not a code",
        "ambiguous",
    ]


def test_split_reason_collection_no_known_code():
    reason, notes = split_reason_and_notes(
        [
            "note one",
            "note two",
        ],
        None,
    )

    assert reason is None

    assert notes == [
        "note one",
        "note two",
    ]


def test_split_reason_collection_merges_existing_notes():
    reason, notes = split_reason_and_notes(
        [
            "ambiguous",
            "manual review",
        ],
        [
            "existing note",
        ],
    )

    assert reason == "ambiguous"

    assert notes == [
        "existing note",
        "manual review",
    ]


def test_split_reason_collection_deduplicates_notes():
    reason, notes = split_reason_and_notes(
        [
            "ambiguous",
            "same note",
        ],
        [
            "same note",
        ],
    )

    assert reason == "ambiguous"
    assert notes == ["same note"]


def test_split_reason_tuple():
    reason, notes = split_reason_and_notes(
        (
            "requires_visual",
            "check drawing",
        ),
        None,
    )

    assert reason == "requires_visual"
    assert notes == ["check drawing"]


def test_split_reason_set():
    reason, notes = split_reason_and_notes(
        {"illegible"},
        None,
    )

    assert reason == "illegible"
    assert notes == []


# ============================================================
# evidence_regions_bbox
# ============================================================

def test_evidence_regions_bbox_non_dict():
    assert evidence_regions_bbox(
        None
    ) is None

    assert evidence_regions_bbox(
        []
    ) is None

    assert evidence_regions_bbox(
        "invalid"
    ) is None


def test_evidence_regions_bbox_single():
    signals = {
        "evidence_regions": [
            {
                "bbox": [
                    10,
                    20,
                    100,
                    80,
                ]
            }
        ]
    }

    result = evidence_regions_bbox(
        signals
    )

    assert result == {
        "x0": 10,
        "y0": 20,
        "x1": 100,
        "y1": 80,
    }


def test_evidence_regions_bbox_multiple():
    signals = {
        "evidence_regions": [
            {
                "bbox": [
                    10,
                    20,
                    100,
                    80,
                ]
            },
            {
                "bbox": [
                    5,
                    30,
                    150,
                    90,
                ]
            },
            {
                "bbox": [
                    20,
                    10,
                    120,
                    120,
                ]
            },
        ]
    }

    result = evidence_regions_bbox(
        signals
    )

    assert result == {
        "x0": 5,
        "y0": 10,
        "x1": 150,
        "y1": 120,
    }


def test_evidence_regions_bbox_float_conversion():
    signals = {
        "evidence_regions": [
            {
                "bbox": [
                    10.8,
                    20.9,
                    100.7,
                    80.6,
                ]
            }
        ]
    }

    result = evidence_regions_bbox(
        signals
    )

    assert result == {
        "x0": 10,
        "y0": 20,
        "x1": 100,
        "y1": 80,
    }


@pytest.mark.parametrize(
    "bbox",
    [
        None,
        [],
        [1, 2, 3],
        [1, 2, 3, 4, 5],
        ["1", 2, 3, 4],
        [10, 10, 10, 20],
        [10, 10, 5, 20],
        [10, 20, 30, 20],
        [10, 20, 30, 10],
    ],
)
def test_evidence_regions_bbox_invalid_region(
    bbox,
):
    signals = {
        "evidence_regions": [
            {"bbox": bbox}
        ]
    }

    assert evidence_regions_bbox(
        signals
    ) is None


def test_evidence_regions_bbox_ignores_invalid_region():
    signals = {
        "evidence_regions": [
            {
                "bbox": [
                    10,
                    10,
                    5,
                    20,
                ]
            },
            {
                "bbox": [
                    20,
                    30,
                    100,
                    90,
                ]
            },
        ]
    }

    result = evidence_regions_bbox(
        signals
    )

    assert result == {
        "x0": 20,
        "y0": 30,
        "x1": 100,
        "y1": 90,
    }


def test_evidence_regions_bbox_region_not_dict():
    signals = {
        "evidence_regions": [
            "invalid",
            123,
            None,
        ]
    }

    assert evidence_regions_bbox(
        signals
    ) is None


def test_evidence_regions_bbox_fallback():
    signals = {
        "evidence_bbox": {
            "x0": 10,
            "y0": 20,
            "x1": 100,
            "y1": 80,
        }
    }

    result = evidence_regions_bbox(
        signals
    )

    assert result == {
        "x0": 10,
        "y0": 20,
        "x1": 100,
        "y1": 80,
    }


def test_evidence_regions_preferred_over_fallback():
    signals = {
        "evidence_regions": [
            {
                "bbox": [
                    1,
                    2,
                    30,
                    40,
                ]
            }
        ],
        "evidence_bbox": {
            "x0": 100,
            "y0": 200,
            "x1": 300,
            "y1": 400,
        },
    }

    result = evidence_regions_bbox(
        signals
    )

    assert result == {
        "x0": 1,
        "y0": 2,
        "x1": 30,
        "y1": 40,
    }


def test_evidence_regions_invalid_then_fallback():
    signals = {
        "evidence_regions": [
            {
                "bbox": [
                    10,
                    10,
                    5,
                    5,
                ]
            }
        ],
        "evidence_bbox": {
            "x0": 20,
            "y0": 30,
            "x1": 100,
            "y1": 120,
        },
    }

    result = evidence_regions_bbox(
        signals
    )

    assert result == {
        "x0": 20,
        "y0": 30,
        "x1": 100,
        "y1": 120,
    }


def test_evidence_bbox_not_dict():
    signals = {
        "evidence_bbox": [
            1,
            2,
            3,
            4,
        ]
    }

    assert evidence_regions_bbox(
        signals
    ) is None


@pytest.mark.parametrize(
    "bbox",
    [
        {
            "x0": None,
            "y0": 2,
            "x1": 3,
            "y1": 4,
        },
        {
            "x0": "1",
            "y0": 2,
            "x1": 3,
            "y1": 4,
        },
        {
            "x0": 10,
            "y0": 20,
            "x1": 10,
            "y1": 30,
        },
        {
            "x0": 10,
            "y0": 20,
            "x1": 5,
            "y1": 30,
        },
        {
            "x0": 10,
            "y0": 20,
            "x1": 30,
            "y1": 20,
        },
    ],
)
def test_evidence_bbox_invalid(bbox):
    signals = {
        "evidence_bbox": bbox
    }

    assert evidence_regions_bbox(
        signals
    ) is None


def test_evidence_bbox_float_conversion():
    signals = {
        "evidence_bbox": {
            "x0": 10.8,
            "y0": 20.9,
            "x1": 100.7,
            "y1": 80.6,
        }
    }

    assert evidence_regions_bbox(
        signals
    ) == {
        "x0": 10,
        "y0": 20,
        "x1": 100,
        "y1": 80,
    }


# ============================================================
# tokens_bbox
# ============================================================

def test_tokens_bbox_empty():
    assert tokens_bbox(
        [],
        {},
    ) is None


def test_tokens_bbox_integer_key():
    token_map = {
        1: {
            "bbox": [
                10,
                20,
                100,
                80,
            ]
        }
    }

    result = tokens_bbox(
        [1],
        token_map,
    )

    assert result == {
        "x0": 10,
        "y0": 20,
        "x1": 100,
        "y1": 80,
    }


def test_tokens_bbox_string_key():
    token_map = {
        "1": {
            "bbox": [
                10,
                20,
                100,
                80,
            ]
        }
    }

    result = tokens_bbox(
        [1],
        token_map,
    )

    assert result == {
        "x0": 10,
        "y0": 20,
        "x1": 100,
        "y1": 80,
    }


def test_tokens_bbox_prefers_string_key():
    token_map = {
        "1": {
            "bbox": [
                10,
                20,
                100,
                80,
            ]
        },
        1: {
            "bbox": [
                500,
                500,
                600,
                600,
            ]
        },
    }

    result = tokens_bbox(
        [1],
        token_map,
    )

    assert result == {
        "x0": 10,
        "y0": 20,
        "x1": 100,
        "y1": 80,
    }


def test_tokens_bbox_multiple():
    token_map = {
        "1": {
            "bbox": [
                10,
                20,
                100,
                80,
            ]
        },
        "2": {
            "bbox": [
                5,
                30,
                150,
                90,
            ]
        },
        "3": {
            "bbox": [
                20,
                10,
                120,
                120,
            ]
        },
    }

    result = tokens_bbox(
        [1, 2, 3],
        token_map,
    )

    assert result == {
        "x0": 5,
        "y0": 10,
        "x1": 150,
        "y1": 120,
    }


def test_tokens_bbox_float_conversion():
    token_map = {
        "1": {
            "bbox": [
                10.8,
                20.9,
                100.7,
                80.6,
            ]
        }
    }

    assert tokens_bbox(
        [1],
        token_map,
    ) == {
        "x0": 10,
        "y0": 20,
        "x1": 100,
        "y1": 80,
    }


def test_tokens_bbox_missing_token():
    assert tokens_bbox(
        [999],
        {},
    ) is None


def test_tokens_bbox_token_not_dict():
    token_map = {
        "1": "invalid"
    }

    assert tokens_bbox(
        [1],
        token_map,
    ) is None


@pytest.mark.parametrize(
    "bbox",
    [
        None,
        [],
        [1, 2, 3],
        [1, 2, 3, 4, 5],
        ["1", 2, 3, 4],
    ],
)
def test_tokens_bbox_invalid_bbox(
    bbox,
):
    token_map = {
        "1": {
            "bbox": bbox
        }
    }

    assert tokens_bbox(
        [1],
        token_map,
    ) is None


def test_tokens_bbox_ignores_invalid_and_uses_valid():
    token_map = {
        "1": {
            "bbox": None
        },
        "2": {
            "bbox": [
                10,
                20,
                100,
                80,
            ]
        },
    }

    assert tokens_bbox(
        [1, 2],
        token_map,
    ) == {
        "x0": 10,
        "y0": 20,
        "x1": 100,
        "y1": 80,
    }


# ============================================================
# extract_ocr_result
# ============================================================

def test_extract_ocr_result_none():
    result = extract_ocr_result(
        None
    )

    for field in OCR_RESULT_FIELDS:
        assert field in result

        assert result[field] is None

        assert result[
            f"{field}_verbatim"
        ] is None

        assert result[
            f"{field}_corrected"
        ] is None

        assert result[
            f"{field}_correction_basis"
        ] is None

        assert result[
            f"{field}_conf"
        ] is None

        assert result[
            f"{field}_source"
        ] is None

        assert result[
            f"{field}_tokens"
        ] == []

        assert result[
            f"{field}_tokens_bbox"
        ] is None

        assert result[
            f"{field}_reason"
        ] is None

        assert result[
            f"{field}_notes"
        ] == []

        assert result[
            f"{field}_review_status"
        ] is None


def test_extract_empty_record():
    result = extract_ocr_result(
        make_record()
    )

    for field in OCR_RESULT_FIELDS:
        assert result[field] is None
        assert result[
            f"{field}_tokens"
        ] == []


def test_extract_prefers_corrected():
    field = make_field(
        verbatim="603/AI",
        corrected="603/A1",
    )

    record = make_record(
        fields={
            "drawing_number": field
        }
    )

    result = extract_ocr_result(
        record
    )

    assert (
        result["drawing_number"]
        == "603/A1"
    )

    assert (
        result[
            "drawing_number_verbatim"
        ]
        == "603/AI"
    )

    assert (
        result[
            "drawing_number_corrected"
        ]
        == "603/A1"
    )


def test_extract_falls_back_to_verbatim():
    field = make_field(
        verbatim="603/A1",
        corrected=None,
    )

    record = make_record(
        fields={
            "drawing_number": field
        }
    )

    result = extract_ocr_result(
        record
    )

    assert (
        result["drawing_number"]
        == "603/A1"
    )


def test_extract_empty_corrected_falls_back():
    field = make_field(
        verbatim="603/A1",
        corrected="",
    )

    record = make_record(
        fields={
            "drawing_number": field
        }
    )

    result = extract_ocr_result(
        record
    )

    assert (
        result["drawing_number"]
        == "603/A1"
    )


def test_extract_confidence_rounded():
    field = make_field(
        confidence=0.87654,
    )

    record = make_record(
        fields={
            "scale": field
        }
    )

    result = extract_ocr_result(
        record
    )

    assert (
        result["scale_conf"]
        == 0.877
    )


def test_extract_confidence_integer():
    field = make_field(
        confidence=1,
    )

    result = extract_ocr_result(
        make_record(
            fields={
                "scale": field
            }
        )
    )

    assert result[
        "scale_conf"
    ] == 1.0


def test_extract_confidence_string_number():
    field = make_field(
        confidence="0.81234",
    )

    result = extract_ocr_result(
        make_record(
            fields={
                "scale": field
            }
        )
    )

    assert result[
        "scale_conf"
    ] == 0.812


def test_extract_confidence_none():
    field = make_field(
        confidence=None,
    )

    result = extract_ocr_result(
        make_record(
            fields={
                "scale": field
            }
        )
    )

    assert result[
        "scale_conf"
    ] is None


def test_extract_source():
    field = make_field(
        source="rule",
    )

    result = extract_ocr_result(
        make_record(
            fields={
                "date": field
            }
        )
    )

    assert (
        result["date_source"]
        == "rule"
    )


def test_extract_empty_source_becomes_none():
    field = make_field(
        source="",
    )

    result = extract_ocr_result(
        make_record(
            fields={
                "date": field
            }
        )
    )

    assert (
        result["date_source"]
        is None
    )


def test_extract_correction_basis():
    field = make_field(
        correction_basis="OCR confusion",
    )

    result = extract_ocr_result(
        make_record(
            fields={
                "drawing_number": field
            }
        )
    )

    assert (
        result[
            "drawing_number_correction_basis"
        ]
        == "OCR confusion"
    )


def test_extract_tokens():
    field = make_field(
        tokens=[1, 2, 3],
    )

    result = extract_ocr_result(
        make_record(
            fields={
                "drawing_title": field
            }
        )
    )

    assert (
        result[
            "drawing_title_tokens"
        ]
        == [1, 2, 3]
    )


def test_extract_tokens_none_becomes_empty():
    field = make_field()
    field.tokens = None

    result = extract_ocr_result(
        make_record(
            fields={
                "drawing_title": field
            }
        )
    )

    assert (
        result[
            "drawing_title_tokens"
        ]
        == []
    )


def test_extract_token_bbox():
    field = make_field(
        tokens=[1, 2],
    )

    token_map = {
        "1": {
            "bbox": [
                10,
                20,
                100,
                80,
            ]
        },
        "2": {
            "bbox": [
                5,
                30,
                150,
                90,
            ]
        },
    }

    result = extract_ocr_result(
        make_record(
            fields={
                "drawing_number": field
            },
            token_map=token_map,
        )
    )

    assert result[
        "drawing_number_tokens_bbox"
    ] == {
        "x0": 5,
        "y0": 20,
        "x1": 150,
        "y1": 90,
    }


def test_extract_evidence_bbox_preferred():
    field = make_field(
        tokens=[1],
        signals={
            "evidence_regions": [
                {
                    "bbox": [
                        100,
                        200,
                        300,
                        400,
                    ]
                }
            ]
        },
    )

    token_map = {
        "1": {
            "bbox": [
                10,
                20,
                30,
                40,
            ]
        }
    }

    result = extract_ocr_result(
        make_record(
            fields={
                "drawing_number": field
            },
            token_map=token_map,
        )
    )

    # Evidence region has priority
    # over token bbox.
    assert result[
        "drawing_number_tokens_bbox"
    ] == {
        "x0": 100,
        "y0": 200,
        "x1": 300,
        "y1": 400,
    }


def test_extract_invalid_evidence_falls_back_to_tokens():
    field = make_field(
        tokens=[1],
        signals={
            "evidence_regions": [
                {
                    "bbox": [
                        100,
                        100,
                        50,
                        50,
                    ]
                }
            ]
        },
    )

    token_map = {
        "1": {
            "bbox": [
                10,
                20,
                30,
                40,
            ]
        }
    }

    result = extract_ocr_result(
        make_record(
            fields={
                "drawing_number": field
            },
            token_map=token_map,
        )
    )

    assert result[
        "drawing_number_tokens_bbox"
    ] == {
        "x0": 10,
        "y0": 20,
        "x1": 30,
        "y1": 40,
    }


def test_extract_reason_and_notes():
    field = make_field(
        reason=[
            "ambiguous",
            "two possible values",
        ],
        notes=[
            "manual review",
        ],
    )

    result = extract_ocr_result(
        make_record(
            fields={
                "drawing_number": field
            }
        )
    )

    assert result[
        "drawing_number_reason"
    ] == "ambiguous"

    assert result[
        "drawing_number_notes"
    ] == [
        "manual review",
        "two possible values",
    ]


def test_extract_review_status():
    field = make_field(
        status="locked",
    )

    result = extract_ocr_result(
        make_record(
            fields={
                "architect": field
            }
        )
    )

    assert (
        result[
            "architect_review_status"
        ]
        == "locked"
    )


def test_extract_complete_field():
    field = make_field(
        tokens=[7],
        verbatim="603/AI",
        corrected="603/A1",
        confidence=0.98765,
        correction_basis="character confusion",
        source="llm",
        signals={
            "evidence_regions": [
                {
                    "bbox": [
                        100,
                        200,
                        300,
                        250,
                    ]
                }
            ]
        },
        reason="ambiguous",
        notes=[
            "corrected using context"
        ],
        status="reviewed",
    )

    record = make_record(
        fields={
            "drawing_number": field
        },
        token_map={
            "7": {
                "bbox": [
                    10,
                    20,
                    30,
                    40,
                ]
            }
        },
    )

    result = extract_ocr_result(
        record
    )

    assert result[
        "drawing_number"
    ] == "603/A1"

    assert result[
        "drawing_number_verbatim"
    ] == "603/AI"

    assert result[
        "drawing_number_corrected"
    ] == "603/A1"

    assert result[
        "drawing_number_correction_basis"
    ] == "character confusion"

    assert result[
        "drawing_number_conf"
    ] == 0.988

    assert result[
        "drawing_number_source"
    ] == "llm"

    assert result[
        "drawing_number_tokens"
    ] == [7]

    assert result[
        "drawing_number_tokens_bbox"
    ] == {
        "x0": 100,
        "y0": 200,
        "x1": 300,
        "y1": 250,
    }

    assert result[
        "drawing_number_reason"
    ] == "ambiguous"

    assert result[
        "drawing_number_notes"
    ] == [
        "corrected using context"
    ]

    assert result[
        "drawing_number_review_status"
    ] == "reviewed"


def test_extract_multiple_fields():
    fields = {
        "drawing_number": make_field(
            corrected="A101",
            confidence=0.95,
        ),
        "project_name": make_field(
            corrected="PROJECT TEST",
            confidence=0.90,
        ),
        "drawing_title": make_field(
            corrected="GROUND FLOOR PLAN",
            confidence=0.92,
        ),
        "architect": make_field(
            corrected="SMITH ARCHITECTS",
            confidence=0.88,
        ),
        "draughtsperson": make_field(
            corrected="JOHN",
            confidence=0.80,
        ),
        "date": make_field(
            corrected="2026-10-07",
            confidence=0.97,
        ),
        "scale": make_field(
            corrected="1:100",
            confidence=0.99,
        ),
        "drawing_type": make_field(
            corrected="plan",
            confidence=0.91,
        ),
    }

    result = extract_ocr_result(
        make_record(fields=fields)
    )

    assert result[
        "drawing_number"
    ] == "A101"

    assert result[
        "project_name"
    ] == "PROJECT TEST"

    assert result[
        "drawing_title"
    ] == "GROUND FLOOR PLAN"

    assert result[
        "architect"
    ] == "SMITH ARCHITECTS"

    assert result[
        "draughtsperson"
    ] == "JOHN"

    assert result[
        "date"
    ] == "2026-10-07"

    assert result[
        "scale"
    ] == "1:100"

    assert result[
        "drawing_type"
    ] == "plan"


def test_extract_ignores_extra_fields():
    record = make_record(
        fields={
            "unknown_field": make_field(
                corrected="SHOULD NOT APPEAR"
            )
        }
    )

    result = extract_ocr_result(
        record
    )

    assert "unknown_field" not in result