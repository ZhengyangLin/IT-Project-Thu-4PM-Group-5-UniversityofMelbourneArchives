from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from uma.trace import Tracer, SEP, SUB


# ============================================================
# Helpers
# ============================================================

def make_candidate(
    text,
    engine="test_engine",
    conf=0.90,
):
    return SimpleNamespace(
        text=text,
        engine=engine,
        conf=conf,
    )


def make_token(
    text,
    bbox=(10, 20, 100, 40),
    conf=0.95,
    agreement=1.0,
    tid=1,
    candidates=None,
):
    x0, y0, x1, y1 = bbox

    return SimpleNamespace(
        text=text,
        bbox=bbox,
        conf=conf,
        agreement=agreement,
        tid=tid,
        candidates=candidates or [],
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        cx=(x0 + x1) / 2,
        cy=(y0 + y1) / 2,
    )


# ============================================================
# Constants
# ============================================================

def test_separator_constants():
    assert SEP == "═" * 78
    assert SUB == "─" * 78


# ============================================================
# __init__
# ============================================================

def test_init_enabled_creates_file(tmp_path):
    tracer = Tracer(
        node_id="NODE001",
        out_dir=str(tmp_path),
        enabled=True,
    )

    expected = (
        tmp_path
        / "trace"
        / "NODE001.txt"
    )

    assert tracer.enabled is True
    assert tracer.node_id == "NODE001"
    assert tracer.path == expected

    assert expected.exists()

    tracer.close()


def test_init_creates_trace_directory(tmp_path):
    tracer = Tracer(
        node_id="NODE002",
        out_dir=str(tmp_path),
    )

    trace_dir = (
        tmp_path
        / "trace"
    )

    assert trace_dir.exists()
    assert trace_dir.is_dir()

    tracer.close()


def test_init_disabled_does_not_create_file(
    tmp_path,
):
    tracer = Tracer(
        node_id="DISABLED",
        out_dir=str(tmp_path),
        enabled=False,
    )

    expected = (
        tmp_path
        / "trace"
        / "DISABLED.txt"
    )

    assert tracer.enabled is False
    assert tracer.node_id == "DISABLED"
    assert tracer.path == expected

    assert not expected.exists()


# ============================================================
# _head
# ============================================================

def test_head_written_on_initialization(
    tmp_path,
):
    with patch(
        "uma.trace.time.strftime",
        return_value="2026-10-07 12:34:56",
    ):
        tracer = Tracer(
            node_id="NODE003",
            out_dir=str(tmp_path),
        )

        tracer.close()

    content = tracer.path.read_text(
        encoding="utf-8"
    )

    assert SEP in content
    assert "node_id  NODE003" in content

    assert (
        "timestamp 2026-10-07 12:34:56"
        in content
    )


def test_head_contains_two_separators(
    tmp_path,
):
    tracer = Tracer(
        node_id="HEAD",
        out_dir=str(tmp_path),
    )

    tracer.close()

    content = tracer.path.read_text(
        encoding="utf-8"
    )

    # Header has two SEP lines.
    # close() adds another two.
    assert content.count(SEP) == 4


# ============================================================
# section
# ============================================================

def test_section_basic(tmp_path):
    tracer = Tracer(
        "SECTION1",
        str(tmp_path),
    )

    tracer.section(
        "OCR RESULTS"
    )

    tracer.close()

    content = tracer.path.read_text(
        encoding="utf-8"
    )

    assert "OCR RESULTS" in content


def test_section_with_note(tmp_path):
    tracer = Tracer(
        "SECTION2",
        str(tmp_path),
    )

    tracer.section(
        "PREPROCESS",
        "Image converted to grayscale",
    )

    tracer.close()

    content = tracer.path.read_text(
        encoding="utf-8"
    )

    assert "PREPROCESS" in content

    assert (
        "Note: Image converted to grayscale"
        in content
    )


def test_section_multiline_note(tmp_path):
    tracer = Tracer(
        "SECTION3",
        str(tmp_path),
    )

    tracer.section(
        "OCR",
        "First line\nSecond line\nThird line",
    )

    tracer.close()

    content = tracer.path.read_text(
        encoding="utf-8"
    )

    assert "Note: First line" in content
    assert "Second line" in content
    assert "Third line" in content

    # Only the first line receives
    # the literal "Note:" prefix.
    assert "Note: Second line" not in content


def test_section_empty_note(tmp_path):
    tracer = Tracer(
        "SECTION4",
        str(tmp_path),
    )

    tracer.section(
        "TEST SECTION",
        "",
    )

    tracer.close()

    content = tracer.path.read_text(
        encoding="utf-8"
    )

    assert "TEST SECTION" in content
    assert "Note:" not in content


def test_section_disabled(tmp_path):
    tracer = Tracer(
        "DISABLED_SECTION",
        str(tmp_path),
        enabled=False,
    )

    # Should simply return without error.
    result = tracer.section(
        "THIS SHOULD NOT BE WRITTEN",
        "note",
    )

    assert result is None
    assert not tracer.path.exists()


# ============================================================
# text
# ============================================================

def test_text_basic(tmp_path):
    tracer = Tracer(
        "TEXT1",
        str(tmp_path),
    )

    tracer.text(
        "Hello world"
    )

    tracer.close()

    content = tracer.path.read_text(
        encoding="utf-8"
    )

    assert "Hello world" in content


def test_text_multiline(tmp_path):
    tracer = Tracer(
        "TEXT2",
        str(tmp_path),
    )

    tracer.text(
        "Line one\nLine two"
    )

    tracer.close()

    content = tracer.path.read_text(
        encoding="utf-8"
    )

    assert "Line one" in content
    assert "Line two" in content


def test_text_removes_trailing_whitespace(
    tmp_path,
):
    tracer = Tracer(
        "TEXT3",
        str(tmp_path),
    )

    tracer.text(
        "Hello world   \n\n"
    )

    tracer.close()

    content = tracer.path.read_text(
        encoding="utf-8"
    )

    assert "Hello world\n" in content
    assert "Hello world   " not in content


def test_text_empty_string(tmp_path):
    tracer = Tracer(
        "TEXT4",
        str(tmp_path),
    )

    tracer.text("")

    tracer.close()

    content = tracer.path.read_text(
        encoding="utf-8"
    )

    assert "End" in content


def test_text_disabled(tmp_path):
    tracer = Tracer(
        "DISABLED_TEXT",
        str(tmp_path),
        enabled=False,
    )

    result = tracer.text(
        "Should not be written"
    )

    assert result is None
    assert not tracer.path.exists()


# ============================================================
# tokens
# ============================================================

def test_tokens_basic(tmp_path):
    tracer = Tracer(
        "TOKENS1",
        str(tmp_path),
    )

    token = make_token(
        text="DRAWING NUMBER",
        bbox=(10, 20, 150, 50),
        conf=0.95,
        agreement=0.90,
        tid=5,
    )

    tracer.tokens(
        [token]
    )

    tracer.close()

    content = tracer.path.read_text(
        encoding="utf-8"
    )

    assert "Total: 1" in content

    assert (
        "'DRAWING NUMBER'"
        in content
    )

    assert "[5]" in content

    assert (
        "bbox=[10, 20, 150, 50]"
        in content
    )

    assert "conf=0.950" in content
    assert "agreement=0.90" in content


def test_tokens_without_tid(tmp_path):
    tracer = Tracer(
        "TOKENS2",
        str(tmp_path),
    )

    token = make_token(
        text="SCALE",
        tid=-1,
    )

    tracer.tokens(
        [token]
    )

    tracer.close()

    content = tracer.path.read_text(
        encoding="utf-8"
    )

    assert "[-]" in content
    assert "'SCALE'" in content


def test_tokens_missing_tid_attribute(
    tmp_path,
):
    tracer = Tracer(
        "TOKENS3",
        str(tmp_path),
    )

    token = SimpleNamespace(
        text="DATE",
        bbox=(10, 20, 50, 40),
        conf=0.9,
        agreement=1.0,
        candidates=[],
        x0=10,
        cy=30,
    )

    tracer.tokens(
        [token]
    )

    tracer.close()

    content = tracer.path.read_text(
        encoding="utf-8"
    )

    assert "[-]" in content
    assert "'DATE'" in content


def test_tokens_with_note(tmp_path):
    tracer = Tracer(
        "TOKENS4",
        str(tmp_path),
    )

    token = make_token(
        "SITE PLAN"
    )

    tracer.tokens(
        [token],
        note="OCR output",
    )

    tracer.close()

    content = tracer.path.read_text(
        encoding="utf-8"
    )

    assert "OCR output" in content
    assert "Total: 1" in content


def test_tokens_without_note(tmp_path):
    tracer = Tracer(
        "TOKENS5",
        str(tmp_path),
    )

    token = make_token(
        "SITE PLAN"
    )

    tracer.tokens(
        [token]
    )

    tracer.close()

    content = tracer.path.read_text(
        encoding="utf-8"
    )

    assert "Total: 1" in content


def test_tokens_empty(tmp_path):
    tracer = Tracer(
        "TOKENS6",
        str(tmp_path),
    )

    tracer.tokens([])

    tracer.close()

    content = tracer.path.read_text(
        encoding="utf-8"
    )

    assert "Total: 0" in content
    assert SUB in content


def test_tokens_disabled(tmp_path):
    tracer = Tracer(
        "DISABLED_TOKENS",
        str(tmp_path),
        enabled=False,
    )

    token = make_token(
        "TEST"
    )

    result = tracer.tokens(
        [token]
    )

    assert result is None
    assert not tracer.path.exists()


# ============================================================
# Token sorting
# ============================================================

def test_tokens_sorted_top_to_bottom(
    tmp_path,
):
    tracer = Tracer(
        "SORT1",
        str(tmp_path),
    )

    bottom = make_token(
        "BOTTOM",
        bbox=(10, 200, 100, 220),
        tid=2,
    )

    top = make_token(
        "TOP",
        bbox=(10, 50, 100, 70),
        tid=1,
    )

    tracer.tokens(
        [bottom, top]
    )

    tracer.close()

    content = tracer.path.read_text(
        encoding="utf-8"
    )

    assert (
        content.index("'TOP'")
        <
        content.index("'BOTTOM'")
    )


def test_tokens_sorted_left_to_right(
    tmp_path,
):
    tracer = Tracer(
        "SORT2",
        str(tmp_path),
    )

    right = make_token(
        "RIGHT",
        bbox=(200, 100, 250, 120),
        tid=2,
    )

    left = make_token(
        "LEFT",
        bbox=(10, 100, 60, 120),
        tid=1,
    )

    tracer.tokens(
        [right, left]
    )

    tracer.close()

    content = tracer.path.read_text(
        encoding="utf-8"
    )

    assert (
        content.index("'LEFT'")
        <
        content.index("'RIGHT'")
    )


# ============================================================
# Candidates
# ============================================================

def test_tokens_candidate_primary(
    tmp_path,
):
    tracer = Tracer(
        "CAND1",
        str(tmp_path),
    )

    candidate = make_candidate(
        text="603/A1",
        engine="paddle",
        conf=0.95,
    )

    token = make_token(
        text="603/A1",
        candidates=[candidate],
    )

    tracer.tokens(
        [token]
    )

    tracer.close()

    content = tracer.path.read_text(
        encoding="utf-8"
    )

    assert "←primary" in content
    assert "paddle" in content
    assert "'603/A1'" in content
    assert "(0.950)" in content


def test_tokens_candidate_not_primary(
    tmp_path,
):
    tracer = Tracer(
        "CAND2",
        str(tmp_path),
    )

    candidate = make_candidate(
        text="603/A2",
        engine="secondary",
        conf=0.80,
    )

    token = make_token(
        text="603/A1",
        candidates=[candidate],
    )

    tracer.tokens(
        [token]
    )

    tracer.close()

    content = tracer.path.read_text(
        encoding="utf-8"
    )

    assert "secondary" in content
    assert "'603/A2'" in content

    # Since candidate text differs from
    # primary token text, it must not
    # receive the primary marker.
    candidate_line = next(
        line
        for line in content.splitlines()
        if "'603/A2'" in line
    )

    assert "←primary" not in candidate_line


def test_tokens_multiple_candidates(
    tmp_path,
):
    tracer = Tracer(
        "CAND3",
        str(tmp_path),
    )

    candidates = [
        make_candidate(
            "ABC",
            "engine1",
            0.9,
        ),
        make_candidate(
            "ABD",
            "engine2",
            0.8,
        ),
        make_candidate(
            "ABE",
            "engine3",
            0.7,
        ),
    ]

    token = make_token(
        "ABC",
        candidates=candidates,
    )

    tracer.tokens(
        [token]
    )

    tracer.close()

    content = tracer.path.read_text(
        encoding="utf-8"
    )

    assert "engine1" in content
    assert "engine2" in content
    assert "engine3" in content

    assert "'ABC'" in content
    assert "'ABD'" in content
    assert "'ABE'" in content


def test_tokens_missing_candidates_attribute(
    tmp_path,
):
    tracer = Tracer(
        "CAND4",
        str(tmp_path),
    )

    token = SimpleNamespace(
        text="TEST",
        bbox=(10, 10, 50, 30),
        conf=0.9,
        agreement=1.0,
        tid=1,
        x0=10,
        cy=20,
    )

    tracer.tokens(
        [token]
    )

    tracer.close()

    content = tracer.path.read_text(
        encoding="utf-8"
    )

    assert "'TEST'" in content


# ============================================================
# close
# ============================================================

def test_close_returns_correct_path(
    tmp_path,
):
    tracer = Tracer(
        "CLOSE1",
        str(tmp_path),
    )

    result = tracer.close()

    expected = str(
        tmp_path
        / "trace"
        / "CLOSE1.txt"
    )

    assert result == expected


def test_close_writes_end(tmp_path):
    tracer = Tracer(
        "CLOSE2",
        str(tmp_path),
    )

    path = tracer.close()

    content = Path(path).read_text(
        encoding="utf-8"
    )

    assert "End" in content


def test_close_closes_file(tmp_path):
    tracer = Tracer(
        "CLOSE3",
        str(tmp_path),
    )

    tracer.close()

    assert tracer.f.closed is True


def test_close_disabled(tmp_path):
    tracer = Tracer(
        "CLOSE_DISABLED",
        str(tmp_path),
        enabled=False,
    )

    result = tracer.close()

    expected = str(
        tmp_path
        / "trace"
        / "CLOSE_DISABLED.txt"
    )

    assert result == expected

    assert not tracer.path.exists()


# ============================================================
# Complete workflow
# ============================================================

def test_complete_trace_workflow(
    tmp_path,
):
    candidate1 = make_candidate(
        "603/A1",
        "paddle",
        0.98,
    )

    candidate2 = make_candidate(
        "603/AI",
        "easyocr",
        0.75,
    )

    token1 = make_token(
        text="603/A1",
        bbox=(100, 100, 200, 130),
        conf=0.98,
        agreement=0.90,
        tid=1,
        candidates=[
            candidate1,
            candidate2,
        ],
    )

    token2 = make_token(
        text="SITE PLAN",
        bbox=(100, 150, 250, 180),
        conf=0.95,
        agreement=1.0,
        tid=2,
    )

    with patch(
        "uma.trace.time.strftime",
        return_value="2026-10-07 20:00:00",
    ):
        tracer = Tracer(
            "NODE100",
            str(tmp_path),
        )

        tracer.section(
            "OCR",
            "OCR recognition results",
        )

        tracer.text(
            "Processing drawing"
        )

        tracer.tokens(
            [token2, token1],
            note="Detected tokens",
        )

        result_path = tracer.close()

    content = Path(
        result_path
    ).read_text(
        encoding="utf-8"
    )

    assert "node_id  NODE100" in content

    assert (
        "timestamp 2026-10-07 20:00:00"
        in content
    )

    assert "OCR" in content

    assert (
        "Note: OCR recognition results"
        in content
    )

    assert "Processing drawing" in content
    assert "Detected tokens" in content
    assert "Total: 2" in content

    assert "'603/A1'" in content
    assert "'SITE PLAN'" in content

    assert "paddle" in content
    assert "easyocr" in content
    assert "←primary" in content

    assert "End" in content