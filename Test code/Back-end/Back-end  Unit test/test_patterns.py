import pytest
from types import SimpleNamespace

from uma.patterns import (
    SCALE_RE,
    VAGUE_SCALE_RE,
    CAPTION_RE,
    DATE_RE,
    YEAR_RE,
    DRAWING_NO_RE,
    DIMENSION_RE,
    DRAUGHTSPERSON_CONFLICT_RE,
    is_drawing_number,
    is_vague_scale,
    alpha_ratio,
    normalize_confusions,
    confusion_variants,
    more_complete,
    scan_candidates,
)


# ============================================================
# SCALE_RE
# ============================================================

@pytest.mark.parametrize(
    "text",
    [
        "SCALE",
        "1:100",
        "1 : 50",
        '1/4" = 1\'',
        "AS SHOWN",
        "AS NOTED",
        "FULL SIZE",
        "N.T.S.",
        "NOT TO SCALE",
        "VARIOUS",
    ],
)
def test_scale_re_matches_valid_scales(text):
    assert SCALE_RE.search(text) is not None


def test_scale_re_rejects_normal_text():
    assert SCALE_RE.search("SITE PLAN") is None


# ============================================================
# VAGUE_SCALE_RE
# ============================================================

@pytest.mark.parametrize(
    "text",
    [
        "AS SHOWN",
        "AS NOTED",
        "VARIOUS",
        "Scale as shown",
    ],
)
def test_vague_scale_regex(text):
    assert VAGUE_SCALE_RE.search(text) is not None


def test_vague_scale_regex_rejects_numeric():
    assert VAGUE_SCALE_RE.search("1:100") is None


# ============================================================
# CAPTION_RE
# ============================================================

@pytest.mark.parametrize(
    "text",
    [
        "NORTH ELEVATION",
        "SOUTH ELEVATION",
        "EAST ELEVATION",
        "WEST ELEVATION",
        "FRONT ELEVATION",
        "REAR ELEVATION",
        "SIDE ELEVATION",
        "PLAN",
        "PERSPECTIVE",
        "SECTION A-A",
        "SECTIONS A-B",
        "DETAIL A",
    ],
)
def test_caption_re_matches(text):
    assert CAPTION_RE.search(text) is not None


def test_caption_re_rejects_unrelated_text():
    assert CAPTION_RE.search("DRAWING NUMBER 603/A1") is None


# ============================================================
# DATE_RE
# ============================================================

@pytest.mark.parametrize(
    "text",
    [
        "12/05/2024",
        "12-05-2024",
        "12.05.2024",
        "1/1/24",
        "JAN 2024",
        "JANUARY 2024",
        "12 JAN 2024",
        "12 DECEMBER 2024",
        "1999",
        "2024",
    ],
)
def test_date_re_matches_dates(text):
    assert DATE_RE.search(text) is not None


def test_date_re_rejects_normal_text():
    assert DATE_RE.search("SITE PLAN") is None


# ============================================================
# YEAR_RE
# ============================================================

@pytest.mark.parametrize(
    "year",
    [
        "1800",
        "1899",
        "1900",
        "1999",
        "2000",
        "2024",
    ],
)
def test_year_re_valid(year):
    assert YEAR_RE.search(year) is not None


@pytest.mark.parametrize(
    "text",
    [
        "1700",
        "2100",
        "SITE",
    ],
)
def test_year_re_invalid(text):
    assert YEAR_RE.search(text) is None


# ============================================================
# DRAWING_NO_RE
# ============================================================

@pytest.mark.parametrize(
    "drawing_number",
    [
        "603/A1",
        "562/D66",
        "123/AB12",
        "A-1234",
        "A1234",
        "AB-123",
        "ABC12345",
        "12345-678",
        "12345/678",
    ],
)
def test_drawing_number_regex_valid(drawing_number):
    assert DRAWING_NO_RE.fullmatch(drawing_number) is not None


@pytest.mark.parametrize(
    "text",
    [
        "SITE PLAN",
        "HELLO",
        "DRAWING",
    ],
)
def test_drawing_number_regex_invalid(text):
    assert DRAWING_NO_RE.fullmatch(text) is None


# ============================================================
# is_drawing_number
# ============================================================

@pytest.mark.parametrize(
    "drawing_number",
    [
        "603/A1",
        "562/D66",
        "A-1234",
        "AB123",
        "12345-678",
    ],
)
def test_is_drawing_number_valid(drawing_number):
    assert is_drawing_number(drawing_number) is True


def test_is_drawing_number_with_whitespace():
    assert is_drawing_number("  603/A1  ") is True


def test_is_drawing_number_with_trailing_punctuation():
    assert is_drawing_number("603/A1.") is True
    assert is_drawing_number("603/A1,") is True
    assert is_drawing_number("603/A1;") is True
    assert is_drawing_number("603/A1:") is True


def test_is_drawing_number_none():
    assert is_drawing_number(None) is False


def test_is_drawing_number_empty():
    assert is_drawing_number("") is False
    assert is_drawing_number("   ") is False


def test_is_drawing_number_date_rejected():
    assert is_drawing_number("12/05/2024") is False


def test_is_drawing_number_year_rejected():
    assert is_drawing_number("2024") is False


def test_is_drawing_number_normal_text_rejected():
    assert is_drawing_number("SITE PLAN") is False


# ============================================================
# DIMENSION_RE
# ============================================================

@pytest.mark.parametrize(
    "text",
    [
        "100",
        "100.5",
        "100-200",
        '1/4"',
        "6'",
        "Ø50",
        "R25",
    ],
)
def test_dimension_re_valid(text):
    assert DIMENSION_RE.fullmatch(text) is not None


def test_dimension_re_invalid():
    assert DIMENSION_RE.fullmatch("SITE PLAN") is None


# ============================================================
# DRAUGHTSPERSON_CONFLICT_RE
# ============================================================

@pytest.mark.parametrize(
    "text",
    [
        "DATE",
        "SCALE",
        "SCALES",
        "DRAWN",
        "DRAWN BY",
        "CHECKED",
        "APPROVED",
        "CLIENT",
        "JOB",
        "PROJECT",
        "TITLE",
        "DRAWING",
        "DWG",
        "DRG",
        "SHEET",
        "REVISION",
    ],
)
def test_draughtsperson_conflict(text):
    assert DRAUGHTSPERSON_CONFLICT_RE.search(text) is not None


def test_draughtsperson_conflict_real_initials():
    assert DRAUGHTSPERSON_CONFLICT_RE.search("JSM") is None


# ============================================================
# is_vague_scale
# ============================================================

@pytest.mark.parametrize(
    "text",
    [
        "AS SHOWN",
        "AS NOTED",
        "VARIOUS",
        "Scale: AS SHOWN",
        "various",
    ],
)
def test_is_vague_scale_true(text):
    assert is_vague_scale(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "1:100",
        "1:50",
        "FULL SIZE",
        "NTS",
    ],
)
def test_is_vague_scale_false(text):
    assert is_vague_scale(text) is False


def test_is_vague_scale_none():
    assert is_vague_scale(None) is False


def test_is_vague_scale_empty():
    assert is_vague_scale("") is False


# ============================================================
# alpha_ratio
# ============================================================

def test_alpha_ratio_all_letters():
    assert alpha_ratio("ABCDEF") == pytest.approx(1.0)


def test_alpha_ratio_spaces_ignored():
    assert alpha_ratio("ABC DEF") == pytest.approx(1.0)


def test_alpha_ratio_half_letters():
    assert alpha_ratio("AB12") == pytest.approx(0.5)


def test_alpha_ratio_no_letters():
    assert alpha_ratio("12345") == pytest.approx(0.0)


def test_alpha_ratio_empty():
    assert alpha_ratio("") == 0.0


def test_alpha_ratio_only_spaces():
    assert alpha_ratio("     ") == 0.0


def test_alpha_ratio_mixed():
    result = alpha_ratio("ABC123")

    assert result == pytest.approx(0.5)


# ============================================================
# normalize_confusions
# ============================================================

def test_normalize_confusions_non_numeric_field_unchanged():
    text = "S1TE PLAN"

    result = normalize_confusions(
        text,
        field="project_name",
    )

    assert result == text


def test_normalize_confusions_safe_word():
    result = normalize_confusions(
        "SITE",
        field="drawing_number",
    )

    # SITE is explicitly protected.
    assert result == "SITE"


def test_normalize_confusions_scale_safe_word():
    result = normalize_confusions(
        "SCALE",
        field="scale",
    )

    assert result == "SCALE"


def test_normalize_confusions_digit_context():
    result = normalize_confusions(
        "I23",
        field="drawing_number",
    )

    assert result == "123"


def test_normalize_confusions_letter_o_near_digit():
    result = normalize_confusions(
        "O23",
        field="drawing_number",
    )

    assert result == "023"


def test_normalize_confusions_letter_s_near_digit():
    result = normalize_confusions(
        "S23",
        field="drawing_number",
    )

    assert result == "523"


def test_normalize_confusions_letter_b_near_digit():
    result = normalize_confusions(
        "B23",
        field="drawing_number",
    )

    assert result == "823"


def test_normalize_confusions_d_near_digit():
    result = normalize_confusions(
        "D23",
        field="drawing_number",
    )

    assert result == "023"


def test_normalize_confusions_lowercase_i():
    result = normalize_confusions(
        "l23",
        field="drawing_number",
    )

    assert result == "123"


def test_normalize_confusions_lowercase_o():
    result = normalize_confusions(
        "o23",
        field="drawing_number",
    )

    assert result == "023"


def test_normalize_confusions_no_confusion_characters():
    text = "12345"

    assert normalize_confusions(
        text,
        field="drawing_number",
    ) == text


def test_normalize_confusions_normal_sentence_without_field():
    text = "HELLO WORLD"

    # Multiple alphabetic words and no digits:
    # conservative mode should leave it unchanged.
    assert normalize_confusions(text) == text


def test_normalize_confusions_empty():
    assert normalize_confusions("") == ""


def test_normalize_confusions_numeric_field_short_word():
    result = normalize_confusions(
        "I",
        field="drawing_number",
    )

    assert result == "1"


# ============================================================
# confusion_variants
# ============================================================

def test_confusion_variants_contains_original():
    result = confusion_variants(
        "I23",
        field="drawing_number",
    )

    assert "I23" in result


def test_confusion_variants_contains_single_correction():
    result = confusion_variants(
        "I23",
        field="drawing_number",
    )

    assert "123" in result


def test_confusion_variants_multiple_confusions():
    result = confusion_variants(
        "IO",
        field="drawing_number",
    )

    assert "IO" in result
    assert "1O" in result
    assert "I0" in result


def test_confusion_variants_respects_max_n():
    result = confusion_variants(
        "IOSBDIOSBD",
        field="drawing_number",
        max_n=3,
    )

    # Original + limited variants.
    # normalize_confusions may add one additional final variant.
    assert len(result) <= 4


def test_confusion_variants_no_confusions():
    result = confusion_variants(
        "12345",
        field="drawing_number",
    )

    assert result == ["12345"]


def test_confusion_variants_original_first():
    text = "I23"

    result = confusion_variants(
        text,
        field="drawing_number",
    )

    assert result[0] == text


# ============================================================
# more_complete
# ============================================================

def test_more_complete_extra_digit():
    ok, extra = more_complete(
        "603",
        "603/A1",
    )

    assert ok is True
    assert extra == "/A1"


def test_more_complete_extra_narrow_character():
    ok, extra = more_complete(
        "FOOT",
        "I FOOT",
    )

    assert ok is True
    assert extra == "I"


def test_more_complete_same_text():
    ok, extra = more_complete(
        "603",
        "603",
    )

    assert ok is False
    assert extra == ""


def test_more_complete_alt_shorter():
    ok, extra = more_complete(
        "603/A1",
        "603",
    )

    assert ok is False
    assert extra == ""


def test_more_complete_main_not_inside_alt():
    ok, extra = more_complete(
        "603",
        "ABC/A1",
    )

    assert ok is False
    assert extra == ""


def test_more_complete_empty_main():
    ok, extra = more_complete(
        "",
        "603/A1",
    )

    assert ok is False
    assert extra == ""


def test_more_complete_empty_alt():
    ok, extra = more_complete(
        "603",
        "",
    )

    assert ok is False
    assert extra == ""


def test_more_complete_none_values():
    ok, extra = more_complete(
        None,
        None,
    )

    assert ok is False
    assert extra == ""


def test_more_complete_extra_too_long():
    ok, extra = more_complete(
        "603",
        "603ABCDEFG",
    )

    assert ok is False
    assert extra == ""


def test_more_complete_non_narrow_extra():
    ok, extra = more_complete(
        "603",
        "AB603",
    )

    assert ok is False
    assert extra == ""


# ============================================================
# scan_candidates
# ============================================================

def test_scan_candidates_finds_more_complete():
    candidate1 = SimpleNamespace(
        text="603"
    )

    candidate2 = SimpleNamespace(
        text="603/A1"
    )

    token = SimpleNamespace(
        text="603",
        candidates=[
            candidate1,
            candidate2,
        ],
    )

    candidate, extra = scan_candidates(token)

    assert candidate is candidate2
    assert extra == "/A1"


def test_scan_candidates_skips_same_text():
    same = SimpleNamespace(
        text="603"
    )

    better = SimpleNamespace(
        text="603/A1"
    )

    token = SimpleNamespace(
        text="603",
        candidates=[
            same,
            better,
        ],
    )

    candidate, extra = scan_candidates(token)

    assert candidate is better
    assert extra == "/A1"


def test_scan_candidates_no_match():
    candidate = SimpleNamespace(
        text="ABC"
    )

    token = SimpleNamespace(
        text="603",
        candidates=[candidate],
    )

    result, extra = scan_candidates(token)

    assert result is None
    assert extra == ""


def test_scan_candidates_empty_candidates():
    token = SimpleNamespace(
        text="603",
        candidates=[],
    )

    result, extra = scan_candidates(token)

    assert result is None
    assert extra == ""


def test_scan_candidates_missing_candidates_attribute():
    token = SimpleNamespace(
        text="603"
    )

    result, extra = scan_candidates(token)

    assert result is None
    assert extra == ""


def test_scan_candidates_first_valid_candidate():
    first = SimpleNamespace(
        text="I FOOT"
    )

    second = SimpleNamespace(
        text="1 FOOT"
    )

    token = SimpleNamespace(
        text="FOOT",
        candidates=[
            first,
            second,
        ],
    )

    result, extra = scan_candidates(token)

    assert result is first
    assert extra == "I"