from types import SimpleNamespace

from unittest.mock import patch



import pytest



from uma.verify import format_check, fuse





# ============================================================

# Helpers

# ============================================================



def make_field(

    name="project_name",

    verbatim=None,

    corrected=None,

    reason=None,

    signals=None,

    confidence=None,

    status=None,

    notes=None,

):

    return SimpleNamespace(

        name=name,

        verbatim=verbatim,

        corrected=corrected,

        reason=reason,

        signals={} if signals is None else signals,

        confidence=confidence,

        status=status,

        notes=[] if notes is None else notes,

    )





def make_cfg(

    *,

    w_agreement=0.30,

    w_ocr_conf=0.30,

    w_format=0.20,

    w_llm_conf=0.20,

    agreement_after_adjudication=0.90,

    llm_conf_map=None,

    correction_conf_ceiling=0.85,

    t_high=0.80,

    t_low=0.50,

):

    return SimpleNamespace(

        w_agreement=w_agreement,

        w_ocr_conf=w_ocr_conf,

        w_format=w_format,

        w_llm_conf=w_llm_conf,

        agreement_after_adjudication=agreement_after_adjudication,

        llm_conf_map=(

            {

                "high": 1.0,

                "medium": 0.6,

                "low": 0.2,

            }

            if llm_conf_map is None

            else llm_conf_map

        ),

        correction_conf_ceiling=correction_conf_ceiling,

        t_high=t_high,

        t_low=t_low,

    )





# ============================================================

# format_check - scale

# ============================================================



def test_format_check_scale_valid():

    fv = make_field(

        name="scale",

        verbatim="1:100",

    )



    assert format_check(fv) == 1.0





def test_format_check_scale_invalid():

    fv = make_field(

        name="scale",

        verbatim="HELLO WORLD",

    )



    assert format_check(fv) == 0.0





def test_format_check_prefers_corrected():

    fv = make_field(

        name="scale",

        verbatim="invalid",

        corrected="1:50",

    )



    assert format_check(fv) == 1.0





def test_format_check_empty_scale():

    fv = make_field(

        name="scale",

        verbatim=None,

        corrected=None,

    )



    assert format_check(fv) == 0.0





# ============================================================

# format_check - date

# ============================================================



def test_format_check_date_valid():

    fv = make_field(

        name="date",

        verbatim="12/05/1950",

    )



    assert format_check(fv) == 1.0





def test_format_check_date_lower_boundary():

    fv = make_field(

        name="date",

        verbatim="1900",

    )



    assert format_check(fv) == 1.0





def test_format_check_date_upper_boundary():

    fv = make_field(

        name="date",

        verbatim="2000",

    )



    assert format_check(fv) == 1.0





def test_format_check_date_below_range():

    fv = make_field(

        name="date",

        verbatim="1899",

    )



    assert format_check(fv) == 0.0





def test_format_check_date_above_range():

    fv = make_field(

        name="date",

        verbatim="2001",

    )



    assert format_check(fv) == 0.0





def test_format_check_date_no_year():

    fv = make_field(

        name="date",

        verbatim="unknown date",

    )



    assert format_check(fv) == 0.0





def test_format_check_date_custom_year_range():

    fv = make_field(

        name="date",

        verbatim="2025",

    )



    assert format_check(

        fv,

        year_range=(2000, 2030),

    ) == 1.0





def test_format_check_date_outside_custom_range():

    fv = make_field(

        name="date",

        verbatim="1999",

    )



    assert format_check(

        fv,

        year_range=(2000, 2030),

    ) == 0.0





# ============================================================

# format_check - drawing_number

# ============================================================



def test_format_check_drawing_number_valid():

    fv = make_field(

        name="drawing_number",

        verbatim="A101",

    )



    with patch(

        "uma.verify.is_drawing_number",

        return_value=True,

    ) as mock_check:

        result = format_check(fv)



    assert result == 1.0

    mock_check.assert_called_once_with("A101")





def test_format_check_drawing_number_invalid():

    fv = make_field(

        name="drawing_number",

        verbatim="INVALID",

    )



    with patch(

        "uma.verify.is_drawing_number",

        return_value=False,

    ):

        assert format_check(fv) == 0.0





# ============================================================

# format_check - draughtsperson

# ============================================================



def test_format_check_draughtsperson_valid():

    fv = make_field(

        name="draughtsperson",

        verbatim="John Smith",

    )



    assert format_check(fv) == 1.0





def test_format_check_draughtsperson_normalizes_spaces():

    fv = make_field(

        name="draughtsperson",

        verbatim="  John     Smith  ",

    )



    assert format_check(fv) == 1.0





def test_format_check_draughtsperson_empty():

    fv = make_field(

        name="draughtsperson",

        verbatim="   ",

    )



    assert format_check(fv) == 0.0





def test_format_check_draughtsperson_date_conflict():
    fv = make_field(
        name="draughtsperson",
        verbatim="12/05/1950",
    )

    assert format_check(fv) == 0.0

def test_format_check_draughtsperson_drawing_number_conflict():
    fv = make_field(
        name="draughtsperson",
        verbatim="A101",
    )

    assert format_check(fv) == 0.0

def test_format_check_draughtsperson_scale_conflict():
    fv = make_field(
        name="draughtsperson",
        verbatim="1:100",
    )

    assert format_check(fv) == 0.0

def test_format_check_draughtsperson_other_conflict():
    fv = make_field(
        name="draughtsperson",
        verbatim="DRAWN BY",
    )

    assert format_check(fv) == 0.0

def test_format_check_draughtsperson_no_letters():
    fv = make_field(
        name="draughtsperson",
        verbatim="123456",
    )

    assert format_check(fv) == 0.0

def test_format_check_other_field_returns_none():

    fv = make_field(

        name="project_name",

        verbatim="University Project",

    )



    assert format_check(fv) is None





def test_format_check_architect_returns_none():

    fv = make_field(

        name="architect",

        verbatim="ABC Architects",

    )



    assert format_check(fv) is None





# ============================================================

# fuse - evidence failure

# ============================================================



def test_fuse_evidence_failure():

    fv = make_field(

        name="scale",

        verbatim="1:100",

        signals={

            "evidence_ok": False,

        },

    )



    cfg = make_cfg()



    result = fuse(fv, cfg)



    assert result is fv

    assert fv.confidence == 0.0

    assert fv.status == "needs_review"



    assert (

        "Verbatim evidence check failed; "

        "confidence was forcibly downgraded"

        in fv.notes

    )





def test_fuse_evidence_failure_returns_early():

    fv = make_field(

        name="scale",

        verbatim="1:100",

        signals={

            "evidence_ok": False,

            "agreement": 1.0,

            "ocr_conf": 1.0,

        },

    )



    cfg = make_cfg()



    fuse(fv, cfg)



    # Despite otherwise perfect signals,

    # evidence failure forces confidence to zero.

    assert fv.confidence == 0.0

    assert fv.status == "needs_review"





# ============================================================

# fuse - no OCR value

# ============================================================



@pytest.mark.parametrize(

    "reason",

    [

        "illegible",

        "low_resolution",

    ],

)

def test_fuse_no_verbatim_rejected(reason):

    fv = make_field(

        name="scale",

        verbatim=None,

        reason=reason,

    )



    result = fuse(

        fv,

        make_cfg(),

    )



    assert result is fv

    assert fv.confidence == 0.0

    assert fv.status == "rejected"





@pytest.mark.parametrize(

    "reason",

    [

        "not_present",

        "ambiguous",

        "requires_visual",

        "other_reason",

    ],

)

def test_fuse_no_verbatim_needs_review(reason):

    fv = make_field(

        name="scale",

        verbatim=None,

        reason=reason,

    )



    fuse(

        fv,

        make_cfg(),

    )



    assert fv.confidence == 0.0

    assert fv.status == "needs_review"





def test_fuse_no_verbatim_without_reason_continues():

    fv = make_field(

        name="project_name",

        verbatim=None,

        reason=None,

        signals={},

    )



    fuse(

        fv,

        make_cfg(),

    )



    assert fv.confidence == 0.0

    assert fv.status == "needs_review"





# ============================================================

# fuse - agreement

# ============================================================



def test_fuse_agreement_only():

    fv = make_field(

        name="project_name",

        verbatim="Project",

        signals={

            "agreement": 0.9,

        },

    )



    cfg = make_cfg()



    fuse(fv, cfg)



    # With only one weighted component,

    # final confidence equals that component.

    assert fv.confidence == 0.9

    assert fv.status == "auto_accept"





def test_fuse_agreement_string_converted_to_float():

    fv = make_field(

        name="project_name",

        verbatim="Project",

        signals={

            "agreement": "0.75",

        },

    )



    fuse(

        fv,

        make_cfg(),

    )



    assert fv.confidence == 0.75

    assert fv.status == "needs_review"





def test_fuse_agreement_adjudicated_adjusted():

    fv = make_field(

        name="project_name",

        verbatim="Project",

        signals={

            "agreement": 0.4,

            "adjudicated": True,

        },

    )



    cfg = make_cfg(

        agreement_after_adjudication=0.9,

    )



    fuse(fv, cfg)



    assert fv.signals[

        "agreement_adjusted"

    ] == 0.9



    assert fv.confidence == 0.9





def test_fuse_agreement_adjudicated_keeps_higher_value():

    fv = make_field(

        name="project_name",

        verbatim="Project",

        signals={

            "agreement": 0.95,

            "adjudicated": True,

        },

    )



    cfg = make_cfg(

        agreement_after_adjudication=0.9,

    )



    fuse(fv, cfg)



    # Adjustment only runs when agreement < 1,

    # but max() preserves 0.95.

    assert fv.signals[

        "agreement_adjusted"

    ] == 0.95



    assert fv.confidence == 0.95





def test_fuse_perfect_agreement_not_adjusted():

    fv = make_field(

        name="project_name",

        verbatim="Project",

        signals={

            "agreement": 1.0,

            "adjudicated": True,

        },

    )



    fuse(

        fv,

        make_cfg(),

    )



    assert (

        "agreement_adjusted"

        not in fv.signals

    )



    assert fv.confidence == 1.0





def test_fuse_not_adjudicated_not_adjusted():

    fv = make_field(

        name="project_name",

        verbatim="Project",

        signals={

            "agreement": 0.4,

            "adjudicated": False,

        },

    )



    fuse(

        fv,

        make_cfg(),

    )



    assert (

        "agreement_adjusted"

        not in fv.signals

    )



    assert fv.confidence == 0.4





# ============================================================

# fuse - OCR confidence

# ============================================================



def test_fuse_ocr_conf_only():

    fv = make_field(

        name="project_name",

        verbatim="Project",

        signals={

            "ocr_conf": 0.85,

        },

    )



    fuse(

        fv,

        make_cfg(),

    )



    assert fv.confidence == 0.85

    assert fv.status == "auto_accept"





def test_fuse_ocr_conf_string():

    fv = make_field(

        name="project_name",

        verbatim="Project",

        signals={

            "ocr_conf": "0.65",

        },

    )



    fuse(

        fv,

        make_cfg(),

    )



    assert fv.confidence == 0.65

    assert fv.status == "needs_review"





# ============================================================

# fuse - format

# ============================================================



def test_fuse_valid_format_added_to_signals():

    fv = make_field(

        name="scale",

        verbatim="1:100",

        signals={},

    )



    fuse(

        fv,

        make_cfg(),

    )



    assert fv.signals[

        "format"

    ] == 1.0



    assert fv.confidence == 1.0

    assert fv.status == "auto_accept"





def test_fuse_invalid_format():

    fv = make_field(

        name="scale",

        verbatim="HELLO WORLD",

        signals={},

    )



    fuse(

        fv,

        make_cfg(),

    )



    assert fv.signals[

        "format"

    ] == 0.0



    assert fv.confidence == 0.0

    assert fv.status == "rejected"





def test_fuse_field_without_format_check():

    fv = make_field(

        name="project_name",

        verbatim="Project",

        signals={},

    )



    fuse(

        fv,

        make_cfg(),

    )



    assert "format" not in fv.signals

    assert fv.confidence == 0.0

    assert fv.status == "needs_review"





# ============================================================

# fuse - LLM confidence

# ============================================================



@pytest.mark.parametrize(

    "llm_value,expected",

    [

        ("high", 1.0),

        ("HIGH", 1.0),

        ("medium", 0.6),

        ("MEDIUM", 0.6),

        ("low", 0.2),

        ("LOW", 0.2),

    ],

)

def test_fuse_llm_confidence_mapping(

    llm_value,

    expected,

):

    fv = make_field(

        name="project_name",

        verbatim="Project",

        signals={

            "llm_confidence": llm_value,

        },

    )



    fuse(

        fv,

        make_cfg(),

    )



    assert fv.confidence == expected





def test_fuse_unknown_llm_confidence_defaults_half():

    fv = make_field(

        name="project_name",

        verbatim="Project",

        signals={

            "llm_confidence": "unknown",

        },

    )



    fuse(

        fv,

        make_cfg(),

    )



    assert fv.confidence == 0.5

    assert fv.status == "needs_review"





def test_fuse_empty_llm_confidence_ignored():

    fv = make_field(

        name="project_name",

        verbatim="Project",

        signals={

            "llm_confidence": "",

        },

    )



    fuse(

        fv,

        make_cfg(),

    )



    assert fv.confidence == 0.0

    assert fv.status == "needs_review"





def test_fuse_none_llm_confidence_ignored():

    fv = make_field(

        name="project_name",

        verbatim="Project",

        signals={

            "llm_confidence": None,

        },

    )



    fuse(

        fv,

        make_cfg(),

    )



    assert fv.confidence == 0.0

    assert fv.status == "needs_review"





# ============================================================

# fuse - weighted calculation

# ============================================================



def test_fuse_weighted_calculation():

    fv = make_field(

        name="scale",

        verbatim="1:100",

        signals={

            "agreement": 0.8,

            "ocr_conf": 0.6,

            "llm_confidence": "high",

        },

    )



    cfg = make_cfg(

        w_agreement=0.30,

        w_ocr_conf=0.30,

        w_format=0.20,

        w_llm_conf=0.20,

    )



    fuse(fv, cfg)



    # (0.3*0.8 + 0.3*0.6 +

    #  0.2*1.0 + 0.2*1.0) / 1.0

    expected = 0.82



    assert fv.confidence == expected

    assert fv.status == "auto_accept"





def test_fuse_weighted_calculation_rounding():

    fv = make_field(

        name="project_name",

        verbatim="Project",

        signals={

            "agreement": 0.7777,

            "ocr_conf": 0.6666,

        },

    )



    cfg = make_cfg(

        w_agreement=0.5,

        w_ocr_conf=0.5,

    )



    fuse(fv, cfg)



    expected = round(

        (

            0.5 * 0.7777

            + 0.5 * 0.6666

        )

        / 1.0,

        3,

    )



    assert fv.confidence == expected





def test_fuse_uses_only_available_components():

    fv = make_field(

        name="project_name",

        verbatim="Project",

        signals={

            "agreement": 0.7,

        },

    )



    cfg = make_cfg(

        w_agreement=0.1,

        w_ocr_conf=0.9,

    )



    fuse(fv, cfg)



    # Missing OCR confidence is not treated

    # as zero; only agreement participates.

    assert fv.confidence == 0.7





# ============================================================

# fuse - corrected value

# ============================================================



def test_fuse_corrected_confidence_ceiling():

    fv = make_field(

        name="project_name",

        verbatim="OLD",

        corrected="NEW",

        signals={

            "agreement": 1.0,

            "ocr_conf": 1.0,

            "corrected": True,

        },

    )



    cfg = make_cfg(

        correction_conf_ceiling=0.85,

    )



    fuse(fv, cfg)



    assert fv.confidence == 0.85





def test_fuse_corrected_below_ceiling_unchanged():

    fv = make_field(

        name="project_name",

        verbatim="OLD",

        corrected="NEW",

        signals={

            "agreement": 0.7,

            "corrected": True,

        },

    )



    cfg = make_cfg(

        correction_conf_ceiling=0.85,

    )



    fuse(fv, cfg)



    assert fv.confidence == 0.7





def test_fuse_corrected_invalid_format_forces_low():

    fv = make_field(

        name="scale",

        verbatim="1:100",

        corrected="HELLO WORLD",

        signals={

            "agreement": 1.0,

            "ocr_conf": 1.0,

            "corrected": True,

        },

    )



    cfg = make_cfg(

        correction_conf_ceiling=0.95,

        t_low=0.50,

    )



    fuse(fv, cfg)



    assert fv.signals["format"] == 0.0



    assert fv.confidence <= 0.50



    assert any(

        "does not match the expected format"

        in note

        for note in fv.notes

    )





def test_fuse_corrected_valid_format_no_format_warning():

    fv = make_field(

        name="scale",

        verbatim="invalid",

        corrected="1:100",

        signals={

            "agreement": 1.0,

            "corrected": True,

        },

    )



    fuse(

        fv,

        make_cfg(),

    )



    assert fv.signals["format"] == 1.0



    assert not any(

        "does not match the expected format"

        in note

        for note in fv.notes

    )





# ============================================================

# fuse - repaired tokens

# ============================================================



def test_fuse_tokens_repaired_caps_confidence():

    fv = make_field(

        name="project_name",

        verbatim="Project",

        signals={

            "agreement": 1.0,

            "tokens_repaired": True,

        },

    )



    cfg = make_cfg(

        t_high=0.80,

    )



    fuse(fv, cfg)



    assert fv.confidence == 0.79

    assert fv.status == "needs_review"





def test_fuse_tokens_repaired_does_not_raise_low_confidence():

    fv = make_field(

        name="project_name",

        verbatim="Project",

        signals={

            "agreement": 0.4,

            "tokens_repaired": True,

        },

    )



    cfg = make_cfg(

        t_high=0.80,

        t_low=0.50,

    )



    fuse(fv, cfg)



    assert fv.confidence == 0.4

    assert fv.status == "rejected"





# ============================================================

# fuse - final status boundaries

# ============================================================



def test_fuse_status_auto_accept_at_high_boundary():

    fv = make_field(

        name="project_name",

        verbatim="Project",

        signals={

            "agreement": 0.80,

        },

    )



    cfg = make_cfg(

        t_high=0.80,

        t_low=0.50,

    )



    fuse(fv, cfg)



    assert fv.confidence == 0.80

    assert fv.status == "auto_accept"





def test_fuse_status_needs_review_below_high():

    fv = make_field(

        name="project_name",

        verbatim="Project",

        signals={

            "agreement": 0.79,

        },

    )



    cfg = make_cfg(

        t_high=0.80,

        t_low=0.50,

    )



    fuse(fv, cfg)



    assert fv.confidence == 0.79

    assert fv.status == "needs_review"





def test_fuse_status_needs_review_at_low_boundary():

    fv = make_field(

        name="project_name",

        verbatim="Project",

        signals={

            "agreement": 0.50,

        },

    )



    cfg = make_cfg(

        t_high=0.80,

        t_low=0.50,

    )



    fuse(fv, cfg)



    assert fv.confidence == 0.50

    assert fv.status == "needs_review"





def test_fuse_status_rejected_below_low():

    fv = make_field(

        name="project_name",

        verbatim="Project",

        signals={

            "agreement": 0.49,

        },

    )



    cfg = make_cfg(

        t_high=0.80,

        t_low=0.50,

    )



    fuse(fv, cfg)



    assert fv.confidence == 0.49

    assert fv.status == "rejected"





# ============================================================

# Complete cases

# ============================================================



def test_fuse_complete_high_confidence_scale():

    fv = make_field(

        name="scale",

        verbatim="1:100",

        signals={

            "evidence_ok": True,

            "agreement": 0.95,

            "ocr_conf": 0.90,

            "llm_confidence": "high",

        },

    )



    cfg = make_cfg()



    result = fuse(fv, cfg)



    assert result is fv

    assert fv.signals["format"] == 1.0

    assert fv.confidence >= cfg.t_high

    assert fv.status == "auto_accept"





def test_fuse_complete_low_confidence_scale():

    fv = make_field(

        name="scale",

        verbatim="invalid",

        signals={

            "agreement": 0.2,

            "ocr_conf": 0.2,

            "llm_confidence": "low",

        },

    )



    cfg = make_cfg()



    result = fuse(fv, cfg)



    assert result is fv

    assert fv.signals["format"] == 0.0

    assert fv.confidence < cfg.t_low

    assert fv.status == "rejected"