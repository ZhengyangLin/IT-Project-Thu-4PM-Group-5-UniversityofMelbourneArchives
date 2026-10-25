from uma.layout import group_rows
from uma.models import Token
from uma.config import LayoutCfg


def test_group_rows():
    # Create two tokens that should belong to the same row
    tokens = [
        Token(text="HELLO", bbox=(0, 0, 40, 20), conf=0.9),
        Token(text="WORLD", bbox=(50, 0, 100, 20), conf=0.9),
    ]

    cfg = LayoutCfg()

    result = group_rows(tokens, cfg)

    # Verify that both tokens are grouped into one row
    assert len(result) == 1
    assert len(result[0].tokens) == 2
    assert result[0].text == "HELLO WORLD"
    assert result[0].row_no == 1