from __future__ import annotations

from statistics import median

from .config import LayoutCfg
from .models import Line, Token


def split_rotated(tokens: list[Token]) -> tuple[list[Token], list[Token]]:
    if not tokens:
        return [], []
    med = median([t.h for t in tokens]) or 20
    upright = [t for t in tokens if not is_rotated(t, med)]
    rotated = [t for t in tokens if is_rotated(t, med)]
    return upright, rotated


def group_rows(tokens: list[Token], cfg: LayoutCfg) -> list[Line]:

    if not tokens:
        return []

    ts = sorted(tokens, key=lambda t: (round(t.cy / 4), t.x0))
    out, cur = [], [ts[0]]


    page_h = max(median([t.h for t in ts]), 1)
    for t in ts[1:]:
        ref_cy = median([x.cy for x in cur])
        same_y = abs(t.cy - ref_cy) <= page_h * cfg.row_y_tol
        near_x = (t.x0 - max(x.x1 for x in cur)) < page_h * cfg.row_max_x_gap
        if same_y and near_x:
            cur.append(t)
        else:
            out.append(Line(sorted(cur, key=lambda x: x.x0)))
            cur = [t]
    out.append(Line(sorted(cur, key=lambda x: x.x0)))
    out.sort(key=lambda line: (line.y0, line.x0))
    for row_no, line in enumerate(out, 1):
        line.row_no = row_no
    return out


def is_rotated(token: Token, median_h: float) -> bool:

    width, height = max(token.w, 1), max(token.h, 1)
    return height / width >= 2.0 and height >= median_h * 1.8
