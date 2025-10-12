from __future__ import annotations
from typing import Dict, Tuple, List
import math

from .dual import Dual

SVG = str


def _svg_header(w: int, h: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'


def _svg_footer() -> str:
    return "</svg>\n"


def render_dual_svg(d: Dual, *, w: int = 800, h: int = 600) -> SVG:
    """Topological witness: draw the *dual graph* with labeled edges.
    Deterministic layout: nodes on a circle; edges straight; labels on midpoints.
    This is always available and printable.
    """
    R = d.regions()
    n = len(R)
    cx, cy = w // 2, h // 2
    rad = int(0.45 * min(w, h))
    pos: Dict[int, Tuple[float, float]] = {}
    for i, r in enumerate(R):
        ang = 2 * math.pi * i / max(1, n)
        pos[r] = (cx + rad * math.cos(ang), cy + rad * math.sin(ang))

    out = [_svg_header(w, h)]
    out.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')

    # edges
    for u, v, lbl in d.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        out.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="black" stroke-width="1"/>'
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        out.append(
            f'<text x="{mx:.1f}" y="{my:.1f}" font-size="10" fill="blue">{lbl}</text>'
        )

    # nodes
    for r in R:
        x, y = pos[r]
        mask = d.masks[r]
        out.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="#f5f5f5" stroke="#333"/>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{y+3:.1f}" text-anchor="middle" font-size="10">{r}</text>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{y+16:.1f}" text-anchor="middle" font-size="9" fill="#666">{mask:0{d.N}b}</text>'
        )

    out.append(_svg_footer())
    return "\n".join(out)


def render_circles_svg(
    circles: List[Tuple[float, float, float]], *, w: int = 800, h: int = 600
) -> SVG:
    """Render a list of Euclidean circles (cx, cy, r) as an SVG.
    This is a geometry witness *if* you have a construction that supplies circles.
    """
    out = [_svg_header(w, h)]
    out.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    for i, (cx, cy, r) in enumerate(circles, start=1):
        out.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="none" stroke="black" stroke-width="1.5"/>'
        )
        out.append(
            f'<text x="{cx + r + 6:.1f}" y="{cy:.1f}" font-size="11" fill="#333">C{i}</text>'
        )
    out.append(_svg_footer())
    return "\n".join(out)
