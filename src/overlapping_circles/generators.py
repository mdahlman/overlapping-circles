from __future__ import annotations
import json
from pathlib import Path
from typing import List

from .dual import Dual
from .canon import canonical_code_label_invariant
from .svg_witness import render_dual_svg, render_circles_svg


def save_svg(svg: str, path: str | Path) -> None:
    Path(path).write_text(svg, encoding="utf-8")


def demo_n2() -> Dual:
    # A minimal N=2 demo dual similar to the classic overlap, with 4 regions + outside.
    masks = {
        0: 0b00,  # outside
        1: 0b01,  # C1 only
        2: 0b10,  # C2 only
        3: 0b11,  # both
    }
    # adj lists with labels (undirected stored twice)
    adj = {
        0: [(1, 1), (2, 2)],
        1: [(0, 1), (3, 2)],
        2: [(0, 2), (3, 1)],
        3: [(1, 2), (2, 1)],
    }
    return Dual(N=2, masks=masks, adj=adj)


def main():
    d = demo_n2()
    code = canonical_code_label_invariant(d)
    print("canonical:", code)
    svg = render_dual_svg(d)
    save_svg(svg, "dual_demo.svg")
    # Example of geometric witness if circles are known:
    geom_svg = render_circles_svg([(320, 300, 160), (480, 300, 160)])
    save_svg(geom_svg, "circles_demo.svg")


if __name__ == "__main__":
    main()
