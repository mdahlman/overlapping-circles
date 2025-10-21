"""
Small playground script for DualPlus.

Run with:
  python -m overlapping_circles_core.dualplus_play

This constructs a simple N=2 arrangement and prints:
- summary()
- generate_label()
- generate_canonical_label()
"""

from overlapping_circles_core.dualplus import DualPlus, Region, DualEdge, draw_dualplus


def make_n3_pair12_pair13_pair23_triple():
    # All pairs intersect, C1 and C3 intersect only within C2
    regions = {
        "r0": Region(id="r0", membership=(0, 0, 0)),
        "r1": Region(id="r1", membership=(1, 0, 0)),
        "r2": Region(id="r2", membership=(0, 0, 1)),
        "r3": Region(id="r3", membership=(0, 1, 0)),
        "r4": Region(id="r4", membership=(0, 1, 0)),
        "r5": Region(id="r5", membership=(1, 1, 0)),
        "r6": Region(id="r6", membership=(0, 1, 1)),
        "r7": Region(id="r7", membership=(1, 1, 1)),
    }
    dual_edges = [
        # Circle 1 boundaries
        DualEdge(id="e1", a="r0", b="r1", circle=1),
        DualEdge(id="e2", a="r3", b="r5", circle=1),
        DualEdge(id="e3", a="r6", b="r7", circle=1),
        DualEdge(id="e4", a="r4", b="r5", circle=1),
        # Circle 2 boundaries
        DualEdge(id="e5", a="r0", b="r3", circle=2),
        DualEdge(id="e6", a="r1", b="r5", circle=2),
        DualEdge(id="e7", a="r0", b="r4", circle=2),
        DualEdge(id="e8", a="r2", b="r6", circle=2),
        # Circle 3 boundaries
        DualEdge(id="e9", a="r0", b="r2", circle=3),
        DualEdge(id="e10", a="r4", b="r6", circle=3),
        DualEdge(id="e11", a="r5", b="r7", circle=3),
        DualEdge(id="e12", a="r3", b="r6", circle=3),
    ]
    return DualPlus(
        regions=regions,
        dual_edges=dual_edges,
        containment=set(),
        W={1: [2, 3, -3, -2], 2: [3, -3, 1, -1], 3: [2, 1, -1, -2]},
    )


def main() -> None:
    dp = make_n3_pair12_pair13_pair23_triple()

    print("=== DualPlus Playground ===")
    print(f"Summary: {dp.summary()}")
    print(f"\nNaive label:     {dp.generate_label()}")
    print(f"\nCanonical label: {dp.generate_canonical_label()}")

    draw_dualplus(
        dp=dp,
        title=f"N=3 make_n3_pair12_pair13_pair23_triple\n{dp.generate_canonical_label()}",
    )


if __name__ == "__main__":
    main()
