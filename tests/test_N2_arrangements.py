import pytest

from overlapping_circles_core.dualplus import DualEdge, DualPlus, Region


def make_n2_disjoint():
    # No intersections, no containment
    regions = {
        "r0": Region(id="r0", membership=(0, 0)),  # outside both
        "r1": Region(id="r1", membership=(1, 0)),  # inside C1 only
        "r2": Region(id="r2", membership=(0, 1)),  # inside C2 only
    }
    dual_edges = [
        # Circle 1 boundary: r0(00) ↔ r1(10)
        DualEdge(id="e1", a="r0", b="r1", circle=1),
        # Circle 2 boundary: r0(00) ↔ r2(01)
        DualEdge(id="e2", a="r0", b="r2", circle=2),
    ]
    return DualPlus(
        regions=regions,
        dual_edges=dual_edges,
        containment=set(),
        W={1: [], 2: []},
    )


def make_n2_nested():
    # Circle 2 inside circle 1 (arbitrary choice)
    regions = {
        "r0": Region(id="r0", membership=(0, 0)),  # outside both
        "r1": Region(id="r1", membership=(1, 0)),  # inside C1, outside C2
        "r2": Region(id="r2", membership=(1, 1)),  # inside C1 and C2
    }
    dual_edges = [
        # Circle 1 boundary: r0(00) ↔ r1(10)
        DualEdge(id="e1", a="r0", b="r1", circle=1),
        # Circle 2 boundary (nested inside C1): r1(10) ↔ r2(11)
        DualEdge(id="e2", a="r1", b="r2", circle=2),
    ]
    return DualPlus(
        regions=regions,
        dual_edges=dual_edges,
        containment={(2, 1)},
        W={1: [], 2: []},
    )


def make_n2_intersecting():
    # One crossing, so W1 sees 2 then -2; W2 sees 1 then -1 (up to rotation/reversal)
    regions = {
        "r0": Region(id="r0", membership=(0, 0)),  # outside both
        "r1": Region(id="r1", membership=(1, 0)),  # inside C1 only
        "r2": Region(id="r2", membership=(0, 1)),  # inside C2 only
        "r3": Region(id="r3", membership=(1, 1)),  # inside both
    }
    dual_edges = [
        # circle 1 edges: r0–r1, r2–r3
        DualEdge(id="e1", a="r0", b="r1", circle=1),
        DualEdge(id="e2", a="r2", b="r3", circle=1),
        # circle 2 edges: r0–r2, r1–r3
        DualEdge(id="e3", a="r0", b="r2", circle=2),
        DualEdge(id="e4", a="r1", b="r3", circle=2),
    ]
    return DualPlus(
        regions=regions,
        dual_edges=dual_edges,
        containment=set(),
        W={1: [2, -2], 2: [1, -1]},
    )


@pytest.mark.parametrize(
    "builder, expected",
    [
        (make_n2_disjoint, "W1:[];W2:[];C:-"),
        (make_n2_nested, "W1:[];W2:[];C:2⊂1"),
        (make_n2_intersecting, "W1:[+2,-2];W2:[+1,-1];C:-"),
    ],
)
def test_canonical_label_n2(builder, expected):
    dp = builder()
    assert dp.generate_canonical_label() == expected
