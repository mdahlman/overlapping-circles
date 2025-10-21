import pytest

from overlapping_circles_core.dualplus import DualEdge, DualPlus, Region


def make_n3_disjoint():
    # Three disjoint circles: no intersections, no containment
    regions = {
        "r0": Region(id="r0", membership=(0, 0, 0)),  # outside all
        "r1": Region(id="r1", membership=(1, 0, 0)),  # inside C1 only
        "r2": Region(id="r2", membership=(0, 1, 0)),  # inside C2 only
        "r3": Region(id="r3", membership=(0, 0, 1)),  # inside C3 only
    }
    dual_edges = [
        # Circle 1 boundaries
        DualEdge(id="e1", a="r0", b="r1", circle=1),
        # Circle 2 boundaries
        DualEdge(id="e2", a="r0", b="r2", circle=2),
        # Circle 3 boundaries
        DualEdge(id="e3", a="r0", b="r3", circle=3),
    ]
    return DualPlus(
        regions=regions,
        dual_edges=dual_edges,
        containment=set(),
        W={1: [], 2: [], 3: []},
    )


def make_n3_pair12_isolated3():
    # Circles 1 and 2 intersect; circle 3 is isolated (no intersections, no containment)
    # Note: the labels above are NOT the labels that generate_canonical_label produces
    # Naive label:     W1:[+2,-2];W2:[+1,-1];W3:[];C:-
    # Canonical label: W1:[];W2:[+3,-3];W3:[+2,-2];C:-
    regions = {
        "r0": Region(id="r0", membership=(0, 0, 0)),  # outside all
        "r1": Region(id="r1", membership=(1, 0, 0)),  # inside C1 only
        "r2": Region(id="r2", membership=(0, 1, 0)),  # inside C2 only
        "r3": Region(id="r3", membership=(1, 1, 0)),  # inside C1 & C2
        "r4": Region(id="r4", membership=(0, 0, 1)),  # inside C3 only
    }
    dual_edges = [
        # Circle 1 boundaries
        DualEdge(id="e1", a="r0", b="r1", circle=1),
        DualEdge(id="e2", a="r2", b="r3", circle=1),
        # Circle 2 boundaries
        DualEdge(id="e3", a="r0", b="r2", circle=2),
        DualEdge(id="e4", a="r1", b="r3", circle=2),
        # Circle 3 boundaries
        DualEdge(id="e5", a="r0", b="r4", circle=3),
    ]
    return DualPlus(
        regions=regions,
        dual_edges=dual_edges,
        containment=set(),
        W={1: [2, -2], 2: [1, -1], 3: []},
    )


def make_n3_pair12_pair23():
    # One middle circle intersecting two disjoint circles
    regions = {
        "r0": Region(id="r0", membership=(0, 0, 0)),  # outside all
        "r1": Region(id="r1", membership=(1, 0, 0)),  # inside C1 only
        "r2": Region(id="r2", membership=(1, 1, 0)),  # inside C1 and C2
        "r3": Region(id="r3", membership=(0, 1, 0)),  # inside C2 only
        "r4": Region(id="r4", membership=(0, 1, 1)),  # inside C2 and C3
        "r5": Region(id="r5", membership=(0, 0, 1)),  # inside C3 only
    }
    dual_edges = [
        # Circle 1 boundaries
        DualEdge(id="e1", a="r0", b="r1", circle=1),
        DualEdge(id="e2", a="r2", b="r3", circle=1),
        # Circle 2 boundaries
        DualEdge(id="e3", a="r0", b="r3", circle=2),
        DualEdge(id="e4", a="r4", b="r5", circle=2),
        DualEdge(id="e5", a="r3", b="r0", circle=2),
        DualEdge(id="e6", a="r1", b="r2", circle=2),
        # Circle 3 boundaries
        DualEdge(id="e7", a="r0", b="r5", circle=3),
        DualEdge(id="e8", a="r4", b="r3", circle=3),
    ]
    return DualPlus(
        regions=regions,
        dual_edges=dual_edges,
        containment=set(),
        W={1: [2, -2], 2: [3, -3, 1, -1], 3: [2, -2]},
    )


def make_n3_classic_venn():
    # desc
    regions = {
        "r0": Region(id="r0", membership=(0, 0, 0)),  # outside all
        "r1": Region(id="r1", membership=(1, 0, 0)),  # inside C1 only
        "r2": Region(id="r2", membership=(1, 1, 0)),  # inside C1 and C2
        "r3": Region(id="r3", membership=(0, 1, 0)),  # inside C2 only
        "r4": Region(id="r4", membership=(1, 0, 1)),  #
        "r5": Region(id="r5", membership=(1, 1, 1)),  #
        "r6": Region(id="r6", membership=(0, 1, 1)),  #
        "r7": Region(id="r7", membership=(0, 0, 1)),  # inside C3 only
    }
    dual_edges = [
        # Circle 1 boundaries
        DualEdge(id="e1", a="r0", b="r1", circle=1),
        DualEdge(id="e2", a="r2", b="r3", circle=1),
        DualEdge(id="e3", a="r5", b="r6", circle=1),
        DualEdge(id="e4", a="r4", b="r7", circle=1),
        # Circle 2 boundaries
        DualEdge(id="e5", a="r0", b="r3", circle=2),
        DualEdge(id="e6", a="r6", b="r7", circle=2),
        DualEdge(id="e7", a="r4", b="r5", circle=2),
        DualEdge(id="e8", a="r1", b="r2", circle=2),
        # Circle 3 boundaries
        DualEdge(id="e9", a="r2", b="r4", circle=3),
        DualEdge(id="e10", a="r3", b="r6", circle=3),
        DualEdge(id="e11", a="r0", b="r7", circle=3),
        DualEdge(id="e12", a="r1", b="r4", circle=3),
    ]
    return DualPlus(
        regions=regions,
        dual_edges=dual_edges,
        containment=set(),
        W={1: [2, 3, -2, -3], 2: [3, 1, -3, -1], 3: [-1, -2, 1, 2]},
    )


def make_n3_middle_void():
    # Three circles arranged so that each pair overlaps but there is no triple-overlap region
    regions = {
        "r0": Region(id="r0", membership=(0, 0, 0)),  # outside all (central void)
        "r1": Region(id="r1", membership=(1, 0, 0)),  # inside C1 only
        "r2": Region(id="r2", membership=(0, 1, 0)),  # inside C2 only
        "r3": Region(id="r3", membership=(0, 0, 1)),  # inside C3 only
        "r4": Region(id="r4", membership=(1, 1, 0)),  # inside C1 & C2
        "r5": Region(id="r5", membership=(1, 0, 1)),  # inside C1 & C3
        "r6": Region(id="r6", membership=(0, 1, 1)),  # inside C2 & C3
        "r7": Region(id="r7", membership=(0, 0, 0)),  # middle void
    }
    dual_edges = [
        # Circle 1 boundaries
        DualEdge(id="e1", a="r0", b="r1", circle=1),
        DualEdge(id="e2", a="r2", b="r4", circle=1),
        DualEdge(id="e3", a="r7", b="r1", circle=1),
        DualEdge(id="e4", a="r3", b="r5", circle=1),
        # Circle 2 boundaries
        DualEdge(id="e5", a="r0", b="r2", circle=2),
        DualEdge(id="e6", a="r1", b="r4", circle=2),
        DualEdge(id="e7", a="r7", b="r2", circle=2),
        DualEdge(id="e8", a="r3", b="r6", circle=2),
        # Circle 3 boundaries
        DualEdge(id="e9", a="r0", b="r3", circle=3),
        DualEdge(id="e10", a="r1", b="r5", circle=3),
        DualEdge(id="e11", a="r7", b="r3", circle=3),
        DualEdge(id="e12", a="r2", b="r6", circle=3),
    ]
    return DualPlus(
        regions=regions,
        dual_edges=dual_edges,
        containment=set(),
        W={1: [2, -2, 3, -3], 2: [1, -1, 3, -3], 3: [1, -1, 2, -2]},
    )


def make_n3_nested12_nested13():
    # C1 is the outer circle; C2 and C3 are inside C1 and disjoint from each other
    regions = {
        "r0": Region(id="r0", membership=(0, 0, 0)),  # outside all
        "r1": Region(id="r1", membership=(1, 0, 0)),  # inside C1 only
        "r2": Region(id="r2", membership=(1, 1, 0)),  # inside C1 & C2
        "r3": Region(id="r3", membership=(1, 0, 1)),  # inside C1 & C3
    }
    dual_edges = [
        # Circle 1 boundaries
        DualEdge(id="e1", a="r0", b="r1", circle=1),
        # Circle 2 boundaries
        DualEdge(id="e2", a="r1", b="r2", circle=2),
        # Circle 3 boundaries
        DualEdge(id="e3", a="r1", b="r3", circle=3),
    ]
    return DualPlus(
        regions=regions,
        dual_edges=dual_edges,
        containment={(2, 1), (3, 1)},  # C2 ⊂ C1 and C3 ⊂ C1
        W={1: [], 2: [], 3: []},
    )


def make_n3_nested12_nested13_pair_23():
    # C1 is the outer circle; C2 and C3 are inside C1 and intersect each other
    regions = {
        "r0": Region(id="r0", membership=(0, 0, 0)),  # outside all
        "r1": Region(id="r1", membership=(1, 0, 0)),  # inside C1 only
        "r2": Region(id="r2", membership=(1, 1, 0)),  # inside C1 & C2
        "r3": Region(id="r3", membership=(1, 0, 1)),  # inside C1 & C3
        "r4": Region(id="r4", membership=(1, 1, 1)),  # inside C1 & C2 & C3
    }
    dual_edges = [
        # Circle 1 boundaries
        DualEdge(id="e1", a="r0", b="r1", circle=1),
        # Circle 2 boundaries
        DualEdge(id="e2", a="r1", b="r2", circle=2),
        DualEdge(id="e3", a="r3", b="r4", circle=2),
        # Circle 3 boundaries
        DualEdge(id="e4", a="r1", b="r3", circle=3),
        DualEdge(id="e5", a="r2", b="r4", circle=3),
    ]
    return DualPlus(
        regions=regions,
        dual_edges=dual_edges,
        containment={(2, 1), (3, 1)},
        W={1: [], 2: [3, -3], 3: [2, -2]},
    )


def make_n3_nested12_nested23():
    # Circle 1 contains Circle 2; Circle 2 contains Circle 3
    regions = {
        "r0": Region(id="r0", membership=(0, 0, 0)),  # outside all
        "r1": Region(id="r1", membership=(1, 0, 0)),  # inside C1 only
        "r2": Region(id="r2", membership=(1, 1, 0)),  # inside C1 & C2
        "r3": Region(id="r3", membership=(1, 1, 1)),  # inside C1 & C2 & C3
    }
    dual_edges = [
        # Circle 1 boundaries
        DualEdge(id="e1", a="r0", b="r1", circle=1),
        # Circle 2 boundaries
        DualEdge(id="e2", a="r1", b="r2", circle=2),
        # Circle 3 boundaries
        DualEdge(id="e3", a="r2", b="r3", circle=3),
    ]
    return DualPlus(
        regions=regions,
        dual_edges=dual_edges,
        containment={(2, 1), (3, 2)},  # C2 ⊂ C1 and C3 ⊂ C2
        W={1: [], 2: [], 3: []},
    )


def make_n3_nested13_isolated2():
    # Circle 1 contains Circle 3; Circle 2 is isolated
    regions = {
        "r0": Region(id="r0", membership=(0, 0, 0)),  # outside all
        "r1": Region(id="r1", membership=(1, 0, 0)),  # inside C1 only
        "r2": Region(id="r2", membership=(0, 1, 0)),  # inside C2 only
        "r3": Region(id="r3", membership=(1, 0, 1)),  # inside C1 & C3
    }
    dual_edges = [
        # Circle 1 boundaries
        DualEdge(id="e1", a="r0", b="r1", circle=1),
        # Circle 2 boundaries
        DualEdge(id="e2", a="r0", b="r2", circle=2),
        # Circle 3 boundaries
        DualEdge(id="e3", a="r1", b="r3", circle=3),
    ]
    return DualPlus(
        regions=regions,
        dual_edges=dual_edges,
        containment={(3, 1)},  # C3 ⊂ C1
        W={1: [], 2: [], 3: []},
    )


def make_n3_nested13_nested23_pair12():
    # Circles 1 and 2 intersect; Circle 3 lies inside the lens formed by their intersection
    regions = {
        "r0": Region(id="r0", membership=(0, 0, 0)),  # outside all
        "r1": Region(id="r1", membership=(1, 0, 0)),  # inside C1 only
        "r2": Region(id="r2", membership=(0, 1, 0)),  # inside C2 only
        "r3": Region(id="r3", membership=(1, 1, 0)),  # inside C1 & C2 (outside C3)
        "r4": Region(id="r4", membership=(1, 1, 1)),  # inside C1 & C2 & C3
    }
    dual_edges = [
        # Circle 1 boundaries
        DualEdge(id="e1", a="r0", b="r1", circle=1),
        DualEdge(id="e2", a="r2", b="r3", circle=1),
        # Circle 2 boundaries
        DualEdge(id="e3", a="r0", b="r2", circle=2),
        DualEdge(id="e4", a="r1", b="r3", circle=2),
        # Circle 3 boundaries
        DualEdge(id="e5", a="r3", b="r4", circle=3),
    ]
    return DualPlus(
        regions=regions,
        dual_edges=dual_edges,
        containment={(3, 1), (3, 2)},  # C3 ⊂ C1 and C3 ⊂ C2
        W={1: [2, -2], 2: [1, -1], 3: []},
    )


def make_n3_nested13_pair12_pair23_triple():
    # desc
    regions = {
        "r0": Region(id="r0", membership=(0, 0, 0)),  # outside all
        "r1": Region(id="r1", membership=(1, 0, 0)),  # inside C1 only
        "r2": Region(id="r2", membership=(1, 0, 1)),  # inside C1 & C3
        "r3": Region(id="r3", membership=(1, 1, 1)),  # inside all 3 circles
        "r4": Region(id="r4", membership=(1, 1, 0)),  # inside C1 & C2
        "r5": Region(id="r5", membership=(0, 1, 0)),  # inside C2 only
    }
    dual_edges = [
        # Circle 1 boundaries
        DualEdge(id="e1", a="r0", b="r1", circle=1),
        DualEdge(id="e2", a="r4", b="r5", circle=1),
        # Circle 2 boundaries
        DualEdge(id="e3", a="r0", b="r5", circle=2),
        DualEdge(id="e4", a="r1", b="r4", circle=2),
        DualEdge(id="e5", a="r2", b="r3", circle=2),
        DualEdge(id="e6", a="r1", b="r4", circle=2),
        # Circle 3 boundaries
        DualEdge(id="e6", a="r1", b="r2", circle=3),
        DualEdge(id="e7", a="r4", b="r3", circle=3),
    ]
    return DualPlus(
        regions=regions,
        dual_edges=dual_edges,
        containment={(3, 1)},  # C3 ⊂ C1
        W={1: [2, -2], 2: [1, 3, -3, -1], 3: [2, -2]},
    )


def make_n3_nested13_pair12():
    # desc
    regions = {
        "r0": Region(id="r0", membership=(0, 0, 0)),  # outside all
        "r1": Region(id="r1", membership=(1, 0, 0)),  # inside C1 only
        "r2": Region(id="r2", membership=(0, 1, 0)),  # inside C2 only
        "r3": Region(id="r3", membership=(1, 1, 0)),  # inside C1 & C2
        "r4": Region(id="r4", membership=(1, 0, 1)),  # inside C1 & C3
    }
    dual_edges = [
        # Circle 1 boundaries
        DualEdge(id="e1", a="r0", b="r1", circle=1),
        DualEdge(id="e2", a="r2", b="r3", circle=1),
        # Circle 2 boundaries
        DualEdge(id="e3", a="r0", b="r2", circle=2),
        DualEdge(id="e4", a="r1", b="r3", circle=2),
        # Circle 3 boundaries
        DualEdge(id="e5", a="r1", b="r4", circle=3),
    ]
    return DualPlus(
        regions=regions,
        dual_edges=dual_edges,
        containment={(3, 1)},  # C3 ⊂ C1
        W={1: [2, -2], 2: [1, -1], 3: []},
    )


def make_n3_pair12_intersectlens12():
    # Circles 1 and 2 intersect; circle 3 overlaps [part of] the lens formed by their intersection as well as C1 and C2 individually
    regions = {
        "r0": Region(id="r0", membership=(0, 0, 0)),  # outside all
        "r1": Region(id="r1", membership=(1, 0, 0)),  # inside C1 only
        "r2": Region(id="r2", membership=(0, 1, 0)),  # inside C2 only
        "r3": Region(id="r3", membership=(1, 1, 0)),  # inside C1 & C2
        "r4": Region(id="r4", membership=(1, 1, 0)),  #
        "r5": Region(id="r5", membership=(1, 0, 1)),  #
        "r6": Region(id="r6", membership=(0, 1, 1)),  #
        "r7": Region(id="r7", membership=(1, 1, 1)),  # inside all 3 circles
    }
    dual_edges = [
        # Circle 1 boundaries
        DualEdge(id="e1", a="r0", b="r1", circle=1),
        DualEdge(id="e2", a="r2", b="r3", circle=1),
        DualEdge(id="e3", a="r6", b="r7", circle=1),
        DualEdge(id="e4", a="r2", b="r4", circle=1),
        # Circle 2 boundaries
        DualEdge(id="e5", a="r0", b="r2", circle=2),
        DualEdge(id="e6", a="r1", b="r3", circle=2),
        DualEdge(id="e7", a="r5", b="r7", circle=2),
        DualEdge(id="e8", a="r1", b="r4", circle=2),
        # Circle 3 boundaries
        DualEdge(id="e9", a="r7", b="r3", circle=3),
        DualEdge(id="e10", a="r1", b="r5", circle=3),
        DualEdge(id="e11", a="r4", b="r7", circle=3),
        DualEdge(id="e12", a="r2", b="r6", circle=3),
    ]
    return DualPlus(
        regions=regions,
        dual_edges=dual_edges,
        containment=set(),
        W={1: [2, 3, -3, -2], 2: [1, 3, -3, -1], 3: [2, -2, 1, -1]},
    )


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


@pytest.mark.parametrize(
    "builder, expected",
    [
        (make_n3_disjoint, "W1:[];W2:[];W3:[];C:-"),
        (make_n3_pair12_isolated3, "W1:[];W2:[+3,-3];W3:[+2,-2];C:-"),
        (make_n3_pair12_pair23, "W1:[+2,-2];W2:[+1,-1,+3,-3];W3:[+2,-2];C:-"),
        (
            make_n3_classic_venn,
            "W1:[+2,+3,-2,-3];W2:[+1,+3,-1,-3];W3:[+1,+2,-1,-2];C:-",
        ),
        (
            make_n3_middle_void,
            "W1:[+2,-2,+3,-3];W2:[+1,-1,+3,-3];W3:[+1,-1,+2,-2];C:-",
        ),
        (make_n3_nested12_nested13, "W1:[];W2:[];W3:[];C:1⊂2,3⊂2"),
        (make_n3_nested12_nested13_pair_23, "W1:[];W2:[+3,-3];W3:[+2,-2];C:2⊂1,3⊂1"),
        (make_n3_nested12_nested23, "W1:[];W2:[];W3:[];C:2⊂1,3⊂2"),
        (make_n3_nested13_isolated2, "W1:[];W2:[];W3:[];C:3⊂2"),
        (make_n3_nested13_nested23_pair12, "W1:[];W2:[+3,-3];W3:[+2,-2];C:1⊂2,1⊂3"),
        (
            make_n3_nested13_pair12_pair23_triple,
            "W1:[+2,+3,-3,-2];W2:[+1,-1];W3:[+1,-1];C:3⊂2",
        ),
        (make_n3_nested13_pair12, "W1:[];W2:[+3,-3];W3:[+2,-2];C:1⊂3"),
        (
            make_n3_pair12_intersectlens12,
            "W1:[+2,+3,-3,-2];W2:[+1,+3,-3,-1];W3:[+1,-1,+2,-2];C:-",
        ),
        (
            make_n3_pair12_pair13_pair23_triple,
            "W1:[+2,+3,-3,-2];W2:[+1,-1,+3,-3];W3:[+1,-1,-2,+2];C:-",
        ),
    ],
)
def test_canonical_label_n3(builder, expected):
    dp = builder()
    assert dp.generate_canonical_label() == expected
