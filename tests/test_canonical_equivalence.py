import pytest
from overlapping_circles_core.dual import Dual

# canonical reference strings
CANONICAL_N3_MIDCIRCLE = "V:001,010,011,100,101|E1_in:[2-3;4-5]|E1_out:[1]|E2_in:[1-3]|E2_out:[2]|E3_in:[1-5]|E3_out:[4]"
CANONICAL_N3_PAIR12_ISOLATED3 = "V:001,010,011,100|E1_in:[2-3]|E1_out:[1]|E2_in:[1-3]|E2_out:[2]|E3_in:[]|E3_out:[4]"
CANONICAL_N3_NESTED13_PAIR12 = "V:001,010,011,101|E1_in:[2-3]|E1_out:[1]|E2_in:[1-3]|E2_out:[2]|E3_in:[1-4]|E3_out:[]"
CANONICAL_N3_NESTED13_NESTED23_PAIR12 = "V:001,010,011,111|E1_in:[2-3]|E1_out:[1]|E2_in:[1-3]|E2_out:[2]|E3_in:[3-4]|E3_out:[]"
CANONICAL_N3_PAIR12_PAIR13 = "V:001,010,011,100,101|E1_in:[2-3;4-5]|E1_out:[1]|E2_in:[1-3]|E2_out:[2]|E3_in:[1-5]|E3_out:[4]"
CANONICAL_N3_NESTED13_PAIR12_PAIR23_TRIPLE = "V:001,010,011,101,111|E1_in:[2-3]|E1_out:[1]|E2_in:[1-3;4-5]|E2_out:[2]|E3_in:[1-4;3-5]|E3_out:[]"
CANONICAL_N3_CLASSIC_VENN = "V:001,010,011,100,101,110,111|E1_in:[2-3;4-5;6-7]|E1_out:[1]|E2_in:[1-3;4-6;5-7]|E2_out:[2]|E3_in:[1-5;2-6;3-7]|E3_out:[4]"
CANONICAL_N3_ALL_DISJOINT = (
    "V:001,010,100|E1_in:[]|E1_out:[1]|E2_in:[]|E2_out:[2]|E3_in:[]|E3_out:[3]"
)
CANONICAL_N3_NESTED13_ISOLATED2 = (
    "V:001,010,101|E1_in:[]|E1_out:[1]|E2_in:[]|E2_out:[2]|E3_in:[1-3]|E3_out:[]"
)
CANONICAL_N3_NESTED12_NESTED_13 = (
    "V:001,011,101|E1_in:[]|E1_out:[1]|E2_in:[1-2]|E2_out:[]|E3_in:[1-3]|E3_out:[]"
)
CANONICAL_N3_NESTED12_NESTED23 = (
    "V:001,011,111|E1_in:[]|E1_out:[1]|E2_in:[1-2]|E2_out:[]|E3_in:[2-3]|E3_out:[]"
)
CANONICAL_N3_NESTED12_NESTED13_PAIR23 = "V:001,011,101,111|E1_in:[]|E1_out:[1]|E2_in:[1-2;3-4]|E2_out:[]|E3_in:[1-3;2-4]|E3_out:[]"


def dual_from_canonical(code: str) -> Dual:
    """Reconstruct a Dual instance directly from a canonical string."""
    parts = code.split("|")
    assert parts[0].startswith("V:")
    masks = parts[0][2:].split(",")
    region_ids = {i + 1: int(m, 2) for i, m in enumerate(masks)}
    N = len(masks[0])
    adj = {i + 1: [] for i in range(len(masks))}
    adj[0] = []  # add outside region 0

    import re

    for p in parts[1:]:
        m = re.match(r"E(\d+)_(in|out):\[(.*)\]$", p)
        if not m:
            continue
        lbl = int(m.group(1))
        kind = m.group(2)
        data = m.group(3)
        if not data:
            continue
        entries = [x for x in data.split(";") if x]
        if kind == "in":
            for pair in entries:
                a, b = [int(x) for x in pair.split("-")]
                adj[a].append((b, lbl))
                adj[b].append((a, lbl))
        else:
            for r in entries:
                r = int(r)
                # add both directions so the outside region is represented
                adj[r].append((0, lbl))
                adj[0].append((r, lbl))
    return Dual(N=N, masks=region_ids, adj=adj)


def test_canonical_n3_references_are_stable():
    """Cannonical codes must not change."""
    d = dual_from_canonical(CANONICAL_N3_MIDCIRCLE)
    assert d.canonical_code() == CANONICAL_N3_MIDCIRCLE

    d = dual_from_canonical(CANONICAL_N3_PAIR12_ISOLATED3)
    assert d.canonical_code() == CANONICAL_N3_PAIR12_ISOLATED3

    d = dual_from_canonical(CANONICAL_N3_NESTED13_PAIR12)
    assert d.canonical_code() == CANONICAL_N3_NESTED13_PAIR12

    d = dual_from_canonical(CANONICAL_N3_NESTED13_NESTED23_PAIR12)
    assert d.canonical_code() == CANONICAL_N3_NESTED13_NESTED23_PAIR12

    d = dual_from_canonical(CANONICAL_N3_PAIR12_PAIR13)
    assert d.canonical_code() == CANONICAL_N3_PAIR12_PAIR13

    d = dual_from_canonical(CANONICAL_N3_NESTED13_PAIR12_PAIR23_TRIPLE)
    assert d.canonical_code() == CANONICAL_N3_NESTED13_PAIR12_PAIR23_TRIPLE

    d = dual_from_canonical(CANONICAL_N3_CLASSIC_VENN)
    assert d.canonical_code() == CANONICAL_N3_CLASSIC_VENN

    d = dual_from_canonical(CANONICAL_N3_ALL_DISJOINT)
    assert d.canonical_code() == CANONICAL_N3_ALL_DISJOINT

    d = dual_from_canonical(CANONICAL_N3_NESTED13_ISOLATED2)
    assert d.canonical_code() == CANONICAL_N3_NESTED13_ISOLATED2

    d = dual_from_canonical(CANONICAL_N3_NESTED12_NESTED_13)
    assert d.canonical_code() == CANONICAL_N3_NESTED12_NESTED_13

    d = dual_from_canonical(CANONICAL_N3_NESTED12_NESTED23)
    assert d.canonical_code() == CANONICAL_N3_NESTED12_NESTED23

    d = dual_from_canonical(CANONICAL_N3_NESTED12_NESTED13_PAIR23)
    assert d.canonical_code() == CANONICAL_N3_NESTED12_NESTED13_PAIR23


def test_isomorphic_variants_CANONICAL_N3_MIDCIRCLE():
    """Arrangement #2 (label-swapped) should canonicalize to the same string."""
    variant = "V:001,010,011,100,110|E1_in:[2-3]|E1_out:[1]|E2_in:[1-3;4-5]|E2_out:[2]|E3_in:[2-5]|E3_out:[4]"
    d = dual_from_canonical(variant)
    assert d.canonical_code() == CANONICAL_N3_MIDCIRCLE

    variant = "V:001,010,011,110|E1_in:[2-3]|E1_out:[1]|E2_in:[1-3]|E2_out:[2]|E3_in:[2-4]|E3_out:[]"
    d = dual_from_canonical(variant)
    assert d.canonical_code() == CANONICAL_N3_NESTED13_PAIR12

    variant = "V:001,010,011,100,110|E1_in:[2-3]|E1_out:[1]|E2_in:[1-3;4-5]|E2_out:[2]|E3_in:[2-5]|E3_out:[4]"
    d = dual_from_canonical(variant)
    assert d.canonical_code() == CANONICAL_N3_PAIR12_PAIR13

    variant = "V:001,010,011,110,111|E1_in:[2-3;4-5]|E1_out:[1]|E2_in:[1-3]|E2_out:[2]|E3_in:[2-4;3-5]|E3_out:[]"
    d = dual_from_canonical(variant)
    assert d.canonical_code() == CANONICAL_N3_NESTED13_PAIR12_PAIR23_TRIPLE

    variant = "V:001,010,011,100,101,111,110|E1_in:[2-3;4-5;6-7]|E1_out:[1]|E2_in:[1-3;4-7;5-6]|E2_out:[2]|E3_in:[1-5;2-7;3-6]|E3_out:[4]"
    d = dual_from_canonical(variant)
    assert d.canonical_code() == CANONICAL_N3_CLASSIC_VENN

    variant = (
        "V:001,010,110|E1_in:[]|E1_out:[1]|E2_in:[]|E2_out:[2]|E3_in:[2-3]|E3_out:[]"
    )
    d = dual_from_canonical(variant)
    assert d.canonical_code() == CANONICAL_N3_NESTED13_ISOLATED2

    variant = "V:001,010,100,101|E1_in:[3-4]|E1_out:[1]|E2_in:[]|E2_out:[2]|E3_in:[1-4]|E3_out:[3]"
    d = dual_from_canonical(variant)
    assert d.canonical_code() == CANONICAL_N3_PAIR12_ISOLATED3

    variant = "V:001,010,100,110|E1_in:[]|E1_out:[1]|E2_in:[3-4]|E2_out:[2]|E3_in:[2-4]|E3_out:[3]"
    d = dual_from_canonical(variant)
    assert d.canonical_code() == CANONICAL_N3_PAIR12_ISOLATED3

    variant = (
        "V:001,011,100|E1_in:[]|E1_out:[1]|E2_in:[1-2]|E2_out:[]|E3_in:[]|E3_out:[3]"
    )
    d = dual_from_canonical(variant)
    assert d.canonical_code() == CANONICAL_N3_NESTED13_ISOLATED2

    variant = "V:001,011,100,101|E1_in:[3-4]|E1_out:[1]|E2_in:[1-2]|E2_out:[]|E3_in:[1-4]|E3_out:[3]"
    d = dual_from_canonical(variant)
    assert d.canonical_code() == CANONICAL_N3_NESTED13_PAIR12
