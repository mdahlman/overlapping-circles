from __future__ import annotations
import pytest
from overlapping_circles_core.dual import Dual
from overlapping_circles_core.canon import canonical_code_label_invariant


def test_fixed_code_monotonic():
    d = Dual(
        N=2,
        masks={0: 0, 1: 1, 2: 2, 3: 3},
        adj={
            0: [(1, 1), (2, 2)],
            1: [(0, 1), (3, 2)],
            2: [(0, 2), (3, 1)],
            3: [(1, 2), (2, 1)],
        },
    )
    code = d._mask_struct_code_fixed_labels()
    assert code.startswith("V:00,01,10,11|E1_in:["), code


def test_label_invariant_stability():
    d = Dual(
        N=2,
        masks={0: 0, 1: 1, 2: 2, 3: 3},
        adj={
            0: [(1, 1), (2, 2)],
            1: [(0, 1), (3, 2)],
            2: [(0, 2), (3, 1)],
            3: [(1, 2), (2, 1)],
        },
    )
    canon = canonical_code_label_invariant(d)
    swapped = Dual(
        N=2,
        masks={0: 0, 1: 2, 2: 1, 3: 3},
        adj={
            0: [(1, 2), (2, 1)],
            1: [(0, 2), (3, 1)],
            2: [(0, 1), (3, 2)],
            3: [(1, 1), (2, 2)],
        },
    )
    canon_swapped = canonical_code_label_invariant(swapped)
    assert canon == canon_swapped
