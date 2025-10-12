from __future__ import annotations
import pytest
from overlapping_circles.dual import Dual
from overlapping_circles.canon import canonical_code_label_invariant


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
    code = d.to_mask_struct_code_fixed_labels()
    assert code.startswith("N=2|M=0,1,2,3|L1=["), code


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
        masks={0: 0, 1: 2, 2: 1, 3: 3},  # swap mask bits 1<->2
        adj={
            0: [(1, 2), (2, 1)],
            1: [(0, 2), (3, 1)],
            2: [(0, 1), (3, 2)],
            3: [(1, 1), (2, 2)],
        },
    )
    canon_swapped = canonical_code_label_invariant(swapped)
    assert canon == canon_swapped
