import pytest
from sympy import sqrt, S, EmptySet, Union
from overlapping_circles.arrangement import Circle, regions_for_arrangement


@pytest.fixture
def venn_three():
    # Classic equilateral 3-circle Venn layout
    return [
        Circle(S(0), S(0), S(1)),
        Circle(S(1), S(0), S(1)),
        Circle(S(1) / 2, sqrt(3) / S(2), S(1)),
    ]


def test_all_masks_present_including_outside(venn_three):
    regions = regions_for_arrangement(venn_three, include_outside=True)
    masks = {m for m, r in regions.items() if r != EmptySet}
    print("Masks with non-empty regions:", masks)
    expected = {"000", "001", "010", "011", "100", "101", "110", "111"}
    assert expected.issubset(masks)


def test_region_types_and_splitting(venn_three):
    regions = regions_for_arrangement(venn_three, include_outside=True)

    # Triple overlap should exist and not simplify to a Union
    reg_111 = regions["111"]
    assert reg_111 != EmptySet
    assert not isinstance(reg_111, Union)

    # A two-circle overlap minus the third may split
    reg_110 = regions["110"]
    if isinstance(reg_110, Union):
        assert len(reg_110.args) >= 2
    else:
        assert reg_110 != EmptySet


def test_outside_region_present(venn_three):
    regions = regions_for_arrangement(venn_three, include_outside=True)
    reg_000 = regions["000"]
    assert reg_000 != EmptySet
    # Should be represented as a Complement of the ambient square minus the union of circles
    assert "Complement" in reg_000.__class__.__name__


def test_outside_region_empty_if_not_requested(venn_three):
    regions = regions_for_arrangement(venn_three, include_outside=False)
    assert regions["000"] == EmptySet
