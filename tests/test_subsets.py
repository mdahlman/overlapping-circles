import pytest
from overlapping_circles_core.subsets import generate_subsets, label_subsets


def test_generate_subsets_3():
    result = generate_subsets(3)
    expected = ["000", "001", "010", "011", "100", "101", "110", "111"]
    assert result == expected


def test_label_subsets_3():
    labels = label_subsets(3)
    assert labels["000"] == "∅"
    assert labels["100"] == "{A}"
    assert labels["011"] == "{B,C}"
    assert labels["111"] == "{A,B,C}"
    assert len(labels) == 8
