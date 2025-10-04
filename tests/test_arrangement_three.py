from overlapping_circles.arrangement import Circle, build_arrangement


def test_three_circles_all_masks_present():
    circles = [
        Circle(0.0, 0.0, 1.0),
        Circle(1.0, 0.0, 1.0),
        Circle(0.5, 0.8660254037844386, 1.0),
    ]
    _, _, faces = build_arrangement(circles)
    masks = {f.bitmask for f in faces}

    # For this "Venn-like" placement, all 8 masks should appear (000..111).
    expected = {"000", "001", "010", "011", "100", "101", "110", "111"}
    assert expected.issubset(masks)
    # And no mask should be something else
    for m in masks:
        assert set(m).issubset({"0", "1"}) and len(m) == 3
