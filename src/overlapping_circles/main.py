import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import string


def generate_subsets(n) -> list[str]:
    """
    Generate all subsets of n circles as bitmasks.
    Each subset is an n-length string of 0/1.
    """
    return ["".join(bits) for bits in product("01", repeat=n)]


def label_subsets(n) -> dict[str, str]:
    labels = list(string.ascii_uppercase[:n])
    subsets = generate_subsets(n)
    result = {}
    for mask in subsets:
        members = [labels[i] for i, bit in enumerate(mask) if bit == "1"]
        result[mask] = "{" + ",".join(members) + "}" if members else "∅"
    return result


def generate_circle(x, y, r=1.0, points=200) -> tuple[np.ndarray, np.ndarray]:
    """Return coordinates for a circle centered at (x,y) with radius r."""
    theta = np.linspace(0, 2 * np.pi, points)
    return x + r * np.cos(theta), y + r * np.sin(theta)


def demo_three_circles():
    # Rough positions for 3 overlapping circles
    centers = [(0, 0), (1, 0), (0.5, 0.866)]  # equilateral triangle layout
    colors = ["r", "g", "b"]

    plt.figure(figsize=(5, 5))
    for (x, y), c in zip(centers, colors):
        X, Y = generate_circle(x, y)
        plt.plot(X, Y, color=c)

    plt.gca().set_aspect("equal")
    plt.title("Three Overlapping Circles (Demo)")
    plt.show()


def main():
    # demo_three_circles()
    subsets = label_subsets(3)
    print("Subsets for 3 circles:")
    for mask, label in subsets.items():
        print(f"  {mask}: {label}")


if __name__ == "__main__":
    main()
