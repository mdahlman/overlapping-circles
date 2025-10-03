import matplotlib.pyplot as plt
import numpy as np


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
    demo_three_circles()


if __name__ == "__main__":
    main()
