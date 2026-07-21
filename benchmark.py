import numpy as np
from tracer import intersect


def test():
    origin = np.array([0, 0, 0], dtype=np.float32)
    direction = np.array([0, 0, 1], dtype=np.float32)
    triangle = np.array(
        [[0, 0.7, 2], [0.7, -0.7, 2], [-0.7, -0.7, 2]], dtype=np.float32
    )
    result = intersect(origin, direction, *triangle)
    assert result is not None


if __name__ == "__main__":
    import timeit

    iterations = 1_000_000
    t = timeit.timeit("test()", globals=locals(), number=iterations)

    avg_us = (t / iterations) * 1_000_000

    print(f"Total time: {t:.4f} s")
    print(f"Average per run: {avg_us:.4f} µs")
