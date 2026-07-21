import unittest
from tracer import intersect
import numpy as np


class Test(unittest.TestCase):

    def test_intersection(self):
        origin = np.array([0, 0, 0], dtype=np.float32)
        triangle = np.array(
            [[0, 0.7, 2], [0.7, -0.7, 2], [-0.7, -0.7, 2]], dtype=np.float32
        )

        # Ray is intersecting triangle
        direction = np.array([0, 0, 1], dtype=np.float32)
        result = intersect(origin, direction, *triangle)
        self.assertNotEqual(result, None)
        # Ray is going backwards
        direction = np.array([0, 0, -1], dtype=np.float32)
        result = intersect(origin, direction, *triangle)
        self.assertEqual(result, None)
        # Ray does not intersect triangle
        direction = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        result = intersect(origin, direction, *triangle)
        self.assertEqual(result, None)


if __name__ == "__main__":
    unittest.main()
