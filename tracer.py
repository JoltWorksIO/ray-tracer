import numpy as np
from PIL import Image


class Mesh:
    def __init__(self, vertices, indices):
        self.vertices = np.array(vertices, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.uint16)

    def __getitem__(self, i):
        base = i * 3
        a = self.vertices[base]
        b = self.vertices[base + 1]
        c = self.vertices[base + 2]
        return a, b, c

    def faces(self):
        return len(self.indices) // 3


def remap(value, src_min, src_max, dst_min, dst_max):
    return ((dst_max - dst_min) / (src_max - src_min)) * (value - src_min) + dst_min


def triple(a, b, c):
    # det(a, b, c) := <a x b, c>
    return np.dot(np.cross(a, b), c)


def intersect(origin, direction, a, b, c):
    # O + td = vAB + wAC + A
    # <=> AO = vAB + wAC - td
    # det(a, b, c) := <a x b, c>
    # V = det(AB, AC, -d)
    # V1 = det(AO, AC, -d)
    # V2 = det(AB, AO, -d)
    # V3 = det(AB, AC, AO)
    # v = V1 / V
    # w = V2 / V
    # t = V3 / V
    # u := 1 - v - w
    # u >= 0 and v >= 0 and w >= 0 and u + v + w <= 1 and t >= 0
    # Ray intersects ABC :<=> u >= 0 and v >= 0 and w >= 0 and t >= 0

    ab = b - a
    ac = c - a

    vol = triple(ab, ac, -direction)
    if np.isclose(vol, 0.0):
        return None

    ao = origin - a
    vol_beta = triple(ao, ac, -direction)
    beta = vol_beta / vol
    if beta < 0:
        return None

    vol_gamma = triple(ab, ao, -direction)
    gamma = vol_gamma / vol
    if gamma < 0:
        return None

    alpha = 1 - beta - gamma
    if alpha < 0:
        return None

    vol_t = triple(ab, ac, ao)
    t = vol_t / vol
    if t < 0:
        return None

    weights = np.array([alpha, beta, gamma])

    return t, weights


def normalize(v):
    return v / np.linalg.norm(v)


def main():
    meshes = []
    meshes.append(Mesh([[0, 0.7, 2], [0.7, -0.7, 2], [-0.7, -0.7, 2]], [0, 1, 2]))

    width = 1280
    height = 720
    image = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    aspect = width / height

    image[:] = [0, 51, 102]

    ray_origin = np.array([0, 0, 0], dtype=np.float32)

    t_far = 100.0

    for y in range(height):
        for x in range(width):
            px = remap(x + 0.5, 0, width, -1, 1) * aspect
            py = remap(y + 0.5, height, 0, -1, 1)

            ray_direction = np.array([px, py, 1], dtype=np.float32) - ray_origin
            unit_direction = normalize(ray_direction)

            t_near = t_far
            for mesh in meshes:
                for i in range(mesh.faces()):
                    a, b, c = mesh[i]

                    hit = intersect(ray_origin, unit_direction, a, b, c)

                    if hit:
                        t, weights = hit
                        if t > 0 and t < t_near:
                            t_near = t
                            color = np.array(255.0 * weights, dtype=np.uint8)
                            image[y, x] = color

    Image.fromarray(image).save("out.png")


if __name__ == "__main__":
    main()
