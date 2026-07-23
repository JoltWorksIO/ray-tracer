import numpy as np
from PIL import Image
from numba import jit


@jit(nopython=True)
def fetch_triangle(vertices, indices, i):
    base = i * 3
    ia = indices[base]
    ib = indices[base + 1]
    ic = indices[base + 2]
    a = vertices[ia]
    b = vertices[ib]
    c = vertices[ic]
    return a, b, c


@jit(nopython=True)
def get_face_count(indices):
    return len(indices) // 3


@jit(nopython=True)
def translate_vertices(vertices, x, y, z):
    t = np.array([x, y, z], dtype=np.float32)
    return [vertex + t for vertex in vertices]


class Mesh:
    def __init__(self, vertices, indices):
        self.vertices = np.array(vertices, dtype=np.float32, copy=True)
        self.indices = np.array(indices, dtype=np.uint16, copy=True)

    def __getitem__(self, i):
        return fetch_triangle(self.vertices, self.indices, i)

    def faces(self):
        return get_face_count(self.indices)

    def translate(self, x=0, y=0, z=0):
        return Mesh(translate_vertices(self.vertices, x, y, z), self.indices)


@jit(nopython=True)
def remap(value, src_min, src_max, dst_min, dst_max):
    return ((dst_max - dst_min) / (src_max - src_min)) * (value - src_min) + dst_min


@jit(nopython=True)
def triple(a, b, c):
    # det(a, b, c) := <a x b, c>
    return np.dot(np.cross(a, b), c)


@jit(nopython=True)
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


@jit(nopython=True)
def normalize(v):
    return v / np.linalg.norm(v)


@jit(nopython=True)
def intersect_mesh(ray_origin, ray_direction, t_min, t_max, vertices, indices):
    t_closest = t_max
    weights_closest = None
    for i in range(get_face_count(indices)):
        # Find intersection with the ray
        a, b, c = fetch_triangle(vertices, indices, i)
        hit = intersect(ray_origin, ray_direction, a, b, c)

        # Keep if closest
        if hit is not None:
            t, uvw = hit
            if t_min <= t < t_closest:
                t_closest = t
                weights_closest = uvw
    return t_closest, weights_closest


def trace(ray_origin, ray_direction, t_min, t_max, meshes):
    # For every object in the scene
    t_closest = t_max
    weights_closest = None
    for mesh in meshes:
        t_hit, weights = intersect_mesh(
            ray_origin, ray_direction, t_min, t_closest, mesh.vertices, mesh.indices
        )
        if weights is not None and t_hit < t_closest:
            t_closest = t_hit
            weights_closest = weights
    return t_closest, weights_closest


def main():
    meshes = []
    triangle = Mesh([[0, 0.7, 2], [0.7, -0.7, 2], [-0.7, -0.7, 2]], [0, 1, 2])
    meshes.append(triangle)
    meshes.append(triangle.translate(2, -1, 4))

    width = 1280
    height = 720
    image = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    aspect = width / height

    image[:] = [0, 51, 102]

    ray_origin = np.array([0, 0, 0], dtype=np.float32)

    t_min = 0.1
    t_max = 100.0

    # For every pixel
    for y in range(height):
        for x in range(width):
            # Construct a ray from the eye
            px = remap(x + 0.5, 0, width, -1, 1) * aspect
            py = remap(y + 0.5, height, 0, -1, 1)
            ray_direction = normalize(
                np.array([px, py, 1], dtype=np.float32) - ray_origin
            )

            t_closest, weights_closest = trace(
                ray_origin, ray_direction, t_min, t_max, meshes
            )

            color = [0, 51, 102]
            if weights_closest is not None:
                color = np.array(255.0 * weights_closest, dtype=np.uint8)

            image[y, x] = color

    Image.fromarray(image).save("out.png")


if __name__ == "__main__":
    main()
