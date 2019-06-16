from PIL import Image
import numpy as np
# import pdb; pdb.set_trace()

WIDTH, HEIGHT = 400, 300
FOV = (0.5) * np.pi
framebuffer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

YELLOW = [255, 255, 0]
BLUE = [0, 0, 255]
RED = [255, 0, 0]
GREEN = [0, 255, 0]
BLACK = [0, 0, 0]

class Ray:
    def __init__(self, p, v):
        self.p = p
        self.v = v

    def projection(self, point):
        u = point - self.p
        uv_dot = np.dot(self.v, u)
        if uv_dot <= 0:
            # Center of the circle does not project onto the ray
            return None
        proj_vector = (uv_dot / np.linalg.norm(ray.v)) * self.v
        return self.p + proj_vector

class Sphere:
    def __init__(self, center, radius, color):
        self.center = center
        self.radius = radius
        self.color = color

    def ray_intersect_dist(self, ray):
        projection = ray.projection(self.center)

        if projection is None: return None

        projection_dist = np.linalg.norm(projection - self.center)

        if projection_dist > self.radius:
            return None
        elif projection_dist == self.radius:
            return projection_dist
        else:
            # TODO: There might be a performance optimisation here, by
            # avoiding squaring `projection_dist`
            proj_to_intersect_dist = np.sqrt(
                np.square(self.radius) - np.square(projection_dist)
            )
            return np.linalg.norm(projection - ray.p) - proj_to_intersect_dist

spheres = [
    Sphere(np.array([-5, 0, -16]), 2, YELLOW),
    Sphere(np.array([-3, 0, -18]), 4, BLUE),
    Sphere(np.array([-1, -3, -16]), 3, GREEN),
]

def cast_ray(ray, spheres):
    color = BLACK
    closest_dist = float("inf")
    for sphere in spheres:
        intersect_dist = sphere.ray_intersect_dist(ray)
        if intersect_dist is None: continue

        if intersect_dist < closest_dist:
            closest_dist = intersect_dist
            color = sphere.color

    return color

for j in range(0, len(framebuffer)):
    for i in range(0, len(framebuffer[j])):
        x =  (2 * (i + 0.5)/WIDTH  - 1) * np.tan(FOV/2) * WIDTH / HEIGHT
        y = -(2 * (j + 0.5)/HEIGHT - 1) * np.tan(FOV/2)
        v = np.array([x, y, -1])
        v = v / np.linalg.norm(v)
        ray = Ray(np.array([0, 0, 0]), v)
        framebuffer[j, i] = cast_ray(ray, spheres)

im = Image.fromarray(framebuffer)
im.save("test.png")
