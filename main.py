from PIL import Image
import numpy as np
# import pdb; pdb.set_trace()

WIDTH, HEIGHT = 400, 300
FOV = (0.3) * np.pi
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

class Light:
    def __init__(self, pos, intensity):
        self.pos = pos
        self.intensity = intensity

class Sphere:
    def __init__(self, center, radius, color):
        self.center = center
        self.radius = radius
        self.color = color

    def ray_intersection(self, ray):
        projection = ray.projection(self.center)

        if projection is None: return None

        projection_dist = np.linalg.norm(projection - self.center)

        if projection_dist > self.radius:
            return None
        elif projection_dist == self.radius:
            intersection, intersect_dist = projection, projection_dist
        else:
            # TODO: There might be a performance optimisation here, by
            # avoiding squaring `projection_dist`
            proj_to_intersect_dist = np.sqrt(
                np.square(self.radius) - np.square(projection_dist)
            )
            intersect_dist = np.linalg.norm(projection - ray.p) - proj_to_intersect_dist
            intersection = ray.p + (intersect_dist * ray.v)

        normal = intersection - self.center
        normal = normal / np.linalg.norm(normal)
        return intersection, intersect_dist, normal

spheres = [
    Sphere(np.array([-5, 0, -16]), 2, YELLOW),
    Sphere(np.array([10, 3, -34]), 4, BLUE),
    Sphere(np.array([-2, 5, -40]), 5, RED),
    Sphere(np.array([1, -3, -16]), 3, GREEN),
]

light = Light(np.array([10, 10, -2]), 1)

def cast_ray(ray, spheres, light):
    color = BLACK
    closest_dist = float("inf")
    closest_intersection = None
    closest_normal = None
    for sphere in spheres:
        result = sphere.ray_intersection(ray)
        if result is None: continue

        intersection, intersect_dist, normal = result

        if intersect_dist < closest_dist:
            closest_dist = intersect_dist
            closest_intersection = intersection
            closest_normal = normal
            color = sphere.color

    if closest_intersection is None: return color

    light_dir = light.pos - closest_intersection
    light_dir = light_dir / np.linalg.norm(light_dir)
    intensity = light.intensity * max(0, np.dot(light_dir, closest_normal))

    return np.minimum(intensity * np.array(color), 255)

for j in range(0, len(framebuffer)):
    for i in range(0, len(framebuffer[j])):
        x =  (2 * (i + 0.5)/WIDTH  - 1) * np.tan(FOV/2) * WIDTH / HEIGHT
        y = -(2 * (j + 0.5)/HEIGHT - 1) * np.tan(FOV/2)
        v = np.array([x, y, -1])
        v = v / np.linalg.norm(v)
        ray = Ray(np.array([0, 0, 0]), v)
        framebuffer[j, i] = cast_ray(ray, spheres, light)

im = Image.fromarray(framebuffer)
im.save("test.png")
