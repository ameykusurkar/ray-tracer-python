from PIL import Image
import numpy as np
# import pdb; pdb.set_trace()

WIDTH, HEIGHT = 1024, 768
FOV = (0.5) * np.pi
framebuffer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

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
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def ray_intersect(self, ray):
        l = self.center - ray.p

        projection = ray.projection(self.center)

        if projection is None: return False

        dist = np.linalg.norm(projection - self.center)

        return dist <= self.radius

sphere = Sphere(np.array([-3, 0, -16]), 2)

def cast_ray(ray, sphere):
    if sphere.ray_intersect(ray):
        return [255, 255, 0]
    else:
        return [0, 0, 0]

for i in range(0, len(framebuffer)):
    for j in range(0, len(framebuffer[i])):
        x =  (2 * (i + 0.5)/WIDTH  - 1) * np.tan(FOV/2) * WIDTH / HEIGHT
        y = -(2 * (j + 0.5)/HEIGHT - 1) * np.tan(FOV/2)
        v = np.array([x, y, -1])
        v = v / np.linalg.norm(v)
        ray = Ray(np.array([0, 0, 0]), v)
        framebuffer[i, j] = cast_ray(ray, sphere)

im = Image.fromarray(framebuffer)
im.save("test.png")
