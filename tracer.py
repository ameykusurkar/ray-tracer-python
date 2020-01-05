from PIL import Image
import numpy as np
import time

from sphere import HittableList, Sphere, INFINITY, normalize
from camera import Camera
from material import Lambertian, Metal, scatter

WIDTH, HEIGHT = 200, 100

def fill_background(ray_p, ray_dir):
    unit_direction = normalize(ray_dir)
    t = 0.5 * (unit_direction[:, 1] + 1)
    t = t[..., np.newaxis]
    return (1 - t) * [1.0, 1.0, 1.0] + t * [0.5, 0.7, 1.0]

# TODO: Make this non-recursive
def color(ray_p, ray_dir, hittable):
    colors = fill_background(ray_p, ray_dir)

    intersection, intersect_dist, normal, material, attenuation = hittable.ray_intersection(ray_p, ray_dir)
    has_hit = intersect_dist < INFINITY

    if not np.any(has_hit):
        return colors

    scatter_p, scatter_dir = scatter(
        material[has_hit],
        ray_p[has_hit], ray_dir[has_hit],
        intersection[has_hit], normal[has_hit],
    )

    colors[has_hit] = attenuation[has_hit] * color(scatter_p, scatter_dir, hittable)
    return colors

def scale_down_pixels(pixels, factor):
    h, w, _ = pixels.shape
    new_h, new_w = int(h / factor), int(w / factor)
    return pixels.reshape(new_h, factor, new_w, factor, 3).mean(axis=(1, 3))

hittable_list = HittableList([
    Sphere(np.array([0, 0, -1]), 0.5, Lambertian([0.8, 0.3, 0.3])),
    Sphere(np.array([-1, 0, -1]), 0.5, Metal([0.8, 0.8, 0.8])),
    Sphere(np.array([1, 0, -1]), 0.5, Metal([0.8, 0.6, 0.2])),
    Sphere(np.array([0, -100.5, -1]), 100, Lambertian([0.8, 0.8, 0.0])),
])

ANTI_ALIASING_FACTOR = 4
camera = Camera(HEIGHT * ANTI_ALIASING_FACTOR, WIDTH * ANTI_ALIASING_FACTOR)

ray_p, ray_dir = camera.get_initial_rays()

# Even though the pixels have the dimensions H x W x 3, we can think of
# them as (H x W) number of rays, reducing the number for dimensions
# while calculating.
ray_dir = ray_dir.reshape(-1, 3)
ray_p = ray_p.reshape(-1, 3)

start = time.time()

pixels = color(ray_p, normalize(ray_dir), hittable_list)

print(f"Time taken: {time.time()-start:0.2f} seconds, Resolution: {WIDTH} x {HEIGHT}")

pixels = pixels.reshape(camera.height, camera.width, 3)
pixels = scale_down_pixels(pixels, ANTI_ALIASING_FACTOR)
pixels = (pixels * 255).astype(np.uint8)

im = Image.fromarray(pixels)
im.save("output.png")
