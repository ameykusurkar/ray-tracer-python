from PIL import Image
import numpy as np
from sphere import HittableList, Sphere, INFINITY, normalize
from camera import Camera

WIDTH, HEIGHT = 200, 100

def fill_background(ray_p, ray_dir):
    unit_direction = normalize(ray_dir)
    t = 0.5 * (unit_direction[:, 1] + 1)
    t = t[..., np.newaxis]
    return (1 - t) * [1.0, 1.0, 1.0] + t * [0.5, 0.7, 1.0]

# TODO: Make this non-recursive
def color(ray_p, ray_dir, hittable):
    colors = fill_background(ray_p, ray_dir)

    intersection, intersect_dist, normal = hittable.ray_intersection(ray_p, ray_dir)
    has_hit = intersect_dist < INFINITY

    if not np.any(has_hit):
        return colors

    diffuse_p = intersection[has_hit]
    diffuse_dir = normalize(normal[has_hit] + random_in_unit_sphere(np.count_nonzero(has_hit)))

    colors[has_hit] = 0.5 * color(diffuse_p, diffuse_dir, hittable)
    return colors

def scale_down_pixels(pixels, factor):
    h, w, _ = pixels.shape
    new_h, new_w = int(h / factor), int(w / factor)
    return pixels.reshape(new_h, factor, new_w, factor, 3).mean(axis=(1, 3))

def random_in_unit_sphere(n):
    result = get_random_coords(n)
    while True:
        # TODO (optimisation): we don't need to re-check vectors that are already
        # known from previous iterations to be within the unit sphere
        not_in_unit_sphere = (result * result).sum(axis=1) >= 1
        if not np.any(not_in_unit_sphere):
            break
        result[not_in_unit_sphere] = get_random_coords(np.count_nonzero(not_in_unit_sphere))
    return result

# Random co-ordinates where each element is between -1 and 1
def get_random_coords(n):
    return 2 * np.random.rand(n, 3) - 1

ray_dir = np.zeros((HEIGHT, WIDTH, 3))

hittable_list = HittableList([
    Sphere(np.array([0, 0, -1]), 0.5, [1, 0, 0]),
    Sphere(np.array([0, -100.5, -1]), 100, [1, 0, 0]),
])

ANTI_ALIASING_FACTOR = 4
camera = Camera(HEIGHT * ANTI_ALIASING_FACTOR, WIDTH * ANTI_ALIASING_FACTOR)

ray_p, ray_dir = camera.get_initial_rays()

# Even though the pixels have the dimensions H x W x 3, we can think of
# them as (H x W) number of rays, reducing the number for dimensions
# while calculating.
ray_dir = ray_dir.reshape(-1, 3)
ray_p = ray_p.reshape(-1, 3)

pixels = color(ray_p, normalize(ray_dir), hittable_list)
pixels = pixels.reshape(camera.height, camera.width, 3)
pixels = scale_down_pixels(pixels, ANTI_ALIASING_FACTOR)
pixels = (pixels * 255).astype(np.uint8)

im = Image.fromarray(pixels)
im.save("output.png")
