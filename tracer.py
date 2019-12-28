from PIL import Image
import numpy as np
from sphere import Sphere, INFINITY, normalize

WIDTH, HEIGHT = 200, 100

def compute_background(ray_p, ray_dir):
    unit_direction = normalize(ray_dir)
    t = 0.5 * (unit_direction[:, 1] + 1)
    t = t[..., np.newaxis]
    return (1 - t) * [1.0, 1.0, 1.0] + t * [0.5, 0.7, 1.0]

def color(ray_p, ray_dir, sphere):
    background = compute_background(ray_p, ray_dir)
    intersection, _, normal = sphere.ray_intersection(ray_p, ray_dir)
    return np.where(intersection < INFINITY, (normal + 1) * 0.5, background)

ray_dir = np.zeros((HEIGHT, WIDTH, 3))

ASPECT_RATIO = WIDTH / HEIGHT
ray_dir[:, :, 0] = np.linspace(-ASPECT_RATIO, ASPECT_RATIO, WIDTH)
# We want the the y-coord to increase upwards
ray_dir[:, :, 1] = np.linspace(1, -1, HEIGHT).reshape(-1, 1)
ray_dir[:, :, 2] = -1

# Even though the pixels have the dimensions H x W x 3, we can think of
# them as (H x W) number of rays, reducing the number for dimensions
# while calculating.
ray_dir = ray_dir.reshape(-1, 3)
ray_p = np.zeros((HEIGHT * WIDTH, 3))

sphere = Sphere(np.array([0, 0, -1]), 0.5, [1, 0, 0])

pixels = color(ray_p, normalize(ray_dir), sphere) * 255
pixels = pixels.reshape(HEIGHT, WIDTH, 3).astype(np.uint8)

im = Image.fromarray(pixels)
im.save("output.png")
