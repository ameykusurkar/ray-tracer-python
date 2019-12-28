from PIL import Image
import numpy as np

WIDTH, HEIGHT = 200, 100

def normalize(v):
    last_dim = v.ndim - 1
    v_mag = np.linalg.norm(v, axis=last_dim, keepdims=True)
    return np.divide(v, v_mag, where=(v_mag > 0))

def color(ray_p, ray_dir):
    unit_direction = normalize(ray_dir)
    t = 0.5 * (unit_direction[:, 1] + 1)
    t = t[..., np.newaxis]
    return (1 - t) * [1.0, 1.0, 1.0] + t * [0.5, 0.7, 1.0]

ray_dir = np.zeros((HEIGHT, WIDTH, 3))

ray_dir[:, :, 0] = np.linspace(0, 1, WIDTH)
# We want the the y-coord to increase upwards
ray_dir[:, :, 1] = np.linspace(1, 0, HEIGHT).reshape(-1, 1)
ray_dir[:, :, 2] = -1

# Even though the pixels have the dimensions H x W x 3, we can think of
# them as (H x W) number of rays, reducing the number for dimensions
# while calculating.
ray_dir = ray_dir.reshape(-1, 3)
ray_p = np.zeros((HEIGHT * WIDTH, 3))

pixels = color(ray_p, ray_dir) * 255
pixels = pixels.reshape(HEIGHT, WIDTH, 3).astype(np.uint8)

im = Image.fromarray(pixels)
im.save("output.png")
