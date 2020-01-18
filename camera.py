import numpy as np

class Camera:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.vfov = np.pi

    def get_initial_rays(self):
        aspect_ratio = self.width / self.height
        half_height = np.tan(self.vfov/2)
        half_width = aspect_ratio * half_height

        ray_dir = np.zeros((self.height, self.width, 3))
        ray_p = np.zeros((self.height, self.width, 3))

        ray_dir[:, :, 0] = np.linspace(-half_width, half_width, self.width)
        # We want the the y-coord to increase upwards
        ray_dir[:, :, 1] = np.linspace(half_height, -half_height, self.height).reshape(-1, 1)
        ray_dir[:, :, 2] = -1

        return ray_p, ray_dir
