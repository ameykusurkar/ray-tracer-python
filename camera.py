import numpy as np

class Camera:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def get_initial_rays(self):
        ray_dir = np.zeros((self.height, self.width, 3))
        ray_p = np.zeros((self.height, self.width, 3))

        aspect_ratio = self.width / self.height
        ray_dir[:, :, 0] = np.linspace(-aspect_ratio, aspect_ratio, self.width)
        # We want the the y-coord to increase upwards
        ray_dir[:, :, 1] = np.linspace(1, -1, self.height).reshape(-1, 1)
        ray_dir[:, :, 2] = -1

        return ray_p, ray_dir
