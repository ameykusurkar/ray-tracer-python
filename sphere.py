import numpy as np

INFINITY = float("inf")

class Sphere:
    def __init__(self, center, radius, color):
        self.center = center
        self.radius = radius
        self.color = color

    def ray_intersection(self, ray_p, ray_dir):
        projection = ray_projection(ray_p, ray_dir, self.center)
        projection_dist = np.linalg.norm(projection - self.center, axis=1)

        # if projection_dist > self.radius: implicitly infinity
        intersection = np.full(ray_p.shape, INFINITY)
        intersect_dist = np.full(ray_p.shape[0], INFINITY)

        intersection[projection_dist == self.radius] = projection[projection_dist == self.radius]
        intersect_dist = np.where(projection_dist == self.radius, projection_dist, intersect_dist)

        proj_to_intersect_dist = np.sqrt(
            np.square(self.radius) - np.square(projection_dist)
        )
        intersect_dist = np.where(
            projection_dist < self.radius,
            np.where(
                # If origin is inside the sphere,
                np.linalg.norm(ray_p - self.center, axis=1) < self.radius,
                # the intersection will be in front of the projection
                np.linalg.norm(projection - ray_p, axis=1) + proj_to_intersect_dist,
                # otherwise it will be behind the projection
                np.linalg.norm(projection - ray_p, axis=1) - proj_to_intersect_dist
            ),
            intersect_dist
        )
        intersection[projection_dist < self.radius] = (ray_p + (intersect_dist[..., np.newaxis] * ray_dir))[projection_dist < self.radius]

        normal = normalize(intersection - self.center)
        return intersection, intersect_dist, normal

#     def get_color(self, points):
#         return np.tile(self.color, (points.shape[0], 1))


def ray_projection(ray_p, ray_dir, point):
    u = point - ray_p
    uv_dot = np.multiply(ray_dir, u).sum(axis=1, keepdims=True)
    # Points behind the ray will never project, hence at infinity
    uv_dot[uv_dot <= 0] = INFINITY
    proj_dist = uv_dot / np.linalg.norm(ray_dir, axis=1, keepdims=True)
    return ray_p + proj_dist * ray_dir

# TODO: Move this to a more sensible place
def normalize(v):
    last_dim = v.ndim - 1
    v_mag = np.linalg.norm(v, axis=last_dim, keepdims=True)
    return np.divide(v, v_mag, where=(v_mag > 0))
