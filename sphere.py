import numpy as np

INFINITY = float("inf")

class HittableList:
    def __init__(self, hittables):
        self.hittables = hittables

    def ray_intersection(self, ray_p, ray_dir):
        closest_dist         = np.full(ray_p.shape[0], INFINITY)
        closest_intersection = np.full(ray_p.shape,    INFINITY)
        closest_normal       = np.full(ray_p.shape,    INFINITY)

        for hittable in self.hittables:
            intersection, intersect_dist, normal = hittable.ray_intersection(ray_p, ray_dir)
            new_closest = intersect_dist < closest_dist
            closest_dist[new_closest]         = intersect_dist[new_closest]
            closest_intersection[new_closest] = intersection[new_closest]
            closest_normal[new_closest]       = normal[new_closest]

        return closest_intersection, closest_dist, closest_normal


class Sphere:
    def __init__(self, center, radius, color):
        self.center = center
        self.radius = radius
        self.color = color

    def ray_intersection(self, ray_p, ray_dir):
        projection = ray_projection(ray_p, ray_dir, self.center)
        projection_dist = np.linalg.norm(projection - self.center, axis=1)

        # if projection_dist > self.radius, there is no intersection
        intersection = np.full(ray_p.shape, INFINITY)
        intersect_dist = np.full(ray_p.shape[0], INFINITY)

        # if projection_dist == self.radius, there is one intersection
        intersection[projection_dist == self.radius] = projection[projection_dist == self.radius]
        intersect_dist = np.where(projection_dist == self.radius, projection_dist, intersect_dist)

        #FIXME: Invalid projection_dist values where there is no projection
        proj_to_intersect_dist = np.sqrt(
            np.square(self.radius) - np.square(projection_dist)
        )

        # if projection_dist < self.radius, the projection is inside the sphere,
        # meaning that there are two intersections. We want the nearer one.
        intersect_dist = np.where(
            projection_dist < self.radius,
            np.where(
                # If the ray origin is inside the sphere,
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


def ray_projection(ray_p, ray_dir, point):
    u = point - ray_p
    uv_dot = np.multiply(ray_dir, u).sum(axis=1, keepdims=True)
    # Points behind the ray will never project, hence at infinity
    uv_dot[uv_dot <= 0] = INFINITY
    dist_from_ray_origin = uv_dot / np.linalg.norm(ray_dir, axis=1, keepdims=True)
    return ray_p + dist_from_ray_origin * ray_dir

# TODO: Move this to a more sensible place
def normalize(v):
    last_dim = v.ndim - 1
    v_mag = np.linalg.norm(v, axis=last_dim, keepdims=True)
    return np.divide(v, v_mag, where=(np.isfinite(v_mag) & (v_mag > 0)))
