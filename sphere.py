import numpy as np

INFINITY = float("inf")

class HittableList:
    def __init__(self, hittables):
        self.hittables = hittables

    def ray_intersection(self, ray_p, ray_dir):
        closest_dist         = np.full(ray_p.shape[0], INFINITY)
        closest_intersection = np.full(ray_p.shape,    INFINITY)
        closest_normal       = np.full(ray_p.shape,    INFINITY)
        closest_material     = np.full(ray_p.shape[0], None)
        closest_albedo       = np.full(ray_p.shape,    1.0)

        for hittable in self.hittables:
            intersection, intersect_dist, normal = hittable.ray_intersection(ray_p, ray_dir)
            new_closest = intersect_dist < closest_dist
            closest_dist[new_closest]         = intersect_dist[new_closest]
            closest_intersection[new_closest] = intersection[new_closest]
            closest_normal[new_closest]       = normal[new_closest]
            closest_material[new_closest]     = type(hittable.material)
            closest_albedo[new_closest]       = hittable.material.albedo

        return closest_intersection, closest_dist, closest_normal, closest_material, closest_albedo


class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

    def ray_intersection(self, ray_p, ray_dir):
        p_to_center = self.center - ray_p # L
        proj_dist = np.multiply(p_to_center, ray_dir).sum(axis=1) # tca
        p_to_center_sq = np.multiply(p_to_center, p_to_center).sum(axis=1) # L^2
        proj_to_center_dist_sq = p_to_center_sq - np.square(proj_dist) # d^2

        intersect_dist = np.full(ray_p.shape[0], INFINITY)
        intersection = np.full(ray_p.shape, INFINITY)

        radius_sq = np.square(self.radius)

        # TODO: Account for when ray_p is inside the sphere, in front of self.center
        has_solution = (proj_to_center_dist_sq <= radius_sq) & (proj_dist > 0)

        # FIXME: Invalid values when projection is outside sphere
        proj_to_intersect_dist = np.sqrt(radius_sq - proj_to_center_dist_sq) # thc

        inside_sphere = has_solution & (p_to_center_sq < radius_sq)
        intersect_dist[inside_sphere] = (proj_dist + proj_to_intersect_dist)[inside_sphere]

        outside_sphere = has_solution & (p_to_center_sq >= radius_sq)
        intersect_dist[outside_sphere] = (proj_dist - proj_to_intersect_dist)[outside_sphere]

        intersection[has_solution] = (ray_p + (intersect_dist[..., np.newaxis] * ray_dir))[has_solution]

        # TODO: Check if we always need to compute intersection, normal
        normal = normalize(intersection - self.center)
        return intersection, intersect_dist, normal


# TODO: Move this to a more sensible place
def normalize(v):
    last_dim = v.ndim - 1
    v_mag = np.linalg.norm(v, axis=last_dim, keepdims=True)
    return np.divide(v, v_mag, where=(np.isfinite(v_mag) & (v_mag > 0)))
