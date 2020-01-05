import numpy as np
from sphere import normalize

# TODO: Vectorise this. This needs to be faster to scale.
def scatter(material, ray_p, ray_dir, intersection, normal):
    scatter_p = np.zeros(normal.shape)
    scatter_dir = np.zeros(normal.shape)
    albedo = np.zeros(normal.shape)

    for mat in set(material):
        curr_material = material == mat
        sctr_p, sctr_dir, alb = mat.scatter(
            ray_p[curr_material],
            ray_dir[curr_material],
            intersection[curr_material],
            normal[curr_material],
        )
        scatter_p[curr_material] = sctr_p
        scatter_dir[curr_material] = sctr_dir
        albedo[curr_material] = alb
    return scatter_p, scatter_dir, albedo

class Lambertian:
    def __init__(self, albedo):
        self.albedo = albedo

    def scatter(self, ray_p, ray_dir, intersection, normal):
        # Add some outward bias, so that the reflected ray does not intersect
        # the object it just reflected off
        scatter_p = intersection + 1e-3 * normal
        scatter_dir = normalize(normal + random_in_unit_sphere(normal.shape[0]))
        return scatter_p, scatter_dir, self.albedo

class Metal:
    def __init__(self, albedo):
        self.albedo = albedo

    def scatter(self, ray_p, ray_dir, intersection, normal):
        # Add some outward bias, so that the reflected ray does not intersect
        # the object it just reflected off
        scatter_p = intersection + 1e-3 * normal
        scatter_dir = reflect(ray_dir, normal)
        return scatter_p, scatter_dir, self.albedo

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

def reflect(incident, normal):
    in_dot = np.multiply(incident, normal).sum(axis=1, keepdims=True)
    return incident - 2 * in_dot * normal
