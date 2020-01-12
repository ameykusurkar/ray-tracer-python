import numpy as np
from sphere import normalize

# TODO: Vectorise this. This needs to be faster to scale.
def scatter(material_types, ray_p, ray_dir, intersection, normal):
    scatter_p = np.zeros(normal.shape)
    scatter_dir = np.zeros(normal.shape)

    for mat_type in set(material_types):
        curr_material = material_types == mat_type
        sctr_p, sctr_dir = mat_type.scatter(
            ray_p[curr_material],
            ray_dir[curr_material],
            intersection[curr_material],
            normal[curr_material],
        )
        scatter_p[curr_material] = sctr_p
        scatter_dir[curr_material] = sctr_dir

    return scatter_p, scatter_dir

class Lambertian:
    def __init__(self, albedo):
        self.albedo = albedo

    @staticmethod
    def scatter(ray_p, ray_dir, intersection, normal):
        # Add some outward bias, so that the reflected ray does not intersect
        # the object it just reflected off
        scatter_p = intersection + 1e-3 * normal
        scatter_dir = normalize(normal + random_in_unit_sphere(normal.shape[0]))
        return scatter_p, scatter_dir

# TODO: Implement fuzziness in reflections
class Metal:
    def __init__(self, albedo):
        self.albedo = albedo

    @staticmethod
    def scatter(ray_p, ray_dir, intersection, normal):
        # Add some outward bias, so that the reflected ray does not intersect
        # the object it just reflected off
        scatter_p = intersection + 1e-3 * normal
        scatter_dir = reflect(ray_dir, normal)
        return scatter_p, scatter_dir

class Dielectric:
    def __init__(self, albedo):
        # Glass doesn't absorb light
        self.albedo = [1, 1, 1]

    @staticmethod
    def scatter(ray_p, ray_dir, intersection, normal):
        refractive_index = 1.5

        # Negate cosine as normal and ray point in opposite directions
        cosine = -1 * np.multiply(normal, ray_dir).sum(axis=1)
        cosine = np.where(cosine < 0, refractive_index * -cosine, cosine)
        refract_prob = 1 - schlick(cosine, refractive_index)

        should_refract = np.random.rand(ray_p.shape[0]) < refract_prob

        scatter_dir = np.zeros(ray_p.shape)

        scatter_dir[should_refract] = refract(
            ray_dir[should_refract], normal[should_refract], refractive_index
        )

        should_reflect = (~should_refract) | np.isnan(scatter_dir[:, 0])
        scatter_dir[should_reflect] = reflect(
            ray_dir[should_reflect], normal[should_reflect]
        )

        # Add some outward bias, so that the reflected ray does not intersect
        # the object it just reflected off
        scatter_p = np.where(should_reflect.reshape(-1, 1),
                             intersection + 1e-3 * normal,
                             intersection - 1e-3 * normal)

        return scatter_p, normalize(scatter_dir)

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

# TODO: Support refractive_index as an array
def refract(incident, normal, refractive_index):
    cos_i = np.multiply(incident, normal).sum(axis=1, keepdims=True)
    outside_surface = cos_i < 0

    cos_i = np.where(outside_surface, -cos_i, cos_i)
    refract_ratio = np.where(outside_surface, 1 / refractive_index, refractive_index)
    normal = np.where(outside_surface, normal, -normal)

    # cos_theta_2 in Snell's law
    cos_r_sq = 1 - np.square(refract_ratio) * (1 - np.square(cos_i))
    cos_r = np.sqrt(cos_r_sq)
    return np.where(
        cos_r_sq > 0,
        normalize(refract_ratio * incident + (refract_ratio * cos_i - cos_r) * normal),
        float("nan")
    )

def schlick(cosine, refractive_index):
    r0 = ((1 - refractive_index) / (1 + refractive_index)) ** 2
    return r0 + (1 - r0) * ((1 - cosine)**5)
