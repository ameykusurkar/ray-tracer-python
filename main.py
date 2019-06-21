from PIL import Image
from collections import namedtuple
import time
import numpy as np
# import pdb; pdb.set_trace()

WIDTH, HEIGHT = 400, 300
FOV = (0.25) * np.pi
INFINITY = float("inf")

YELLOW = [255, 255, 0]
BLUE = [0, 0, 255]
RED = [255, 0, 0]
GREEN = [0, 255, 0]
BLACK = [0, 0, 0]
WHITE = [255, 255, 255]
BG_COLOR = [52, 73, 94]

Light = namedtuple("Light", ["pos", "intensity"])
class Material:
    def __init__(self, refractive_index, ambient_reflection, diffuse_reflection, specular_reflection, refraction_constant, specular_exponent):
        self.refractive_index = refractive_index
        self.ambient_reflection = ambient_reflection
        self.diffuse_reflection = diffuse_reflection
        self.specular_reflection = specular_reflection
        self.refraction_constant = refraction_constant
        self.specular_exponent = specular_exponent

np_attr = np.vectorize(getattr)

IVORY = Material(1.0, 0.1, 0.6, 2, 0.0, 200)
RUBBER = Material(1.0, 0.0, 0.9, 0.1, 0.0, 10)
MIRROR = Material(1.0, 0.8, 0.0, 10, 0.0, 1425)
GLASS = Material(1.5, 0.1, 0.0, 0.5, 0.8, 125)
DEFAULT_MATERIAL = Material(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

REFLECT_DEPTH = 4

# ray_dir: N, 3
# ray_p: N, 3
# point: 3
# return: N, 3 (projection)
# Points behind the ray will project at INFINITY
def ray_projection(ray_p, ray_dir, point):
    u = point - ray_p
    uv_dot = np.multiply(ray_dir, u).sum(axis=1, keepdims=True)
    # Points behind the ray will never project, hence at infinity
    uv_dot[uv_dot <= 0] = INFINITY
    proj_dist = uv_dot / np.linalg.norm(ray_dir, axis=1, keepdims=True)
    return ray_p + proj_dist * ray_dir

def normalize(v):
    v_mag = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, v_mag, where=(v_mag > 0))

class Ray:
    def __init__(self, p, v):
        self.p = p
        self.v = v

    def projection(self, point):
        u = point - self.p
        uv_dot = np.dot(self.v, u)
        if uv_dot <= 0:
            # Point does not project onto the ray
            return None
        proj_vector = (uv_dot / np.linalg.norm(ray.v)) * self.v
        return self.p + proj_vector

class Plane:
    def __init__(self, p0, normal, color, material):
        self.p0 = p0
        self.normal = normal
        self.color = color
        self.material = material

    # intersection: N, 3
    # intersect_dist: N
    # normal: N, 3
    def ray_intersection(self, ray_p, ray_dir):
        vn_dot = np.dot(ray_dir, self.normal)
        # Points parallel to the ray will never project, hence at infinity
        vn_dot[vn_dot == 0] = INFINITY
        intersect_dist = np.dot(self.p0 - ray_p, self.normal) / vn_dot
        # Points behind the ray will never project, hence at infinity
        intersect_dist[intersect_dist < 0] = INFINITY
        intersection = ray_p + (intersect_dist[..., np.newaxis] * ray_dir)
        return intersection, intersect_dist, np.tile(self.normal, (ray_p.shape[0], 1))

    def get_color(self, points):
        return np.tile(self.color, (points.shape[0], 1))

class CheckerBoard(Plane):
    def get_color(self, points):
        # TODO: This only works for planes of form y = c
        x_even = np.floor((points[:, 0] - self.p0[0]) / 2) % 2 == 0
        z_even = np.floor((points[:, 2] - self.p0[2]) / 2) % 2 == 0
        return np.where((x_even ^ z_even)[..., np.newaxis], BLACK, WHITE)

class Sphere:
    def __init__(self, center, radius, color, material):
        self.center = center
        self.radius = radius
        self.color = color
        self.material = material

    # intersection: N, 3
    # intersect_dist: N
    # normal: N, 3
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

    def get_color(self, points):
        return np.tile(self.color, (points.shape[0], 1))

objects = [
    Sphere(np.array([-6, 1, -40]), 5, RED, IVORY),
    Sphere(np.array([-5, -2, -20]), 2, YELLOW, RUBBER),
    Sphere(np.array([10, 0, -30]), 4, BLUE, MIRROR),
    Sphere(np.array([1, -1, -30]), 3, GREEN, GLASS),
    CheckerBoard(np.array([0, -4, 0]), np.array([0, 1, 0]), GREEN, IVORY),
]

lights = [
    Light(np.array([-2, 10, -20]), 1),
    Light(np.array([-16, 10, 16]), 0.8),
]

def reflect(incident, normal):
    return incident - 2 * normal * np.multiply(incident, normal).sum(axis=1, keepdims=True)

def refract(incident, normal, refractive_index):
    cos_i = -np.multiply(incident, normal).sum(axis=1, keepdims=True)
    refractive_index = refractive_index[..., np.newaxis]

    cos_i = np.where(cos_i < 0, -cos_i, cos_i)
    refract_ratio = np.where(cos_i < 0, refractive_index, 1 / refractive_index)

    # cos_theta_2 in Snell's law
    cos_r = np.sqrt(1 - np.square(refract_ratio) * (1 - np.square(cos_i)))
    return refract_ratio * incident + (refract_ratio * cos_i - cos_r) * normal


# ray_dir: N, 3
# ray_p: N, 3
# OUT:
# closest_intersection: N, 3
# closest_normal: N, 3
# color: N, 3
# material: N
def scene_intersection(ray_p, ray_dir, objects):
    color                = np.full(ray_p.shape,    BG_COLOR)
    closest_dist         = np.full(ray_p.shape[0], INFINITY)
    closest_intersection = np.full(ray_p.shape,    INFINITY)
    closest_normal       = np.full(ray_p.shape,    INFINITY)
    material             = np.full(ray_p.shape[0], DEFAULT_MATERIAL)

    for obj in objects:
        intersection, intersect_dist, normal = obj.ray_intersection(ray_p, ray_dir)
        new_closest = intersect_dist < closest_dist

        closest_dist[new_closest]         = intersect_dist[new_closest]
        closest_intersection[new_closest] = intersection[new_closest]
        closest_normal[new_closest]       = normal[new_closest]
        color[new_closest]                = obj.get_color(intersection)[new_closest]
        material[new_closest]             = obj.material

    return closest_intersection, closest_normal, color, material

# ray_dir: N, 3
# ray_p: N, 3
# return: N, 3 (the 3 RGB channels)
def cast_ray(ray_p, ray_dir, objects, lights, depth=0):
    if depth > REFLECT_DEPTH:
        return np.tile(BG_COLOR, (ray_p.shape[0], 1))

    intersection, normal, color, material = scene_intersection(ray_p, ray_dir, objects)

    offset = 1e-3 * normal
    diffuse_intensity = np.zeros(ray_p.shape[0])
    specular_intensity = np.zeros(ray_p.shape[0])
    for light in lights:
        light_dir = normalize(light.pos - intersection)

        ln_dot = np.multiply(light_dir, normal).sum(axis=1)
        ln_dot[np.isnan(ln_dot)] = 0

        shadow_p = np.where(ln_dot[..., np.newaxis] < 0, intersection - offset, intersection + offset)
        shadow_intersection, _, _, _ = scene_intersection(shadow_p, light_dir, objects)
        no_shadow = np.all(np.isinf(shadow_intersection), axis=1)

        diffuse_intensity += np.where(no_shadow, light.intensity * np.maximum(0, ln_dot), 0)
        rv_dot = np.maximum(0, np.multiply(-reflect(-light_dir, normal), ray_dir).sum(axis=1))
        specular_intensity += np.where(no_shadow, light.intensity * np.power(rv_dot, np_attr(material, "specular_exponent")), 0)

    reflect_dir = normalize(reflect(ray_dir, normal))
    reflect_p = np.where(np.multiply(reflect_dir, normal).sum(axis=1, keepdims=True) < 0, intersection - offset, intersection + offset)
    reflect_color = cast_ray(reflect_p, reflect_dir, objects, lights, depth + 1)

    refract_dir = normalize(refract(ray_dir, normal, np_attr(material, "refractive_index")))
    refract_p = np.where(np.multiply(refract_dir, normal).sum(axis=1, keepdims=True) < 0, intersection - offset, intersection + offset)
    refract_color = cast_ray(refract_p, refract_dir, objects, lights, depth + 1)

    # TODO: Find a nicer way to get the diffuse reflection
    intensity = (diffuse_intensity * np_attr(material, "diffuse_reflection"))[..., np.newaxis] * color + \
                (specular_intensity * np_attr(material, "specular_reflection"))[..., np.newaxis] * np.array(WHITE) + \
                np_attr(material, "ambient_reflection")[..., np.newaxis] * np.array(reflect_color) + \
                np_attr(material, "refraction_constant")[..., np.newaxis] * np.array(refract_color)

    return np.where(~np.isinf(intersection), intensity, BG_COLOR)

start = time.time()

ray_dir = np.zeros((HEIGHT, WIDTH, 3))
i = np.arange(WIDTH).reshape(1, WIDTH)
j = np.arange(HEIGHT).reshape(HEIGHT, 1)
ray_dir[:, :, 0] =  (2 * (i + 0.5)/WIDTH  - 1) * np.tan(FOV/2) * WIDTH / HEIGHT
ray_dir[:, :, 1] = -(2 * (j + 0.5)/HEIGHT - 1) * np.tan(FOV/2)
ray_dir[:, :, 2] = -1

ray_dir = normalize(ray_dir.reshape(-1, 3))
ray_p = np.zeros((HEIGHT * WIDTH, 3))

framebuffer = cast_ray(ray_p, ray_dir, objects, lights)
framebuffer = framebuffer.reshape(HEIGHT, WIDTH, 3)
rgb_max = framebuffer.max(axis=2, keepdims=True)
framebuffer = np.where(rgb_max > 255, np.divide(framebuffer, rgb_max, where=(rgb_max != 0)) * 255, framebuffer)

print(f"Time taken: {time.time()-start:0.2f} seconds")
im = Image.fromarray(framebuffer.astype(np.uint8))
im.save("test.png")
