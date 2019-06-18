from PIL import Image
from collections import namedtuple
import numpy as np
# import pdb; pdb.set_trace()

WIDTH, HEIGHT = 400, 300
FOV = (0.3) * np.pi
framebuffer = np.zeros((HEIGHT, WIDTH, 3))

YELLOW = [255, 255, 0]
BLUE = [0, 0, 255]
RED = [255, 0, 0]
GREEN = [0, 255, 0]
BLACK = [0, 0, 0]
BG_COLOR = [52, 73, 94]

Light = namedtuple("Light", ["pos", "intensity"])
Material = namedtuple("Material", ["refractive_index", "ambient_reflection", "diffuse_reflection", "specular_reflection", "refraction_constant", "specular_exponent"])

IVORY = Material(1.0, 0.1, 0.6, 0.3, 0.0, 50)
RUBBER = Material(1.0, 0.0, 0.9, 0.1, 0.0, 10)
MIRROR = Material(1.0, 0.8, 0.0, 10, 0.0, 1425)
GLASS = Material(1.5, 0.1, 0.0, 0.5, 0.8, 125)

REFLECT_DEPTH = 4

class Ray:
    def __init__(self, p, v):
        self.p = p
        self.v = v

    def projection(self, point):
        u = point - self.p
        uv_dot = np.dot(self.v, u)
        if uv_dot <= 0:
            # Center of the circle does not project onto the ray
            return None
        proj_vector = (uv_dot / np.linalg.norm(ray.v)) * self.v
        return self.p + proj_vector

class Sphere:
    def __init__(self, center, radius, color, material):
        self.center = center
        self.radius = radius
        self.color = color
        self.material = material

    def ray_intersection(self, ray):
        projection = ray.projection(self.center)

        if projection is None: return None

        projection_dist = np.linalg.norm(projection - self.center)

        if projection_dist > self.radius:
            return None
        elif projection_dist == self.radius:
            intersection, intersect_dist = projection, projection_dist
        else:
            # TODO: There might be a performance optimisation here, by
            # avoiding squaring `projection_dist`
            proj_to_intersect_dist = np.sqrt(
                np.square(self.radius) - np.square(projection_dist)
            )
            if np.linalg.norm(ray.p - self.center) < self.radius:
                # If ray origin is inside the sphere, the intersection will be ahead
                intersect_dist = np.linalg.norm(projection - ray.p) + proj_to_intersect_dist
            else:
                intersect_dist = np.linalg.norm(projection - ray.p) - proj_to_intersect_dist
            intersection = ray.p + (intersect_dist * ray.v)

        normal = intersection - self.center
        normal = normal / np.linalg.norm(normal)
        return intersection, intersect_dist, normal

spheres = [
    Sphere(np.array([-5, 4, -20]), 2, YELLOW, RUBBER),
    Sphere(np.array([10, 3, -50]), 4, BLUE, MIRROR),
    Sphere(np.array([-2, 5, -40]), 5, RED, IVORY),
    Sphere(np.array([-1, 0, -10]), 1, GREEN, GLASS),
]

lights = [
    Light(np.array([-2, 10, -20]), 1),
    Light(np.array([-16, 10, 16]), 0.8),
]

def reflect(incident, normal):
    return incident - 2 * normal * np.dot(incident, normal)

def refract(incident, normal, refractive_index):
    cos_i = -np.dot(incident, normal)
    refract_ratio = 1 / refractive_index

    if cos_i < 0:
        cos_i *= -1
        refract_ratio = 1 / refract_ratio

    # cos_theta_2 in Snell's law
    cos_r = np.sqrt(1 - np.square(refract_ratio) * (1 - np.square(cos_i)))
    return refract_ratio * incident + (refract_ratio * cos_i - cos_r) * normal

def scene_intersection(ray, spheres):
    color = None
    closest_dist = float("inf")
    closest_intersection = None
    closest_normal = None
    material = None
    for sphere in spheres:
        result = sphere.ray_intersection(ray)
        if result is None: continue

        intersection, intersect_dist, normal = result

        if intersect_dist < closest_dist:
            closest_dist = intersect_dist
            closest_intersection = intersection
            closest_normal = normal
            color = sphere.color
            material = sphere.material

    if closest_intersection is None: return None

    return closest_intersection, closest_normal, color, material

def cast_ray(ray, spheres, lights, depth=0):
    if depth > REFLECT_DEPTH: return BG_COLOR

    result = scene_intersection(ray, spheres)
    if result is None: return BG_COLOR
    intersection, normal, color, material = result

    offset = 1e-3 * normal
    diffuse_intensity = 0
    specular_intensity = 0
    for light in lights:
        light_dir = light.pos - intersection
        light_dist = np.linalg.norm(light_dir)
        light_dir = light_dir / light_dist

        offset = 1e-3 * normal
        ln_dot = np.dot(light_dir, normal)
        shadow_p = intersection - offset if ln_dot < 0 else intersection + offset
        if scene_intersection(Ray(shadow_p, light_dir), spheres): continue

        diffuse_intensity += light.intensity * max(0, ln_dot)
        rv_dot = max(0, np.dot(-reflect(-light_dir, normal), ray.v))
        specular_intensity += light.intensity * np.power(rv_dot, material.specular_exponent)

    reflect_dir = reflect(ray.v, normal)
    reflect_dir = reflect_dir / np.linalg.norm(reflect_dir)
    reflect_p = intersection - offset if np.dot(reflect_dir, normal) < 0 else intersection + offset
    reflect_color = cast_ray(Ray(reflect_p, reflect_dir), spheres, lights, depth + 1)

    refract_dir = refract(ray.v, normal, material.refractive_index)
    refract_dir = refract_dir / np.linalg.norm(refract_dir)
    refract_p = intersection - offset if np.dot(refract_dir, normal) < 0 else intersection + offset
    refract_color = cast_ray(Ray(refract_p, refract_dir), spheres, lights, depth + 1)

    intensity = diffuse_intensity * np.array(color) * material.diffuse_reflection + \
                specular_intensity * np.array([255, 255, 255]) * material.specular_reflection + \
                np.array(reflect_color) * material.ambient_reflection + \
                np.array(refract_color) * material.refraction_constant

    return intensity

for j in range(0, len(framebuffer)):
    for i in range(0, len(framebuffer[j])):
        x =  (2 * (i + 0.5)/WIDTH  - 1) * np.tan(FOV/2) * WIDTH / HEIGHT
        y = -(2 * (j + 0.5)/HEIGHT - 1) * np.tan(FOV/2)
        v = np.array([x, y, -1])
        v = v / np.linalg.norm(v)
        ray = Ray(np.array([0, 0, 0]), v)
        framebuffer[j, i] = cast_ray(ray, spheres, lights)

for j in range(0, len(framebuffer)):
    for i in range(0, len(framebuffer[j])):
        color = framebuffer[j, i]
        if color.max() > 255:
            color = color / color.max() * 255
            framebuffer[j, i] = color

im = Image.fromarray(framebuffer.astype(np.uint8))
im.save("test.png")
