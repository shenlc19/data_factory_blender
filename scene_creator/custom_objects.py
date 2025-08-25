import blenderproc as bproc
import argparse
import os
import numpy as np
import random
import json
import mathutils
import bpy
import random
from typing import *
import bpy
import glob
from mathutils import Vector, Matrix, Euler, Quaternion
import math
from blenderproc.python.types.MeshObjectUtility import MeshObject

def scene_bbox(scene_meshes) -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def normalize_scene(obj, normalize_factor=1) -> Tuple[float, Vector]:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        Tuple[float, Vector]: The scale factor and the offset applied to the scene.
    """

    bbox_min, bbox_max = scene_bbox([obj])
    scale = normalize_factor / max(bbox_max - bbox_min)
    obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox([obj])
    offset = -(bbox_min + bbox_max) / 2
    obj.matrix_world.translation += offset
    
    return scale, offset

parser = argparse.ArgumentParser()

parser.add_argument('objects', nargs='?', help="Path to a directory of objects")
parser.add_argument('output_dir')
args = parser.parse_args()

# Clear the current scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Function to append all objects from a .blend file
def append_all_objects(blend_path):
    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        # Only append objects, you can extend to meshes, materials, etc.
        data_to.objects = data_from.objects

    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)

bproc.init()

num_sel = 6
all_objs = glob.glob(os.path.join(args.objects, '*'))
blend_files = random.sample(all_objs, k=6)

# Loop through and append objects from all files
for blend_file in blend_files:
    append_all_objects(blend_file)

objects_to_sample = []
for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.name != 'PLANE':
        obj.select_set(True)
        try:
            normalize_factor = random.uniform(0.08, 0.23)
            normalize_scene(obj, normalize_factor=normalize_factor)
            objects_to_sample.append(MeshObject(obj))
        except:
            pass
        obj.select_set(False)

# create room
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),]
            #    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
            #    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
            #    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
            #    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')
light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                   emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))    
light_plane.replace_materials(light_plane_material)

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(200)
light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                        elevation_min = 5, elevation_max = 89, uniform_volume = False)
light_point.set_location(location)

CC_TEXTURE_PATH = '/DATA_EDS2/shenlc2403/data_factory/BlenderProc/resources/cctextures'
# sample CC Texture and assign to room planes
cc_textures = bproc.loader.load_ccmaterials(CC_TEXTURE_PATH)
random_cc_texture = np.random.choice(cc_textures)
for plane in room_planes:
    plane.replace_materials(random_cc_texture)

# Define a function that samples the initial pose of a given object above the ground
def sample_initial_pose(obj: bproc.types.MeshObject):
    obj.set_location(bproc.sampler.upper_region(objects_to_sample_on=room_planes[0:1],
                                                min_height=1, max_height=4, face_sample_range=[0.4, 0.6]))
    obj.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, np.pi * 2]))


# Sample objects on the given surface
placed_objects = bproc.object.sample_poses_on_surface(objects_to_sample=objects_to_sample,
                                         surface=room_planes[0],
                                         sample_pose_func=sample_initial_pose,
                                         min_distance=0.01,
                                         max_distance=0.2)

poi = bproc.object.compute_poi(placed_objects)

# BVH tree used for camera obstacle checks
# bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(placed_objects)

num_cams = 120
# Add translational random walk on top of the POI
poi_drift = bproc.sampler.random_walk(total_length = num_cams, dims = 3, step_magnitude = 0.005, 
                                      window_size = 5, interval = [-0.03, 0.03], distribution = 'uniform')

# Rotational camera shaking as a random walk: Sample an axis angle representation
camera_shaking_rot_angle = bproc.sampler.random_walk(total_length = num_cams, dims = 1, step_magnitude = np.pi/32, window_size = 5,
                                                     interval = [-np.pi/6, np.pi/6], distribution = 'uniform', order = 2)
camera_shaking_rot_axis = bproc.sampler.random_walk(total_length = num_cams, dims = 3, window_size = 10, distribution = 'normal')
camera_shaking_rot_axis /= np.linalg.norm(camera_shaking_rot_axis, axis=1, keepdims=True)

poses = 0
radius_min = 0.61
radius_max = 1.24
radius = random.uniform(radius_min, radius_max)
camera_poses = []

random_curve_length = random.uniform(0.75, 1.0)
for i in range(num_cams):
    location_cam = np.array([radius*np.cos(i/num_cams * random_curve_length * np.pi), radius*np.sin(i/num_cams * random_curve_length * np.pi), radius])
    # Compute rotation based on vector going from location towards poi + drift
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi + poi_drift[i] - location_cam)
    # random walk axis-angle -> rotation matrix
    R_rand = np.array(mathutils.Matrix.Rotation(camera_shaking_rot_angle[i], 3, camera_shaking_rot_axis[i]))
    # Add the random walk to the camera rotation 
    rotation_matrix = R_rand @ rotation_matrix
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location_cam, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)
    camera_poses.append(cam2world_matrix.tolist())

# activate depth rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

# # render the whole pipeline
# data = bproc.renderer.render()

# # Write data in bop format
# bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data'),
#                        dataset = args.bop_dataset_name,
#                        depths = data["depth"],
#                        colors = data["colors"], 
#                        color_file_format = "JPEG",
#                        ignore_dist_thres = 10)

# We do not render here, but save the scene.

os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(args.output_dir, 'transforms.json'), 'w') as f:
    json.dump(camera_poses, f)
save_scene_dir = os.path.join(args.output_dir, 'lm.blend')
bpy.ops.wm.save_as_mainfile(filepath=save_scene_dir)