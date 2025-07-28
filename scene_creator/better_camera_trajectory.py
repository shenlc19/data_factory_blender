import blenderproc as bproc
import argparse
import os
import numpy as np
import random
import json
import mathutils
import bpy

parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path', nargs='?', help="Path to the bop datasets parent directory")
parser.add_argument('bop_dataset_name', nargs='?', help="Main BOP dataset")
parser.add_argument('cc_textures_path', nargs='?', default="resources/cctextures", help="Path to downloaded cc textures")
parser.add_argument('output_dir', nargs='?', help="Path to where the final files will be saved ")
args = parser.parse_args()

bproc.init()

# load a random sample of bop objects into the scene
sampled_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, args.bop_dataset_name),
                                  mm2m = True,
                                  sample_objects = True,
                                  num_of_objs_to_sample = 10)

# load distractor bop objects
distractor_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'tless'),
                                     model_type = 'cad',
                                     mm2m = True,
                                     sample_objects = True,
                                     num_of_objs_to_sample = 3)
distractor_bop_objs += bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'lm'),
                                      mm2m = True,
                                      sample_objects = True,
                                      num_of_objs_to_sample = 3)

# load BOP datset intrinsics
bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(args.bop_parent_path, args.bop_dataset_name))

# set shading and physics properties and randomize PBR materials
for j, obj in enumerate(sampled_bop_objs + distractor_bop_objs):
    obj.set_shading_mode('auto')
        
    mat = obj.get_materials()[0]
    if obj.get_cp("bop_dataset_name") in ['itodd', 'tless']:
        grey_col = np.random.uniform(0.3, 0.9)   
        mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])        
    mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
    mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0, 1.0))
        
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

# sample CC Texture and assign to room planes
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)
random_cc_texture = np.random.choice(cc_textures)
for plane in room_planes:
    plane.replace_materials(random_cc_texture)

# Define a function that samples the initial pose of a given object above the ground
def sample_initial_pose(obj: bproc.types.MeshObject):
    obj.set_location(bproc.sampler.upper_region(objects_to_sample_on=room_planes[0:1],
                                                min_height=1, max_height=4, face_sample_range=[0.4, 0.6]))
    obj.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, np.pi * 2]))

# Sample objects on the given surface
placed_objects = bproc.object.sample_poses_on_surface(objects_to_sample=sampled_bop_objs + distractor_bop_objs,
                                         surface=room_planes[0],
                                         sample_pose_func=sample_initial_pose,
                                         min_distance=0.01,
                                         max_distance=0.2)

poi = bproc.object.compute_poi(placed_objects)

# BVH tree used for camera obstacle checks
# bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(placed_objects)

# Add translational random walk on top of the POI
poi_drift = bproc.sampler.random_walk(total_length = 25, dims = 3, step_magnitude = 0.005, 
                                      window_size = 5, interval = [-0.03, 0.03], distribution = 'uniform')

# Rotational camera shaking as a random walk: Sample an axis angle representation
camera_shaking_rot_angle = bproc.sampler.random_walk(total_length = 25, dims = 1, step_magnitude = np.pi/32, window_size = 5,
                                                     interval = [-np.pi/6, np.pi/6], distribution = 'uniform', order = 2)
camera_shaking_rot_axis = bproc.sampler.random_walk(total_length = 25, dims = 3, window_size = 10, distribution = 'normal')
camera_shaking_rot_axis /= np.linalg.norm(camera_shaking_rot_axis, axis=1, keepdims=True)

poses = 0
num_cams = 25
radius_min = 0.61
radius_max = 1.24
radius = random.uniform(radius_min, radius_max)
camera_poses = []
for i in range(num_cams):
    location_cam = np.array([radius*np.cos(i/25 * np.pi), radius*np.sin(i/25 * np.pi), radius])
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