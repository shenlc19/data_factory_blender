import argparse, sys, os, math, re, glob
from typing import *
import bpy
from mathutils import Vector, Matrix
import numpy as np
import json
import glob

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import *


def init_lighting():
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    
    # Create key light
    default_light = bpy.data.objects.new("Default_Light", bpy.data.lights.new("Default_Light", type="POINT"))
    bpy.context.collection.objects.link(default_light)
    default_light.data.energy = 1000
    default_light.location = (4, 1, 6)
    default_light.rotation_euler = (0, 0, 0)
    
    # create top light
    top_light = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
    bpy.context.collection.objects.link(top_light)
    top_light.data.energy = 2000
    top_light.location = (0, 0, 10)
    top_light.scale = (100, 100, 100)
    
    # create bottom light
    bottom_light = bpy.data.objects.new("Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA"))
    bpy.context.collection.objects.link(bottom_light)
    bottom_light.data.energy = 1000
    bottom_light.location = (0, 0, -10)
    bottom_light.rotation_euler = (0, 0, 0)
    
    return {
        "default_light": default_light,
        "top_light": top_light,
        "bottom_light": bottom_light
    }

def main(arg):
    os.makedirs(arg.output_folder, exist_ok=True)

    if arg.object.endswith(".blend"):
        delete_invisible_objects()
    else:
        init_scene()
        load_object(arg.object)
        if arg.split_normal:
            split_mesh_normal()
        # delete_custom_normals()

    delete_animation_data() # lihong add 20250126
    # delete_armature() # lihong add 20250126
    delete_gltf_not_imported() # lihong add 20250126

    if args.engine == 'BLENDER_EEVEE_NEXT':
        enable_backface_culling(flag=False)

    print('[INFO] Scene initialized.')

    # normalize scene
    scale, offset = normalize_scene()
    print('[INFO] Scene normalized.')


    init_lighting()
    # for obj in bpy.data.objects:
    #     if obj.type == 'MESH':
    #         obj.rotation_euler = (0, 0, 0)

    cam = init_camera() # Initialize camera, add camera to scene
    print('[INFO] Camera and lighting initialized.')

    # bpy.ops.wm.save_mainfile(filepath=os.path.join(arg.output_folder, 'test.blend'))

    # Initialize context
    init_render(engine=arg.engine, resolution=arg.resolution, geo_mode=arg.geo_mode, film_transparent=True)
    
    outputs, spec_nodes, _ = init_nodes(
        save_alpha=arg.save_alpha,
        save_depth=arg.save_depth,
        save_normal=arg.save_normal,
        save_low_normal=arg.save_low_normal,
        save_albedo=arg.save_albedo,
        save_mist=arg.save_mist,
        save_pbr=arg.save_pbr,
        save_env=True
    )

    # Override material
    if arg.geo_mode:
        override_material()
    
    # Create a list of views
    to_export = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": [offset.x, offset.y, offset.z],
        "resolution": arg.resolution,
        "frames": []
    }
    rotatios = json.loads(arg.rotatios)
    for i, rota in enumerate(rotatios):
        # import ipdb;ipdb.set_trace()
        # 相机在正面

        # static camera
        # cam.location = (0, -2, 0)
        # cam.rotation_euler = (90*np.pi/180, 0, 0)

        # rotating camera
        # Calculate circular position at height 2
        radius = 2  # Distance from center
        x = radius * np.cos(2 * np.pi * i / len(rotatios))
        y = radius * np.sin(2 * np.pi * i / len(rotatios)) 
        z = 1  # Height
        
        cam.location = (x, y, z)
        
        # Point camera toward center (0, 0, 0)
        direction = Vector((0, 0, 0)) - Vector((x, y, z))
        cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

        cam.data.lens = 16 / np.tan((40 / 180 * np.pi) / 2)

        # light_location = (
        #     view['radius'] * np.cos(view['yaw']) * np.cos(view['pitch']),
        #     view['radius'] * np.sin(view['yaw']) * np.cos(view['pitch']),
        #     view['radius'] * np.sin(view['pitch'])
        # )

        # setup_pointlight_position(light=scene_lights['default_light'], engery=1000, location=light_location, rotation_euler=(0, 0, 0))
        # if arg.save_depth:
            # spec_nodes['depth_map'].inputs[1].default_value = view['radius'] - 0.5 * np.sqrt(3)
            # spec_nodes['depth_map'].inputs[2].default_value = view['radius'] + 0.5 * np.sqrt(3)

        bpy.context.scene.render.filepath = os.path.join(arg.output_folder, f'{i:03d}.png') if not arg.save_low_normal else os.path.join(arg.output_folder, 'low_normal_image', f'{i:03d}_low_normal.png')

        for name, output in outputs.items():
            os.makedirs(os.path.join(arg.output_folder, f'{name.split(".")[0]}'), exist_ok=True)
            output.file_slots[0].path = os.path.join(arg.output_folder, f'{name.split(".")[0]}', f'{i:03d}_{name}')
            
        # Render the scene
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()

        # import ipdb;ipdb.set_trace()
        for name, output in outputs.items():
            ext = EXT[output.format.file_format]
            path = glob.glob(f'{output.file_slots[0].path}0001.{ext}')[0]
            os.rename(path, f'{output.file_slots[0].path}.{ext}')
            
        # Save camera parameters
        metadata = {
            "file_path": f'{os.path.join("image", f"{i:03d}.png")}',
            "camera_angle_x": (40 / 180 * np.pi),
            "transform_matrix": get_transform_matrix(cam),
            # "light_location": light_location,
            # "light_rotation": (0, 0, 0),
            # "light_energy": 1000,
        }
        for name, output in outputs.items():
            metadata[name] = f'{os.path.join(name.split(".")[0], f"{i:03d}_{name}.{EXT[output.format.file_format]}")}'

        # if arg.save_depth:
        #     metadata['depth'] = {
        #         'min': view['radius'] - 0.5 * np.sqrt(3),
        #         'max': view['radius'] + 0.5 * np.sqrt(3)
        #     }

        to_export["frames"].append(metadata)
    
    # Save the camera parameters
    with open(os.path.join(arg.output_folder, 'transforms.json'), 'w') as f:
        json.dump(to_export, f, indent=4)
        
    if arg.save_mesh:
        # triangulate meshes
        unhide_all_objects()
        convert_to_meshes()
        triangulate_meshes()
        print('[INFO] Meshes triangulated.')
        
        # export ply mesh
        # bpy.ops.export_mesh.ply(filepath=os.path.join(arg.output_folder, 'mesh.ply'))
        bpy.ops.wm.ply_export(filepath=os.path.join(arg.output_folder, 'mesh.ply'))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--rotatios', type=str, help='JSON string of views. Contains a list of {yaw, pitch, radius, fov} object.')
    parser.add_argument('--object', type=str, help='Path to the 3D model file to be rendered.')
    parser.add_argument('--output_folder', type=str, default='/tmp', help='The path the output will be dumped to.')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution of the images.')
    parser.add_argument('--engine', type=str, default='CYCLES', help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')
    parser.add_argument('--film_transparent', action='store_true', help='Film transparent mode for rendering.')
    parser.add_argument('--geo_mode', action='store_true', help='Geometry mode for rendering.')
    parser.add_argument('--save_alpha', action='store_true', help='Save the alpha maps.')
    parser.add_argument('--save_depth', action='store_true', help='Save the depth maps.')
    parser.add_argument('--save_normal', action='store_true', help='Save the normal maps.')
    parser.add_argument('--save_albedo', action='store_true', help='Save the albedo maps.')
    parser.add_argument('--save_mist', action='store_true', help='Save the mist distance maps.')
    parser.add_argument('--split_normal', action='store_true', help='Split the normals of the mesh.')
    parser.add_argument('--save_mesh', action='store_true', help='Save the mesh as a .ply file.')
    parser.add_argument('--save_pbr', action='store_true', help='Save the pbr maps.')
    parser.add_argument('--save_low_normal', action='store_true', help='Save the low normal maps.')

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)
    

# commond
'''
/baai-cwm-1/baai_cwm_ml/algorithm/hong.li/tools/blender-3.0.1-linux-x64/blender -b -P /baai-cwm-1/baai_cwm_ml/algorithm/hong.li/code/3dgen/TRELLIS/dataset_toolkits/blender_script/render_pbr.py --\
    --views 150 --object /baai-cwm-1/baai_cwm_ml/algorithm/hong.li/code/3dgen/TRELLIS/datasets/ObjaverseXL_sketchfab/raw/hf-objaverse-v1/glbs/000-034/1bb177d4e6f6470ba167ef5e4d8e2596.glb \
    --resolution 512 --output_folder /baai-cwm-1/baai_cwm_ml/algorithm/hong.li/code/3dgen/TRELLIS/datasets/tmp \
    --engine CYCLES --save_mesh --save_normal
'''
