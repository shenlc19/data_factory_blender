import blenderproc as bproc
import os, glob, json, natsort, sys
import numpy as np
import random

import bpy

def add_area_light(option: str) -> None:
    assert option in ['fixed', 'random']

    # bpy.data.objects["Light"].select_set(True)
    # bpy.ops.object.delete()

    bpy.ops.object.light_add(type="AREA")
    light = bpy.data.lights["Area"]

    if option == 'fixed':
        light.energy = 160000
        bpy.data.objects["Area"].location[0] = 0
        bpy.data.objects["Area"].location[1] = 2
        bpy.data.objects["Area"].location[2] = 2.5

    elif option == 'random':
        light.energy = random.uniform(80000, 120000)
        bpy.data.objects["Area"].location[0] = random.uniform(-2., 2.)
        bpy.data.objects["Area"].location[1] = random.uniform(-2., 2.)
        bpy.data.objects["Area"].location[2] = random.uniform(1.0, 3.0)

    bpy.data.objects["Area"].scale[0] = 200
    bpy.data.objects["Area"].scale[1] = 200
    bpy.data.objects["Area"].scale[2] = 200


def main():

    import sys

    base_dir = sys.argv[1]
    
    # json_path = '/DATA_EDS2/shenlc2403/data_factory/data_factory_blender/datasets/glassverse_v0_120_views_hdri/003000/transforms.json'
    json_path = os.path.join(sys.argv[1], 'transforms.json')
    
    blend_path = os.path.join(sys.argv[1], 'lm.blend')
    bproc.loader.load_blend(blend_path)
    
    
    # Lighting
    add_area_light(option='fixed')

    bproc.camera.set_resolution(512, 512)
    for obj in bpy.context.scene.objects:
        if obj.type == 'CAMERA':
            print(f"Before: sensor_height = {obj.data.sensor_height}")
            obj.data.sensor_height = 32  # Remove the duplicate assignment
            print(f"After: sensor_height = {obj.data.sensor_height}")
            print(f"Before: sensor_width = {obj.data.sensor_width}")
            obj.data.sensor_width = 32  # Remove the duplicate assignment
            print(f"After: sensor_width = {obj.data.sensor_width}")
    
    fov_x, fov_y = bproc.camera.get_fov()
    bpy.context.scene.render.engine = 'CYCLES'
    
    with open(json_path) as f:
        transforms = json.load(f)
        
    # for frame in transforms['frames'][:10]:
    #     camera_pose = frame['transform_matrix']
    #     bproc.camera.add_camera_pose(camera_pose)

    for frame in transforms:
        camera_pose = frame
        bproc.camera.add_camera_pose(camera_pose)
        
    bproc.renderer.set_output_format(enable_transparency=True)
    bproc.renderer.enable_depth_output(True)
    # render the whole pipeline
    data = bproc.renderer.render()
    data.update(bproc.renderer.render_segmap(map_by=["class", "instance", "name"]))
    
    scene_name = os.path.basename(base_dir)
    output_dir = f'output/glasverse_mask/{scene_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # write the data to a .hdf5 container (had better create a new "hdf5" subdir)
    output_split_dir = os.path.join(output_dir, "hdf5")
    os.makedirs(output_split_dir, exist_ok=True)
    bproc.writer.write_hdf5(output_split_dir, data)

if __name__ == '__main__':
    bproc.init()
    main()