import blenderproc as bproc
import bpy

bproc.init()

import os, glob

file_path = '/DATA_EDS2/shenlc2403/data_factory/data_factory_blender/datasets/primitives_v0/00000/lm.blend'

def process_single_file(file_path):

    # Load the Blender file
    bpy.ops.wm.open_mainfile(filepath=file_path)

    # Update all image paths that contain old directory
    old_path = "/baai-cwm-1/baai_cwm_ml/algorithm/shaocong.xu/exp/BlenderProc/resources/cctextures"
    new_path = "/DATA_EDS2/shenlc2403/data_factory/BlenderProc/resources/cctextures"

    for img in bpy.data.images:
        print(img.filepath)
        if old_path in img.filepath:
            img.filepath = img.filepath.replace(old_path, new_path)
            img.reload()

    old_path = "/baai-cwm-1/baai_cwm_ml/algorithm/hong.li/public_data/rendering_data/matsynth_processed_v2/"
    new_path = "/DATA_EDS2/shenlc2403/data_factory/matsynth_processed_v2/"

    for img in bpy.data.images:
        print(img.filepath)
        if old_path in img.filepath:
            img.filepath = img.filepath.replace(old_path, new_path)
            img.reload()

    # Save the updated Blender file
    bpy.ops.wm.save_as_mainfile(filepath=file_path)

file_paths = glob.glob('/DATA_EDS2/shenlc2403/data_factory/data_factory_blender/datasets/primitives_v0/*/lm.blend')
for file in file_paths:
    process_single_file(file)