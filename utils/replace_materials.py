import blenderproc as bproc

import bpy
import os, shutil, glob, sys

from blenderproc.python.material import MaterialLoaderUtility
from blenderproc.python.types.MaterialUtility import Material
from blenderproc.python.utility.Utility import Utility, resolve_path
from blenderproc.python.loader.CCMaterialLoader import _CCMaterialLoader
from blenderproc.python.types.MeshObjectUtility import MeshObject

bproc.init()

CC_BASE_DIR = '/DATA_EDS2/shenlc2403/data_factory/BlenderProc/resources/cctextures'
MATSYNTH_BASE_DIR = '/DATA_EDS2/shenlc2403/data_factory/matsynth_processed_v2'

def load_ccmaterial(base_path):
    material_name = os.path.basename(base_path)
    base_image_path = os.path.join(base_path, f"{material_name}_2K-JPG_Color.jpg")
    asset = material_name
    add_custom_properties = {}
    new_mat = MaterialLoaderUtility.create_new_cc_material(asset, add_custom_properties)
    # construct all image paths
    ambient_occlusion_image_path = base_image_path.replace("Color", "AmbientOcclusion")
    metallic_image_path = base_image_path.replace("Color", "Metalness")
    roughness_image_path = base_image_path.replace("Color", "Roughness")
    alpha_image_path = base_image_path.replace("Color", "Opacity")
    normal_image_path = base_image_path.replace("Color", "Normal")
    # Filenames have been changed (blender uses opengl normal maps)
    if not os.path.exists(normal_image_path):
        normal_image_path = base_image_path.replace("Color", "NormalGL")
    displacement_image_path = base_image_path.replace("Color", "Displacement")

    _CCMaterialLoader.create_material(new_mat, base_image_path, ambient_occlusion_image_path,
                                                  metallic_image_path, roughness_image_path, alpha_image_path,
                                                  normal_image_path, displacement_image_path)

    return Material(new_mat)

def load_matsynth_material(base_path):
    material_name = os.path.basename(base_path)
    base_image_path = glob.glob(os.path.join(base_path, "*Color.*"))[0]
    asset = material_name
    add_custom_properties = {}
    new_mat = MaterialLoaderUtility.create_new_cc_material(asset, add_custom_properties)

    ambient_occlusion_image_path = base_image_path.replace("Color", "AmbientOcclusion")
    metallic_image_path = base_image_path.replace("Color", "Metallness")
    roughness_image_path = base_image_path.replace("Color", "Roughness")
    alpha_image_path = base_image_path.replace("Color", "Opacity")
    normal_image_path = base_image_path.replace("Color", "NormalGL")
    displacement_image_path = base_image_path.replace("Color", "Displacement")

    _CCMaterialLoader.create_material(new_mat, base_image_path, ambient_occlusion_image_path,
                                                  metallic_image_path, roughness_image_path, alpha_image_path,
                                                  normal_image_path, displacement_image_path)

    return Material(new_mat)

base_path = '/DATA_EDS2/shenlc2403/data_factory/BlenderProc/resources/cctextures/Tiles013'
mat = load_ccmaterial(base_path)

matsynth_base_path = '/DATA_EDS2/shenlc2403/data_factory/matsynth_processed_v2/st_kitchen_tiling_002'
matsynth_mat = load_matsynth_material(matsynth_base_path)

if __name__ == '__main__':
    blend_file_path = sys.argv[1]
    bproc.loader.load_blend(path=blend_file_path)

    scene_objects = []

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and ('Plane' not in obj.name):
            scene_objects.append(MeshObject(obj))

    import random
    replace_num = random.randint(1, len(scene_objects))
    replace_material_objects = random.sample(scene_objects, k=replace_num)

    cc_materials_base_paths = random.sample(
        glob.glob(os.path.join(CC_BASE_DIR, '*')),
        k=replace_num,
    )
    cc_materials_to_replace = [load_ccmaterial(mat_base_path) for mat_base_path in cc_materials_base_paths]

    matsynth_materials_base_paths = random.sample(
        glob.glob(os.path.join(MATSYNTH_BASE_DIR, '*')),
        k=replace_num,
    )
    matsynth_materials_to_replace = [load_matsynth_material(mat_base_path) for mat_base_path in matsynth_materials_base_paths]
    # ['/DATA_EDS2/shenlc2403/data_factory/BlenderProc/resources/cctextures/WoodSiding001',
    # '/DATA_EDS2/shenlc2403/data_factory/BlenderProc/resources/cctextures/Tiles012',
    # '/DATA_EDS2/shenlc2403/data_factory/BlenderProc/resources/cctextures/Tiles013',]]

    for idx, replace_material_object in enumerate(replace_material_objects):
        mat_type = random.choice(['cc', 'matsynth'])
        replace_material_object.replace_materials(matsynth_materials_to_replace[idx])
        # if mat_type == 'cc':
        #     replace_material_object.replace_materials(cc_materials_to_replace[idx])
        # elif mat_type == 'matsynth':
        #     replace_material_object.replace_materials(matsynth_materials_to_replace[idx])

    ORIG_BASE_DIR = 'datasets/primitives_v0'
    SAVE_BASE_DIR = 'datasets/primitives_v0_material_replaced'

    scene_name = os.path.basename(os.path.dirname(blend_file_path))
    save_dir = os.path.join(SAVE_BASE_DIR, scene_name)
    os.makedirs(save_dir, exist_ok=True)

    orig_save_dir = os.path.join(ORIG_BASE_DIR, scene_name)

    shutil.copy(os.path.join(orig_save_dir, 'transforms.json'),
                os.path.join(save_dir, 'transforms.json'))

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(save_dir, 'lm.blend'))
