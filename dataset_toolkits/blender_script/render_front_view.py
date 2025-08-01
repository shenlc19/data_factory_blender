import argparse, sys, os, math, re, glob
from typing import *
import bpy
from mathutils import Vector, Matrix
import numpy as np
import json
import glob

def make_output_dir(output_dir: str, name: str):
    output_dir = os.path.join(output_dir, name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

"""=============== BLENDER ==============="""

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    # "obj": bpy.ops.import_scene.obj,
    "obj": bpy.ops.wm.obj_import,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

EXT = {
    'PNG': 'png',
    'JPEG': 'jpg',
    'OPEN_EXR': 'exr',
    'TIFF': 'tiff',
    'BMP': 'bmp',
    'HDR': 'hdr',
    'TARGA': 'tga'
}

def init_render(engine='CYCLES', resolution=512, geo_mode=False, device_index=0):
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True
    
    if engine == 'CYCLES': 
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.samples = 128 if not geo_mode else 1
        bpy.context.scene.cycles.filter_type = 'BOX'
        bpy.context.scene.cycles.filter_width = 1
        bpy.context.scene.cycles.diffuse_bounces = 1
        bpy.context.scene.cycles.glossy_bounces = 1
        bpy.context.scene.cycles.transparent_max_bounces = 3 if not geo_mode else 0
        bpy.context.scene.cycles.transmission_bounces = 3 if not geo_mode else 1
        bpy.context.scene.cycles.use_denoising = True

        # 设置可见的设备  
        # import ipdb; ipdb.set_trace()
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        
        # bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'OPTIX'
        # bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'OPENCL'

        # bpy.context.scene.cycles.optix.device_index = 0
    elif engine == 'BLENDER_EEVEE':
        bpy.context.scene.eevee.taa_render_samples = 128
        bpy.context.scene.eevee.use_gtao = True
        bpy.context.scene.eevee.use_ssr = True
        bpy.context.scene.eevee.use_bloom = True
        bpy.context.scene.render.use_high_quality_normals = True

    elif engine == 'BLENDER_EEVEE_NEXT': # 4.3 version
        bpy.context.scene.eevee.taa_render_samples = 128
        bpy.context.scene.eevee.use_gtao = True
        bpy.context.scene.render.use_high_quality_normals = True
        bpy.context.scene.eevee.use_shadows = True
        bpy.context.scene.eevee.use_raytracing = True
        # bpy.context.scene.eevee.compositer_device = 'GPU'

    
def init_nodes(save_depth=False, save_normal=False, save_low_normal=False, save_albedo=False, save_mist=False, save_pbr=False):
    if not any([save_depth, save_normal, save_low_normal, save_albedo, save_mist, save_pbr]):
        return {}, {}
    outputs = {}
    spec_nodes = {}

    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True
    # bpy.context.scene.view_layers['View Layer'].use_pass_z = save_depth
    # bpy.context.scene.view_layers['View Layer'].use_pass_normal = save_normal
    # bpy.context.scene.view_layers['View Layer'].use_pass_diffuse_color = save_albedo
    # bpy.context.scene.view_layers['View Layer'].use_pass_mist = save_mist
    
    bpy.context.view_layer.use_pass_z = save_depth
    bpy.context.view_layer.use_pass_normal = save_normal or save_low_normal
    bpy.context.view_layer.use_pass_diffuse_color = save_albedo
    bpy.context.view_layer.use_pass_mist = save_mist

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    for n in nodes:
        nodes.remove(n)
    
    render_layers = nodes.new('CompositorNodeRLayers')
    
    if save_depth:
        bpy.context.view_layer.use_pass_z = save_depth
        depth_file_output = nodes.new('CompositorNodeOutputFile')
        depth_file_output.base_path = ''
        depth_file_output.file_slots[0].use_node_format = True
        depth_file_output.format.file_format = 'OPEN_EXR'
        depth_file_output.format.color_depth = '16'
        depth_file_output.format.color_mode = 'RGBA'

        alpha_depth = nodes.new('CompositorNodeSetAlpha')
        links.new(render_layers.outputs['Depth'], alpha_depth.inputs['Image'])
        links.new(render_layers.outputs['Alpha'], alpha_depth.inputs['Alpha'])
        links.new(alpha_depth.outputs['Image'], depth_file_output.inputs['Image'])
        # links.new(render_layers.outputs["Depth"], depth_file_output.inputs["Image"])

        outputs['depth'] = depth_file_output
    
    if save_normal:
        # import ipdb; ipdb.set_trace()
        normal_file_output = nodes.new('CompositorNodeOutputFile')
        normal_file_output.base_path = ''
        normal_file_output.file_slots[0].use_node_format = True
        normal_file_output.format.file_format = 'OPEN_EXR'
        normal_file_output.format.color_mode = 'RGB'
        normal_file_output.format.color_depth = '16'

        # Add a Math node to remap the normal values
        # math_node = nodes.new('CompositorNodeMath')
        # math_node.operation = 'MULTIPLY'
        # math_node.inputs[1].default_value = 0.5

        # # Add another Math node to offset
        # offset_node = nodes.new('CompositorNodeMath')
        # offset_node.operation = 'ADD'
        # offset_node.inputs[1].default_value = 0.5

        # # Connect the nodes
        # links.new(render_layers.outputs['Normal'], math_node.inputs[0])
        # links.new(math_node.outputs[0], offset_node.inputs[0])
        # links.new(offset_node.outputs[0], normal_file_output.inputs[0])

        links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
        
        outputs['normal'] = normal_file_output
    
    if save_low_normal:
        # link remove the normal node
        unable_normals_texture_output()
        low_normal_file_output = nodes.new('CompositorNodeOutputFile')
        low_normal_file_output.base_path = ''
        low_normal_file_output.file_slots[0].use_node_format = True
        low_normal_file_output.format.file_format = 'OPEN_EXR'
        low_normal_file_output.format.color_mode = 'RGB'
        low_normal_file_output.format.color_depth = '16'
        
        # Add a Math node to remap the normal values
        # math_node = nodes.new('CompositorNodeMath')
        # math_node.operation = 'MULTIPLY'
        # math_node.inputs[1].default_value = 0.5

        # # Add another Math node to offset
        # offset_node = nodes.new('CompositorNodeMath')
        # offset_node.operation = 'ADD'
        # offset_node.inputs[1].default_value = 0.5

        # # Connect the nodes
        # links.new(render_layers.outputs['Normal'], math_node.inputs[0])
        # links.new(math_node.outputs[0], offset_node.inputs[0])
        # links.new(offset_node.outputs[0], low_normal_file_output.inputs[0])
        links.new(render_layers.outputs['Normal'], low_normal_file_output.inputs[0])
        outputs['low_normal'] = low_normal_file_output
            
    if save_albedo:
        albedo_file_output = nodes.new('CompositorNodeOutputFile')
        albedo_file_output.base_path = ''
        albedo_file_output.file_slots[0].use_node_format = True
        albedo_file_output.format.file_format = 'PNG'
        albedo_file_output.format.color_mode = 'RGBA'
        albedo_file_output.format.color_depth = '8'
        
        alpha_albedo = nodes.new('CompositorNodeSetAlpha')
        
        links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
        links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])
        links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])
        
        outputs['albedo'] = albedo_file_output
        
    if save_mist:
        bpy.data.worlds['World'].mist_settings.start = 0
        bpy.data.worlds['World'].mist_settings.depth = 10
        
        mist_file_output = nodes.new('CompositorNodeOutputFile')
        mist_file_output.base_path = ''
        mist_file_output.file_slots[0].use_node_format = True
        mist_file_output.format.file_format = 'PNG'
        mist_file_output.format.color_mode = 'BW'
        mist_file_output.format.color_depth = '16'
        
        links.new(render_layers.outputs['Mist'], mist_file_output.inputs[0])
        
        outputs['mist'] = mist_file_output
    
    if save_pbr:
        aov_base_color = enable_pbr_output(rl=render_layers, attr_name="Base Color", color_mode="RGBA", file_format="PNG", color_depth="16", outputs=outputs)
        aov_roughness = enable_pbr_output(rl=render_layers, attr_name="Roughness", color_mode="BW", file_format="OPEN_EXR", color_depth="16",outputs=outputs)
        aov_metallic = enable_pbr_output(rl=render_layers, attr_name="Metallic", color_mode="BW", file_format="OPEN_EXR", color_depth="16",outputs=outputs)

    return outputs, spec_nodes

def unable_normals_texture_output():
    for material in bpy.data.materials:
        material.use_nodes = True
        node_tree = material.node_tree
        nodes = node_tree.nodes
        if "Principled BSDF" not in nodes.keys():
            continue
        else:
            normal_input = nodes["Principled BSDF"].inputs["Normal"]
            if normal_input.is_linked:
                # 断开该链接
                node_tree.links.remove(normal_input.links[0])

def enable_pbr_output(rl, output_dir='', attr_name='Roughness', color_mode="RGB", file_format="PNG", color_depth="8", file_prefix: str = "", outputs=dict()):

    flag = False

    for material in bpy.data.materials:
        if not material.use_nodes:  
            continue

        node_tree = material.node_tree
        nodes = node_tree.nodes
        if "Principled BSDF" not in nodes.keys():
            flag = True
            break

        roughness_input = nodes["Principled BSDF"].inputs[attr_name]

        if roughness_input.is_linked:
            linked_node = roughness_input.links[0].from_node
            linked_socket = roughness_input.links[0].from_socket

            aov_output = nodes.new("ShaderNodeOutputAOV")
            aov_output.aov_name = attr_name
            
            node_tree.links.new(linked_socket, aov_output.inputs[0])

        else:
            fixed_roughness = roughness_input.default_value
            if isinstance(fixed_roughness, float):
                roughness_value = nodes.new("ShaderNodeValue")
                input_idx = 1
            else:
                roughness_value = nodes.new("ShaderNodeRGB")
                input_idx = 0

            roughness_value.outputs[0].default_value = fixed_roughness

            aov_output = nodes.new("ShaderNodeOutputAOV")
            aov_output.aov_name = attr_name
            node_tree.links.new(roughness_value.outputs[0], aov_output.inputs[0])

    if flag:
        return
    
    tree = bpy.context.scene.node_tree
    links = tree.links
    if "Render Layers" not in tree.nodes:
        rl = tree.nodes.new("CompositorNodeRLayers")
    else:
        rl = tree.nodes["Render Layers"]

    roughness_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    roughness_file_output.base_path = ''
    roughness_file_output.file_slots[0].use_node_format = True
    roughness_file_output.format.file_format = file_format
    roughness_file_output.format.color_mode = color_mode
    roughness_file_output.format.color_depth = color_depth
    
    # 获取或创建AOV  
    view_layer = bpy.context.view_layer  
    aov = view_layer.aovs.add()
    aov.name = attr_name  

    if color_mode == "RGBA":
        roughness_alpha = tree.nodes.new(type="CompositorNodeSetAlpha")
        tree.links.new(rl.outputs[attr_name], roughness_alpha.inputs["Image"])
        tree.links.new(rl.outputs["Alpha"], roughness_alpha.inputs["Alpha"])
        tree.links.new(roughness_alpha.outputs["Image"], roughness_file_output.inputs["Image"])
    else:
        tree.links.new(rl.outputs[attr_name], roughness_file_output.inputs["Image"])

    outputs[attr_name] = roughness_file_output

    return aov

def init_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def init_camera():
    cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty
    return cam

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
    top_light.data.energy = 10000
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


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    print(f"Loading object from {object_path}")
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True, import_shading='NORMALS')
    # elif file_extension in {"obj"}:
    #     import_function(filepath=object_path)
    else:
        import_function(filepath=object_path)

def delete_animation_data():
    # 关闭选中对象的所有动画效果  
    # bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        # 删除关键帧  
        if obj.animation_data:  
            obj.animation_data_clear()  
        # 删除物理模拟  
        if obj.rigid_body:  
            bpy.context.view_layer.objects.active = obj  
            bpy.ops.rigidbody.object_remove()  
        # 删除修改器（动态绘画、布料等）  
        if obj.modifiers:  
            for mod in obj.modifiers:  
                if mod.type in ['CLOTH', 'SOFT_BODY', 'FLUID', 'DYNAMIC_PAINT']:  
                    obj.modifiers.remove(mod)  
        # 删除粒子系统  
        if obj.particle_systems:  
            for ps in obj.particle_systems:  
                obj.particle_systems.remove(ps)  

def delete_armature():
    for armature in bpy.data.armatures:
        armature = bpy.data.armatures.get(armature.name)
        bpy.data.armatures.remove(armature)

def delete_gltf_not_imported():
    # delete Icosphere*
    # for obj in bpy.context.scene.objects:
    #     if obj.name.startswith("Icosphere"):
    #         bpy.data.objects.remove(obj)
    collection_name = "glTF_not_exported"  # 替换为你想要删除的集合名称  

    # 获取集合  
    collection = bpy.data.collections.get(collection_name)  

    if collection:  
        # 递归删除集合及其所有内容  
        bpy.data.collections.remove(collection, do_unlink=True)  
        
def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    # bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)
        
def split_mesh_normal():
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.split_normals()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action="DESELECT")
            
def delete_custom_normals():
     for this_obj in bpy.data.objects:
        if this_obj.type == "MESH":
            bpy.context.view_layer.objects.active = this_obj
            bpy.ops.mesh.customdata_custom_splitnormals_clear()

def override_material():
    new_mat = bpy.data.materials.new(name="Override0123456789")
    new_mat.use_nodes = True
    new_mat.node_tree.nodes.clear()
    bsdf = new_mat.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
    bsdf.inputs[0].default_value = (0.5, 0.5, 0.5, 1)
    bsdf.inputs[1].default_value = 1
    output = new_mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    new_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    bpy.context.scene.view_layers['View Layer'].material_override = new_mat

def unhide_all_objects() -> None:
    """Unhides all objects in the scene.

    Returns:
        None
    """
    for obj in bpy.context.scene.objects:
        obj.hide_set(False)
        
def convert_to_meshes() -> None:
    """Converts all objects in the scene to meshes.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"][0]
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
    bpy.ops.object.convert(target="MESH")
        
def triangulate_meshes() -> None:
    """Triangulates all meshes in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.reveal()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")

def scene_bbox() -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
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

def normalize_scene() -> Tuple[float, Vector]:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        Tuple[float, Vector]: The scale factor and the offset applied to the scene.
    """
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(scene_root_objects) > 1:
        # create an empty object to be used as a parent for all root objects
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)

        # parent all root objects to the empty object
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    
    return scale, offset

def get_transform_matrix(obj: bpy.types.Object) -> list:
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0, 0, 0, 1])
    return matrix

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
    
    # import ipdb; ipdb.set_trace()
    delete_animation_data() # lihong add 20250126
    # delete_armature() # lihong add 20250126
    delete_gltf_not_imported() # lihong add 20250126
    print('[INFO] Scene initialized.')
    
    # normalize scene
    scale, offset = normalize_scene()
    print('[INFO] Scene normalized.')
    
    # Initialize camera and lighting
    cam = init_camera()
    init_lighting()
    print('[INFO] Camera and lighting initialized.')

    # Initialize context
    init_render(engine=arg.engine, resolution=arg.resolution)
    outputs, spec_nodes = init_nodes()
    
    # Create a list of views
    to_export = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": [offset.x, offset.y, offset.z],
        "resolution": arg.resolution,
        "frames": []
    }
    # for i, view in enumerate(views):
        
    cam.location = (
        2, 0, 0
    )

    save_name = 'front_view'

    bpy.context.scene.render.filepath = os.path.join(arg.output_folder, f'{save_name}.png')
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
        # os.rename(path, f'{output.file_slots[0].path}.{ext}')
        os.rename(path, f'{save_name}.{ext}')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--object', type=str, help='Path to the 3D model file to be rendered.')
    parser.add_argument('--output_folder', type=str, default='/tmp', help='The path the output will be dumped to.')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution of the images.')
    parser.add_argument('--engine', type=str, default='CYCLES', help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')
    parser.add_argument('--split_normal', action='store_true', help='Split the normals of the mesh.')

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
