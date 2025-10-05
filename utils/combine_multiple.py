import bpy
import os

# List your input .blend file paths
blend_files = [
    "datasets/glass_objects/a4e8232b-52d3-4713-9b95-e24b7e0aa660.blend",
    "datasets/glass_objects/0a7f00bd-6566-4376-8ac8-b47d9d786579.blend",
    # Add more as needed
]

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

# Loop through and append objects from all files
for blend_file in blend_files:
    append_all_objects(blend_file)

# Save the combined scene as test.blend
output_path = "test.blend"
bpy.ops.wm.save_as_mainfile(filepath=output_path)