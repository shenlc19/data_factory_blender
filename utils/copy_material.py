import bpy
import os

# Method 1: Append materials from another .blend file
def append_materials_from_file(filepath, material_names=None):
    """
    Append materials from another .blend file
    
    Args:
        filepath: Path to the source .blend file
        material_names: List of material names to append (None = all materials)
    """
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    # Append materials
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        if material_names:
            # Append specific materials
            data_to.materials = [mat for mat in data_from.materials if mat in material_names]
        else:
            # Append all materials
            data_to.materials = data_from.materials
    
    print(f"Appended {len(data_to.materials)} materials from {filepath}")

# Method 2: Link materials from another .blend file
def link_materials_from_file(filepath, material_names=None):
    """
    Link materials from another .blend file (updates when source changes)
    
    Args:
        filepath: Path to the source .blend file
        material_names: List of material names to link (None = all materials)
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    # Link materials
    with bpy.data.libraries.load(filepath, link=True) as (data_from, data_to):
        if material_names:
            data_to.materials = [mat for mat in data_from.materials if mat in material_names]
        else:
            data_to.materials = data_from.materials
    
    print(f"Linked {len(data_to.materials)} materials from {filepath}")

# Method 3: Apply material to selected objects
def apply_material_to_objects(material_name, object_names=None):
    """
    Apply a material to objects
    
    Args:
        material_name: Name of the material to apply
        object_names: List of object names (None = selected objects)
    """
    # Get the material
    material = bpy.data.materials.get(material_name)
    if not material:
        print(f"Material '{material_name}' not found")
        return
    
    # Get target objects
    if object_names:
        objects = [bpy.data.objects.get(name) for name in object_names if bpy.data.objects.get(name)]
    else:
        objects = bpy.context.selected_objects
    
    # Apply material to objects
    for obj in objects:
        if obj and obj.type == 'MESH':
            # Clear existing materials
            obj.data.materials.clear()
            # Add the new material
            obj.data.materials.append(material)
            print(f"Applied material '{material_name}' to {obj.name}")

# Method 4: Copy all materials from objects in another file
def copy_materials_from_objects(filepath, object_names=None):
    """
    Copy materials from specific objects in another file
    
    Args:
        filepath: Path to the source .blend file
        object_names: List of object names to copy materials from
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    # Append objects with their materials
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        if object_names:
            data_to.objects = [obj for obj in data_from.objects if obj in object_names]
        else:
            data_to.objects = data_from.objects
    
    # Link objects to scene temporarily to access their materials
    imported_materials = set()
    for obj in data_to.objects:
        if obj.data and hasattr(obj.data, 'materials'):
            for mat in obj.data.materials:
                if mat:
                    imported_materials.add(mat.name)
    
    # Remove the temporary objects (keep materials)
    for obj in data_to.objects:
        bpy.data.objects.remove(obj)
    
    print(f"Imported {len(imported_materials)} materials from objects")

# Method 5: Save current materials to a library file
def save_materials_to_library(filepath):
    """
    Save all materials in current scene to a library file
    
    Args:
        filepath: Path where to save the library file
    """
    # Mark all materials as assets (optional, for Asset Browser)
    for mat in bpy.data.materials:
        mat.asset_mark()
    
    # Save the file
    bpy.ops.wm.save_as_mainfile(filepath=filepath)
    print(f"Saved material library to {filepath}")

# Example usage:
if __name__ == "__main__":
    # Example 1: Append specific materials
    source_file = "/path/to/your/source_file.blend"
    append_materials_from_file(source_file, ["Material.001", "Glass_Material"])
    
    # Example 2: Apply material to selected objects
    apply_material_to_objects("Material.001")
    
    # Example 3: Apply material to specific objects
    apply_material_to_objects("Glass_Material", ["Cube", "Sphere"])
    
    # Example 4: Append all materials
    append_materials_from_file(source_file)
    
    # Example 5: Link materials (they'll update when source file changes)
    link_materials_from_file(source_file, ["Metal_Material"])

# Utility functions
def list_materials_in_file(filepath):
    """List all materials in a .blend file without importing them"""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []
    
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        return list(data_from.materials)

def list_current_materials():
    """List all materials in current file"""
    return [mat.name for mat in bpy.data.materials]

# Advanced: Batch process multiple files
def batch_append_materials(source_files, material_filter=None):
    """
    Append materials from multiple files
    
    Args:
        source_files: List of file paths
        material_filter: Function to filter materials (e.g., lambda name: 'metal' in name.lower())
    """
    for filepath in source_files:
        if os.path.exists(filepath):
            with bpy.data.libraries.load(filepath) as (data_from, data_to):
                if material_filter:
                    data_to.materials = [mat for mat in data_from.materials if material_filter(mat)]
                else:
                    data_to.materials = data_from.materials
            print(f"Processed {filepath}")
        else:
            print(f"File not found: {filepath}")

bpy.ops.wm.open_mainfile(filepath='datasets/primitives_v0/00000/lm.blend')

append_materials_from_file('B端浅色的副本_副本.blend')

for idx, obj in enumerate(bpy.data.objects):
    if obj and obj.type == 'MESH' and obj.name != 'Plane':
        obj.data.materials.clear()
        obj.data.materials.append(bpy.data.materials[idx])

save_path = 'datasets/transparent_primitives_v0/00000/lm.blend'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=save_path)