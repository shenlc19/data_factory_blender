import bpy
import os
import bmesh

OUTPUT_DIR = 'datasets/Carverse/D760/batch3/single_objects'

def save_each_mesh_as_blend():
    """
    Save each mesh object as a separate .blend file
    Simple approach that actually works
    """
    
    original_filepath = bpy.data.filepath
    
    if not original_filepath:
        print("Error: Save the current file first!")
        return
    
    # output_dir = os.path.dirname(original_filepath)
    output_dir = OUTPUT_DIR
    original_filename = os.path.splitext(os.path.basename(original_filepath))[0]
    
    # Get list of mesh object names ONLY
    mesh_names = [obj.name for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    if not mesh_names:
        print("No mesh objects found!")
        return
    
    print(f"Found {len(mesh_names)} mesh objects: {mesh_names}")
    
    # Process each mesh
    for mesh_name in mesh_names:
        print(f"Processing: {mesh_name}")
        
        # Reopen original file fresh
        bpy.ops.wm.open_mainfile(filepath=original_filepath)
        
        # Find the target mesh object
        target_obj = None
        for obj in bpy.context.scene.objects:
            if obj.name == mesh_name and obj.type == 'MESH':
                target_obj = obj
                break
        
        if not target_obj:
            print(f"Could not find mesh: {mesh_name}")
            continue
        
        # Select only the target mesh
        bpy.ops.object.select_all(action='DESELECT')
        target_obj.select_set(True)
        bpy.context.view_layer.objects.active = target_obj
        
        # Delete everything else
        bpy.ops.object.select_all(action='INVERT')
        bpy.ops.object.delete()
        
        # Save the file
        safe_name = "".join(c for c in mesh_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        if not safe_name:
            safe_name = "mesh"
        
        output_filename = f"{original_filename}_{safe_name}.blend"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            bpy.ops.wm.save_as_mainfile(filepath=output_path)
            print(f"Saved: {output_filename}")
        except Exception as e:
            print(f"Error saving {output_filename}: {e}")
    
    # Reload original file one last time
    bpy.ops.wm.open_mainfile(filepath=original_filepath)
    print("Done!")

# Run the function
if __name__ == "__main__":
    import sys
    # bpy.ops.wm.open_mainfile(filepath='datasets/glass_objects/a4e8232b-52d3-4713-9b95-e24b7e0aa660.blend')
    bpy.ops.wm.open_mainfile(filepath=sys.argv[4])
    # Choose which method to use:
    # Method 1: Reload file each time (slower but more memory efficient)
    save_each_mesh_as_blend()