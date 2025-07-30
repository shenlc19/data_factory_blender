import bpy
import sys
import argparse
import bmesh
from mathutils import Vector
import numpy as np

def main(args):
    # Clear existing mesh objects (optional - remove if you want to keep existing objects)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    
    # Import the PLY file
    bpy.ops.wm.ply_import(filepath=args.object)
    
    # Get the imported object (it should be the active object after import)
    obj = bpy.context.active_object
    
    if obj is None or obj.type != 'MESH':
        print("Error: No mesh object found after import")
        return
    
    bpy.context.scene.cursor.location = args.center  # or any coordinates

    # Then set origin to cursor
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')

    obj.location = Vector(list(-np.array(args.center)))
    scale = args.scale
    obj.scale = (scale, scale, scale)
    obj.location = Vector([0, 0, 0])

    bpy.ops.wm.save_mainfile(filepath=args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Import PLY file, position at center coordinates, move to origin, and scale.')
    
    parser.add_argument('--object', type=str, required=True,
                       help='Path to the PLY file to be imported.')
    
    parser.add_argument('--save_path', type=str, required=True,
                       help='Path to the PLY file to be imported.')
    
    parser.add_argument('--center', type=float, nargs=3, default=None,
                       help='Center coordinates as three values: x y z (optional).')
    
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Scale factor to resize the object (default: 1.0).')
    
    # Parse arguments after "--" separator
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    
    main(args)