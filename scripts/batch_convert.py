from concurrent.futures import ThreadPoolExecutor
import os, glob, natsort

def process_single(model_path):
    if os.path.exists(model_path.replace('.blend', '.obj').replace('3Ddata', '3Ddata_obj')):
        print(os.path.basename(model_path), 'exists, skip.')
    os.system(f'~/blender/blender-4.3.2-linux-x64/blender -b -P convert_blend_to_obj.py {model_path}')

model_paths = natsort.natsorted(glob.glob('datasets/Carverse/blenderkit/3Ddata/*.blend'))

with ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(process_single, model_name) for model_name in model_paths]