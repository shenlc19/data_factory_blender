from concurrent.futures import ThreadPoolExecutor
import os, glob, natsort

MAX_WORKERS=32

def process_single(blend_file):
    os.system(f"/DATA_EDS2/shenlc2403/blender/blender-4.3.2-linux-x64/blender -b -P split_objects.py \
            {blend_file}")

blend_files = natsort.natsorted(glob.glob('datasets/Carverse/3d_168/raw/blend/*'))

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_single, blend_file) for blend_file in blend_files]