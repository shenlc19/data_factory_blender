import os, glob, natsort
from concurrent.futures import ThreadPoolExecutor

def process_single(scene_id):
    os.system(
        f"blenderproc run scene_creator/custom_objects.py datasets/single_glass_objects_depth_filtered \
        datasets/glassverse_v0_filtered/{scene_id:06d}"
    )

with ThreadPoolExecutor(max_workers=32) as executor:
    futures = [executor.submit(process_single, idx) for idx in range(3000, 8000)]