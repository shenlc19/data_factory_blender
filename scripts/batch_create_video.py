import os, glob, natsort
from concurrent.futures import ThreadPoolExecutor

def process_single(scene_id):
    os.system(
        f"python create_video.py {scene_id}"
    )

model_paths = natsort.natsorted(glob.glob('datasets/hunyuan_demo/*'))
breakpoint()

with ThreadPoolExecutor(max_workers=32) as executor:
    futures = [executor.submit(process_single, scene_id) for scene_id in model_paths]