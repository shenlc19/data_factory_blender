from concurrent.futures import ThreadPoolExecutor
import os, glob, natsort

def single_task(dir_name):
    os.system(f"python vis_normal.py {dir_name}")

model_paths = natsort.natsorted(glob.glob('datasets/hunyuan_demo/*'))[:1]
with ThreadPoolExecutor(max_workers=32) as executor:
    futures = [executor.submit(single_task, model_name) for idx, model_name in enumerate(model_paths)]