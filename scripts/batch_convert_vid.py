from concurrent.futures import ThreadPoolExecutor
import os, glob, natsort

def single_task(dir_name):
    model_name = os.path.basename(dir_name)
    os.system(f"python convert_vid.py {dir_name} {dir_name}/{model_name}")

model_paths = natsort.natsorted(glob.glob('datasets/hunyuan_demo/*'))[:1]
with ThreadPoolExecutor(max_workers=32) as executor:
    futures = [executor.submit(single_task, model_name) for idx, model_name in enumerate(model_paths)]