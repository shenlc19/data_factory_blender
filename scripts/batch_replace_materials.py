from concurrent.futures import ThreadPoolExecutor
import os, glob, natsort

def single_task(model_name):
    os.system(f"blenderproc run replace_materials.py {model_name}")

model_paths = natsort.natsorted(glob.glob('datasets/primitives_v0/*/lm.blend'))[4:]

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(single_task, model_name) for idx, model_name in enumerate(model_paths)]