from concurrent.futures import ThreadPoolExecutor
import os, glob, natsort

NUM_GPUS = 4

def single_task(model_name, gpu_idx):
    os.system(f"CUDA_VISIBLE_DEVICES={gpu_idx} python dataset_toolkits/render_primitives_cam_rotation.py {model_name}")
    
model_paths = natsort.natsorted(
    glob.glob('datasets/primitives_v0_material_replaced/*/lm.blend')
)[4:]

breakpoint()

with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
    futures = [executor.submit(single_task, model_name, (idx % NUM_GPUS)) for idx, model_name in enumerate(model_paths)]