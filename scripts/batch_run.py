from concurrent.futures import ThreadPoolExecutor
import os, glob, natsort

NUM_GPUS = 4

def single_task(model_name, gpu_idx):
    os.system(f"CUDA_VISIBLE_DEVICES={gpu_idx} python dataset_toolkits/render_car_pbr.py {model_name}")
    
# model_paths = natsort.natsorted(
#     glob.glob('/DATA_EDS2/shenlc2403/data_factory/data_factory_blender/datasets/primitives_v0/*/lm.blend')
# )
    
model_paths = natsort.natsorted(
    glob.glob('datasets/hunyuan/*/*.obj')
)[1:]

breakpoint()

with ThreadPoolExecutor(max_workers=NUM_GPUS*2) as executor:
    futures = [executor.submit(single_task, model_name, (idx % NUM_GPUS)) for idx, model_name in enumerate(model_paths)]