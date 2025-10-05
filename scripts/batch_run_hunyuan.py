from concurrent.futures import ThreadPoolExecutor
import os, glob, natsort

NUM_GPUS = 4

def single_task(model_name, gpu_idx):
    model_id = os.path.basename(model_name).split('.')[0]

    os.system(f"CUDA_VISIBLE_DEVICES={gpu_idx} python dataset_toolkits/render_car_pbr.py \"{model_name}\"")
    
# model_paths = natsort.natsorted(
#     glob.glob('datasets/Carverse/sketchfab_new/*/*/*.glb')
# )

model_paths = glob.glob('datasets/hunyuan_models/*')

with ThreadPoolExecutor(max_workers=NUM_GPUS*2) as executor:
    futures = [executor.submit(single_task, model_name, (idx % NUM_GPUS)) for idx, model_name in enumerate(model_paths)]