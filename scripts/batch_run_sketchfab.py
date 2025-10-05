from concurrent.futures import ThreadPoolExecutor
import os, glob, natsort

NUM_GPUS = 2

def single_task(model_name, gpu_idx):
    model_id = os.path.basename(model_name).split('.')[0]
    save_dir = os.path.join('datasets/carverse_sketchfab_512sample_1024_12view_even_light', model_id)
    if os.path.exists(os.path.join(save_dir, '011.png')):
        # print(model_id, 'exists, skip.')
        return
    else:
        print(model_name, 'missing')
    os.system(f"CUDA_VISIBLE_DEVICES={gpu_idx} python dataset_toolkits/lh_render_pbr_sketchfab.py {model_name}")
    
# model_paths = natsort.natsorted(
#     glob.glob('datasets/Carverse/sketchfab_new/*/*/*.glb')
# )

import json
with open('0808.json') as f:
    blenderkit_ids = json.load(f)["complete_wheel_models"]

model_paths = []
for model_id in blenderkit_ids:
    model_paths.append(os.path.join('datasets/Carverse/sketchfab', model_id + '.glb'))
    # model_paths += glob.glob(os.path.join(f'datasets/Carverse/sketchfab_0808/*/{model_id}/*.glb'))

model_paths = natsort.natsorted(
    glob.glob('datasets/Carverse/sketchfab_new/*/*.glb') + \
    glob.glob('datasets/Carverse/sketchfab_new/*/*/*.glb')
)[1:]

breakpoint()

with ThreadPoolExecutor(max_workers=NUM_GPUS*2) as executor:
    futures = [executor.submit(single_task, model_name, (idx % NUM_GPUS)) for idx, model_name in enumerate(model_paths)]