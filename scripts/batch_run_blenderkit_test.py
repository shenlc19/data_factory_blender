from concurrent.futures import ThreadPoolExecutor
import os, glob, natsort

NUM_GPUS = 2

def single_task(model_name, gpu_idx):
    model_id = os.path.basename(model_name).split('.')[0]
    if os.path.exists(os.path.join('datasets/carverse_blenderkit_512sample_1024_12view_even_light', model_id, '011.png')):
        # print(model_id, 'exists, skip.')
        return
    else:
        print(model_id, 'missing')
    os.system(f"CUDA_VISIBLE_DEVICES={gpu_idx} python dataset_toolkits/lh_render_pbr_blenderkit.py {model_name}")
    
# model_paths = natsort.natsorted(
#     glob.glob('datasets/Carverse/sketchfab_new/*/*/*.glb')
# )

blenderkit_ids = []
with open('blenderkit_test_set.txt') as f:
    for line in f.readlines():
        blenderkit_ids.append(line.replace('\n', ''))

model_paths = []
for model_id in blenderkit_ids:
    model_paths.append(os.path.join('datasets/Carverse/blenderkit/3Ddata', model_id + '.blend'))

model_paths = model_paths[1:]
breakpoint()

with ThreadPoolExecutor(max_workers=NUM_GPUS*2) as executor:
    futures = [executor.submit(single_task, model_name, (idx % NUM_GPUS)) for idx, model_name in enumerate(model_paths)]