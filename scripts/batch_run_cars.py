from concurrent.futures import ThreadPoolExecutor
import os, glob, natsort

NUM_GPUS = 2

def single_task(model_name, gpu_idx):
    model_id = os.path.basename(model_name).split('.')[0]
    if os.path.exists(os.path.join('datasets/carverse_texverse_4k', model_id, '011.png')):
        # print(model_id, 'exists, skip.')
        return
    else:
        print(model_id, 'missing')
    os.system(f"CUDA_VISIBLE_DEVICES={gpu_idx} python dataset_toolkits/lh_render_pbr.py {model_name}")
    
# model_paths = natsort.natsorted(
#     glob.glob('datasets/Carverse/sketchfab_new/*/*/*.glb')
# )

# import json
# with open('wheel_check_results_blender.json') as f:
#     blenderkit_ids = json.load(f)["complete_wheel_models"]

# model_paths = []
# for model_id in blenderkit_ids:
#     model_paths.append(os.path.join('datasets/Carverse/blenderkit/3Ddata', model_id + '.blend'))

model_paths = natsort.natsorted(glob.glob('datasets/texverse_4k_cars/*'))
model_paths = model_paths[1:]
breakpoint()

with ThreadPoolExecutor(max_workers=NUM_GPUS*2) as executor:
    futures = [executor.submit(single_task, model_name, (idx % NUM_GPUS)) for idx, model_name in enumerate(model_paths)]