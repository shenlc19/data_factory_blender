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

model_paths = []
with open('0808_car_remaining.txt') as f:
    for line in f.readlines():
        model_name = line.replace('\n', '')
        model_paths.append(glob.glob(f'datasets/Carverse/sketchfab_0808/*/{model_name}/model.glb')[0])

model_paths = model_paths[4:]
breakpoint()

with ThreadPoolExecutor(max_workers=NUM_GPUS*2) as executor:
    futures = [executor.submit(single_task, model_name, (idx % NUM_GPUS)) for idx, model_name in enumerate(model_paths)]