from concurrent.futures import ThreadPoolExecutor
import os, glob, natsort

NUM_GPUS = 4

def single_task(model_name, gpu_idx):
    model_id = os.path.basename(os.path.dirname(model_name)).split('.')[0]
    if not os.path.exists(os.path.join('datasets/glassverse_v0_120_views_hdri', model_id, '119.png')):
        # print(model_id)
        os.system(f"CUDA_VISIBLE_DEVICES={gpu_idx} python dataset_toolkits/slc_render_pbr.py {model_name}")
    
model_paths = natsort.natsorted(
    glob.glob('datasets/glassverse_v0_filtered/*/lm.blend')
)[2964:]

breakpoint()

with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
    futures = [executor.submit(single_task, model_name, (idx % NUM_GPUS)) for idx, model_name in enumerate(model_paths)]