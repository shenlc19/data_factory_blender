from concurrent.futures import ThreadPoolExecutor
import os, glob, natsort

NUM_GPUS=6

def single_task(model_path, subset, gpu_idx=0):
    model_id = os.path.basename(model_path).split('.')[0]
    os.system(f'CUDA_VISIBLE_DEVICES={gpu_idx} /DATA_EDS2/shenlc2403/blender/blender-4.3.2-linux-x64/blender -b -P dataset_toolkits/blender_script/render_front_view.py -- \
        --object {model_path} \
        --output_folder datasets/front_view/{subset}/{model_id}')
    
model_paths = natsort.natsorted(
    glob.glob('datasets/Carverse/blenderkit/3Ddata/*')
)

subset = 'blenderkit'

breakpoint()

with ThreadPoolExecutor(max_workers=NUM_GPUS*2) as executor:
    futures = [executor.submit(single_task, model_name, subset, (idx % NUM_GPUS)) for idx, model_name in enumerate(model_paths)]