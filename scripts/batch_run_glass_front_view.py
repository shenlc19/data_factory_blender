from concurrent.futures import ThreadPoolExecutor
import os, glob, natsort

NUM_GPUS = 6

def process_single(blend_path, gpu_idx=0):
    model_id = os.path.basename(blend_path).replace(' ', '_')
    os.system(
        f"CUDA_VISIBLE_DEVICES={gpu_idx} /DATA_EDS2/shenlc2403/blender/blender-4.3.2-linux-x64/blender -b -P dataset_toolkits/blender_script/render_front_view_glass.py \
        -- \
        --object \"{blend_path}\" \
        --output_folder datasets/front_view/glass_objects/{model_id}"
    )

blend_files = natsort.natsorted(
    glob.glob("datasets/single_glass_objects/*.blend")
)

with ThreadPoolExecutor(max_workers=NUM_GPUS*2) as executor:
    futures = [executor.submit(process_single, blend_file, (idx % NUM_GPUS)) for idx, blend_file in enumerate(blend_files)]
