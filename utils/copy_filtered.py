import json, os
from tqdm import tqdm

with open('depth_filtered.json') as f:
    model_paths = json.load(f)

for file in tqdm(model_paths):
    src_dir = os.path.join('/DATA_EDS2/shenlc2403/data_factory/data_factory_blender/datasets/single_glass_objects',
                            os.path.basename(file))
    tgt_dir = '/DATA_EDS2/shenlc2403/data_factory/data_factory_blender/datasets/single_glass_objects_depth_filtered'
    os.system(f"cp {src_dir} {tgt_dir}")