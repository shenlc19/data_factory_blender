import os, pyexr, glob, json
import numpy as np
from PIL import Image
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

def process_single(save_path):
    img_path = os.path.join(save_path, 'side_view.png')
    depth_path = os.path.join(save_path, 'depth/000_depth.exr')
    depth = pyexr.read(depth_path)
    img = np.array(Image.open(img_path))

    rgb_mask = img.sum(-1) > 10
    valid_depth = (0.1 < depth[..., 0][rgb_mask]) * (depth[..., 0][rgb_mask] < 10)
    ratio = valid_depth.sum() / rgb_mask.sum()

    return save_path, ratio

with open('transparent_id_list_futher_filter.json') as f:
    model_paths = json.load(f)

depth_filtered = []
# model_paths = model_paths[:10]

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process_single, os.path.dirname(model_path)) for model_path in model_paths]
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        model_path, ratio = result
        if ratio > 0.96:
            depth_filtered.append(model_path)

with open('depth_filtered.json', 'w') as f:
    json.dump(depth_filtered, f, indent=4) 
