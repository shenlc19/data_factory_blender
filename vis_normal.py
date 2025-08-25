import pyexr
import Imath
import numpy as np
import cv2 # or Pillow
import os, torch, utils3d, json

def visualize_normal_map_exr(normal_map, save_path):

    # Reshape to image dimensions
    R = normal_map[..., 0]
    G = normal_map[..., 1]
    B = normal_map[..., 2]

    # Stack channels to form an RGB image
    # Assuming the normal map components are typically in [-1, 1]
    # and we want to map them to [0, 1] for display (like the reference image)
    # Visualization formula: color = (normal_component + 1) / 2
    # Then convert to 8-bit for display
    normal_image_float = np.stack([(R + 1) / 2, (G + 1) / 2, (B + 1) / 2], axis=-1)

    # Clamp values to [0, 1] to handle potential out-of-range floats
    normal_image_float = np.clip(normal_image_float, 0, 1)

    # Convert to 8-bit integer (0-255) for display
    normal_image_display = (normal_image_float * 255).astype(np.uint8)

    # OpenCV expects BGR by default, so convert if needed for display
    # If your R, G, B correspond to X, Y, Z, you might want to adjust channel order
    # For example, some normal maps might have Y flipped, or Z as up.
    # The common visualization is X=Red, Y=Green, Z=Blue
    normal_image_bgr = cv2.cvtColor(normal_image_display, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, normal_image_bgr)

# Example usage:
# visualize_normal_map_exr("your_normal_map.exr")
    
def world2camera_normal(normal: torch.Tensor, c2w_mat_gl: torch.Tensor) -> torch.Tensor:
    h, w, _ = normal.shape
    normal_cam = normal.reshape(-1, 3) @ c2w_mat_gl[:3, :3]
    return normal_cam.reshape(h, w, 3)

import sys    
result_base_dir = sys.argv[1]

with open(os.path.join(result_base_dir, 'transforms.json')) as f:
    transforms = json.load(f)

save_dir = os.path.join(result_base_dir, 'normal_vis')
os.makedirs(save_dir, exist_ok=True)

for idx, image_meta in enumerate(transforms['frames']):
    fov = image_meta['camera_angle_x']
    intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
    c2w_mat_gl = torch.tensor(image_meta["transform_matrix"])

    normal = torch.tensor(
        pyexr.read(os.path.join(result_base_dir, f"normal/{idx:03d}_normal.exr"))
    )[..., :3]
    normal_cam = world2camera_normal(normal, c2w_mat_gl)

    visualize_normal_map_exr(normal_cam, os.path.join(save_dir, f"{idx:03d}.png"))
    
# === Rewrite the following code === #

# import json
# with open('datasets/car_demo/examples/0bc00c7cd32c4d6da2098fbc2ab1eff0/transforms.json') as f:
#     transforms = json.load(f)

# import torch, utils3d
# image_meta = transforms['frames'][0]
# fov = image_meta['camera_angle_x']
# intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
# c2w_mat_gl = torch.tensor(image_meta["transform_matrix"])
# c2w_mat_cv = c2w_mat_gl.clone()
# c2w_mat_cv[:, 1:3] *= -1
# extrinsics = torch.inverse(c2w_mat_gl)

# def world2camera_normal(normal: torch.Tensor, c2w_mat_gl: torch.Tensor) -> torch.Tensor:
#     h, w, _ = normal.shape
#     normal_cam = normal.reshape(-1, 3) @ c2w_mat_gl[:3, :3]
#     return normal_cam.reshape(h, w, 3)


# normal = torch.tensor(
#     pyexr.read("datasets/car_demo/examples/0bc00c7cd32c4d6da2098fbc2ab1eff0/normal/000_normal.exr")
# )[..., :3]
# normal_cam = world2camera_normal(normal, c2w_mat_gl)


# # Usage
# visualize_normal_map_exr(normal_cam, "example_normal_map.png")