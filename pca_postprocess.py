"""
Created on Sun Jul 7 11:50:22 2024
@author: ZHIYANG
"""
import open3d as o3d
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os, glob, natsort, json, shutil
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import pyexr
# Step 1: 加载点云数据

BLENDER_INSTALLATION_PATH = '/baai-cwm-vepfs/cwm/hong.li/blender'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-4.4.3-linux-x64/blender'

x_main, y_main, z_main = list(np.eye(3))

def calculate_pca_direction(mesh_path):
    point_cloud = o3d.io.read_triangle_mesh(mesh_path)
    # Step 2: 计算点云的中心
    points = np.asarray(point_cloud.vertices)
    center = np.mean(points, axis=0)
    points -= center
    # Step 3: 计算协方差矩阵
    covariance_matrix = np.dot(points.T, points) / points.shape[0]
    # Step 4: 计算特征值和特征向量
    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
    # Step 5: 创建主方向的点云
    main_direction = eigen_vectors[:, -1] # 最大特征值对应的特征向量（主方向）

    axis = np.eye(3).tolist()[np.argmax(np.abs(main_direction))]

    return axis

def render_front_view(model_path, extra_rot, gpu_idx=0):
    model_id = os.path.basename(model_path).split('.')[0]
    os.system(f'CUDA_VISIBLE_DEVICES={gpu_idx} {BLENDER_PATH} -b -P dataset_toolkits/blender_script/render_front_view.py -- \
        --object {model_path} \
        --output_folder datasets/front_view/{model_id} \
        --extra_rot {extra_rot}')
    
def render_side_view(model_path, extra_rot, gpu_idx=0):
    model_id = os.path.basename(model_path).split('.')[0]
    os.system(f'CUDA_VISIBLE_DEVICES={gpu_idx} {BLENDER_PATH} -b -P dataset_toolkits/blender_script/render_side_view.py -- \
        --object {model_path} \
        --output_folder datasets/side_view/{model_id} \
        --extra_rot {extra_rot}')

def load_and_resize(model_path, save_path, center, scale):
    x, y, z = center
    os.system(f'{BLENDER_PATH} -b -P load_and_resize.py \
            -- \
            --object {model_path} \
            --save_path {save_path} \
            --center {x} {y} {z} \
            --scale {scale}')
    
def qwen_inference(model, processor, messages):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text

def load_camera_parameters(filepath="camera_params.json"):
    """
    Load camera parameters from JSON file
    """
    with open(filepath, 'r') as f:
        params = json.load(f)
    return params

def depth_to_3d(depth_map, camera_params, scale_factor=1.0, xs=None, ys=None):
    """
    Project depth map to 3D points using camera parameters
    
    Args:
        depth_map: 2D numpy array with depth values
        camera_params: Dictionary with camera parameters
        scale_factor: Scale factor for depth values (if needed)
    
    Returns:
        points_3d: Nx3 array of 3D points
        colors: Nx3 array of colors (if you have RGB image)
    """
    # Extract camera parameters
    K = np.array(camera_params["intrinsic_matrix"])
    R = np.array(camera_params["rotation_matrix"])
    t = np.array(camera_params["translation_vector"])
    
    height, width = depth_map.shape
    
    # Create pixel coordinate grids
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.flatten()
    v = v.flatten()

    depth = depth_map.flatten() * scale_factor

    # Filter out invalid depths (assuming 0 or very small values are invalid)
    # valid_mask = (depth > 1e-6) * (depth < 100)
    # u = u[valid_mask]
    # v = v[valid_mask]
    # depth = depth[valid_mask]
    
    # Convert to homogeneous coordinates
    pixel_coords = np.stack([u, v, np.ones_like(u)], axis=0)
    
    # Back-project to 3D camera coordinates
    K_inv = np.linalg.inv(K)
    cam_coords = K_inv @ pixel_coords
    cam_coords = cam_coords * depth.reshape(1, -1)
    
    # Add homogeneous coordinate
    cam_coords_homo = np.vstack([cam_coords, np.ones((1, cam_coords.shape[1]))])
    
    # Transform to world coordinates
    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R.T  # Inverse rotation
    T[:3, 3] = -R.T @ t  # Inverse translation
    
    world_coords = T @ cam_coords_homo
    points_3d = world_coords[:3, :].T
    
    return points_3d.reshape(height, width, 3) #, valid_mask.reshape(height, width)

def process_single_mesh(mesh_path, model, processor, gpu_idx=0):
    # get main axis
    main_axis = calculate_pca_direction(mesh_path)

    # if facing x, rotate 90 degrees
    rot_angle = 0
    if main_axis == [1, 0, 0]:
        rot_angle = 90

    # render image
    render_front_view(mesh_path, rot_angle)

    # let qwen determine whether the rotated model is facing -y or +y
    model_id = os.path.basename(mesh_path).split('.')[0]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"datasets/front_view/{model_id}/front_view.png",
                },
                {"type": "text", "text": "Determine whether this is the front or back of a car. The response should either be 'front' or 'rear'."},
            ],
        }
    ]

    # Preparation for inference
    output_text = qwen_inference(model, processor, messages)
    print(output_text)

    shutil.rmtree(f"datasets/front_view/{model_id}/")
    
    # if facing -y, rotate 180 degrees
    if output_text[0] == 'rear':
        rot_angle += 180
        
    # align front wheel axis 
    render_side_view(mesh_path, rot_angle)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"datasets/side_view/{model_id}/side_view.png",
                },
                {"type": "text", "text": "Determine the pixel coordinate of the **center** of the front wheel of the car. The response should two integers separated by a comma 'x,y', representing the pixel coordinate."},
            ],
        }
    ]

    depth_map = pyexr.read(f'datasets/side_view/{model_id}/depth/000_depth.exr')
    camera_params = load_camera_parameters(f'datasets/side_view/{model_id}/camera_params.json')
    
    # Preparation for inference
    output_text = qwen_inference(model, processor, messages)
    print(output_text)

    x, y = [int(num) for num in output_text[0].split(',')[:2]]
    x = np.array(x)
    y = np.array(y)
    
    points3d = depth_to_3d(depth_map[..., 0], camera_params) #, xs=x, ys=y)
    front_wheel_center = points3d[y, x]
    print(front_wheel_center)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"datasets/side_view/{model_id}/side_view.png",
                },
                {"type": "text", "text": "Estimate the length of the car in this image in meters. It doesn't have to be very precise but the result should be reasonable. Return a single **float type number**."},
            ],
        }
    ]

    # Preparation for inference
    output_text = qwen_inference(model, processor, messages)
    print(output_text)

    scale = float(output_text[0])

    load_and_resize(
        f'datasets/side_view/{model_id}/mesh.ply',
        f'ori_normalized/{model_id}.blend',
        front_wheel_center,
        scale
    )

if __name__ == '__main__':
    ckpt_path = '/baai-cwm-vepfs/cwm/licheng.shen/llm/Qwen2.5-VL-7B-Instruct'

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ckpt_path, torch_dtype="auto", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(ckpt_path)

    import sys
    mesh_path = sys.argv[1] # main axis: [0, 1.0, 0]，在blender中看到的：正对的是(1, 0, 0)
    # mesh_path = 'cars_results/0a20d980aaad45d284a6f267d2546df2.obj' # main axis: [0, 0, 1]，在blender中看到的：正对的是(0, -1, 0)
    process_single_mesh(mesh_path, model, processor)
    
    pass