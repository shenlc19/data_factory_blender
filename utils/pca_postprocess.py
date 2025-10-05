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
# Step 1: 加载点云数据

BLENDER_INSTALLATION_PATH = '/DATA_EDS2/shenlc2403/blender'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-4.3.2-linux-x64/blender'

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

def render_front_view(model_path, gpu_idx=0):
    model_id = os.path.basename(model_path).split('.')[0]
    os.system(f'CUDA_VISIBLE_DEVICES={gpu_idx} {BLENDER_PATH} -b -P dataset_toolkits/blender_script/render_front_view.py -- \
        --object {model_path} \
        --output_folder datasets/front_view/{model_id}')

def judge_front_or_rear():
    pass

def process_single_mesh(mesh_path, model, processor, gpu_idx=0):
    # get main axis
    main_axis = calculate_pca_direction(mesh_path)
    breakpoint()

    # if facing x, rotate 90 degrees
    rot_angle = 0
    if main_axis == [1, 0, 0]:
        rot_angle = 90
        print(mesh_path)
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        transform_matrix = np.array([[0, -1, 0],
                        [1, 0, 0],
                        [0, 0, 1]])
        full_transform = np.eye(4)
        full_transform[:3, :3] = transform_matrix
        mesh.transform(full_transform)
        o3d.io.write_triangle_mesh(mesh_path, mesh)

    breakpoint()

    # render image
    render_front_view(mesh_path)

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
                {"type": "text", "text": "Determine whether this is the front or back of a car."},
            ],
        }
    ]

    # Preparation for inference
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
    print(output_text)

    # if facing -y, rotate 180 degrees
    pass

if __name__ == '__main__':
    # ckpt_path = '/DATA_EDS2/shenlc2403/llm/models/Qwen2.5-VL-7B-Instruct'

    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     ckpt_path,
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # processor = AutoProcessor.from_pretrained(ckpt_path)

    mesh_path = 'cars_results/0a6acbe0724b4fd0901b5a42c19c8952.obj' # main axis: [0, 1.0, 0]，在blender中看到的：正对的是(1, 0, 0)
    # mesh_path = 'cars_results/0a20d980aaad45d284a6f267d2546df2.obj' # main axis: [0, 0, 1]，在blender中看到的：正对的是(0, -1, 0)
    process_single_mesh(mesh_path, model=None, processor=None)
    
    pass