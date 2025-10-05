import os
import json
import numpy as np  
import cv2  
import pyexr
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
from utils import sphere_hammersley_sequence
import time

BLENDER_LINK = 'https://download.blender.org/release/Blender4.3/blender-4.3.2-linux-x64.tar.xz'
# BLENDER_INSTALLATION_PATH = '/baai-cwm-1/baai_cwm_ml/algorithm/hong.li/tools'
BLENDER_INSTALLATION_PATH = '/DATA_EDS2/shenlc2403/blender'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-4.3.2-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        # os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-4.3.2-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')
        print('Blender installed', flush=True)
    
    # 设置环境变量
    os.system('export EGL_DRIVER=nvidia')
    os.system('export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d')


def _render(file_path, sha256, output_dir, num_views, normal_map=False):
    output_folder = os.path.join(output_dir, sha256)
    
    # # Build camera {yaw, pitch, radius, fov}
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    # yaws = []
    # pitchs = []
    # offset = (np.random.rand(), np.random.rand())
    # for i in range(num_views):
    #     y, p = sphere_hammersley_sequence(i, num_views, offset)
    #     yaws.append(y)
    #     pitchs.append(p)
    # # radius = [1.2] * num_views
    # # fov = [60 / 180 * np.pi] * num_views

    # optimized cameras
    # Sample yaw uniformly from 0 to 360 degrees
    yaws = np.random.uniform(0, 360, num_views) / 180 * np.pi
    min_pitch, max_pitch = 0, 30
    # Sample pitch uniformly from min_pitch to max_pitch degrees
    pitches = np.random.uniform(min_pitch, max_pitch, num_views) / 180 * np.pi

    yaw_noise_std = 0.05
    pitch_noise_std = 0.05
    yaw_noise = np.random.normal(0, yaw_noise_std, num_views)
    pitch_noise = np.random.normal(0, pitch_noise_std, num_views)
    
    # Add noise to angles
    perturbed_yaws = yaws + yaw_noise
    perturbed_pitches = pitches + pitch_noise
    
    # Handle yaw wraparound (keep in [0, 360) range)
    perturbed_yaws = perturbed_yaws % 360
    
    # Clamp pitch to valid range
    perturbed_pitches = np.clip(perturbed_pitches, min_pitch, max_pitch)

    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(perturbed_yaws, perturbed_pitches, radius, fov)]
    
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render_pbr.py'),
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--resolution', '1024',
        '--output_folder', output_folder,
        '--engine', 'CYCLES',
        # '--engine', 'BLENDER_EEVEE_NEXT',
        '--save_mesh',
        '--save_normal',
        '--save_depth',
        # '--save_albedo',
        '--save_pbr'
    ]
    if file_path.endswith('.blend'):
        args.insert(1, file_path)
    
    # call(args, stdout=DEVNULL, stderr=DEVNULL)
    call(args)
        

    if normal_map:
        args = [
            BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render_pbr.py'),
            '--',
            '--views', json.dumps(views),
            '--object', os.path.expanduser(file_path),
            '--resolution', '512',
            '--output_folder', output_folder,
            '--engine', 'CYCLES',
            '--save_low_normal'
        ]
        if file_path.endswith('.blend'):
            args.insert(1, file_path)
        # call(args, stdout=DEVNULL, stderr=DEVNULL)
        call(args)
    
    # if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        # return {'sha256': sha256, 'rendered': True}

def results_to_npz(output_folder):

    # 定义图像文件夹路径  
    image_folder = output_folder  # 替换为你的图像文件夹路径  

    base_color_list = []    
    depth_list = []
    metallic_list = []
    normal_list = []
    roughness_list = []

    # 遍历 150 个视图  
    for i in range(150):  
        # 构建文件名  
        base_color_path = os.path.join(image_folder, f'{i:03d}_Base Color.png')  
        depth_path = os.path.join(image_folder, f'{i:03d}_depth.exr')  
        metallic_path = os.path.join(image_folder, f'{i:03d}_Metallic.png')  
        normal_path = os.path.join(image_folder, f'{i:03d}_normal.png')  
        roughness_path = os.path.join(image_folder, f'{i:03d}_Roughness.png')  

        # 读取图像  
        # import ipdb; ipdb.set_trace()
        base_color = cv2.imread(base_color_path).astype(np.uint8)
        # depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # EXR 文件  
        depth = pyexr.open(depth_path).get().astype(np.uint16)
        metallic = cv2.imread(metallic_path).astype(np.uint8)
        normal = cv2.imread(normal_path).astype(np.uint8)
        roughness = cv2.imread(roughness_path).astype(np.uint8)

        #每个属性单独组成为一个 npz 文件
        base_color_list.append(base_color)
        depth_list.append(depth)
        metallic_list.append(metallic)
        normal_list.append(normal)
        roughness_list.append(roughness)

    base_color_list = np.array(base_color_list)
    depth_list = np.array(depth_list)
    metallic_list = np.array(metallic_list)
    normal_list = np.array(normal_list)
    roughness_list = np.array(roughness_list)

    np.savez(f'{output_folder}/base_color.npz', base_color_list)
    np.savez(f'{output_folder}/depth.npz', depth_list)
    np.savez(f'{output_folder}/metallic.npz', metallic_list)
    np.savez(f'{output_folder}/normal.npz', normal_list)
    np.savez(f'{output_folder}/roughness.npz', roughness_list)


if __name__ == '__main__':
    # dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--output_dir', type=str, required=False,
    #                     default= "/baai-cwm-1/baai_cwm_ml/algorithm/hong.li/code/3dgen/TRELLIS/datasets/tmp",
    #                     help='Directory to save the metadata')
    # parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
    #                     help='Filter objects with aesthetic score lower than this value')
    # parser.add_argument('--instances', type=str, default=None,
    #                     help='Instances to process')
    # parser.add_argument('--num_views', type=int, default=2,
    #                     help='Number of views to render')
    # dataset_utils.add_args(parser)
    # parser.add_argument('--rank', type=int, default=0)
    # parser.add_argument('--world_size', type=int, default=1)
    # parser.add_argument('--max_workers', type=int, default=8)
    # opt = parser.parse_args(sys.argv[2:])
    # opt = edict(vars(opt))

    # os.makedirs(os.path.join(opt.output_dir, 'renders'), exist_ok=True)
    
    # install blender
    print('Checking blender...', flush=True)
    _install_blender()

    # # get file list
    # if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
    #     raise ValueError('metadata.csv not found')
    # metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    # if opt.instances is None:
    #     metadata = metadata[metadata['local_path'].notna()]
    #     if opt.filter_low_aesthetic_score is not None:
    #         metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
    #     if 'rendered' in metadata.columns:
    #         metadata = metadata[metadata['rendered'] == False]
    # else:
    #     if os.path.exists(opt.instances):
    #         with open(opt.instances, 'r') as f:
    #             instances = f.read().splitlines()
    #     else:
    #         instances = opt.instances.split(',')
    #     metadata = metadata[metadata['sha256'].isin(instances)]

    # start = len(metadata) * opt.rank // opt.world_size
    # end = len(metadata) * (opt.rank + 1) // opt.world_size
    # metadata = metadata[start:end]
    # records = []

    # # filter out objects that are already processed
    # for sha256 in copy.copy(metadata['sha256'].values):
    #     if os.path.exists(os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json')):
    #         records.append({'sha256': sha256, 'rendered': True})
    #         metadata = metadata[metadata['sha256'] != sha256]
                
    # print(f'Processing {len(metadata)} objects...')

    # process objects
    # func = partial(_render, output_dir=opt.output_dir, num_views=opt.num_views)
    # rendered = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Rendering objects')
    # rendered = pd.concat([rendered, pd.DataFrame.from_records(records)])
    # rendered.to_csv(os.path.join(opt.output_dir, f'rendered_{opt.rank}.csv'), index=False)

    # file_path = "/baai-cwm-1/baai_cwm_ml/algorithm/hong.li/code/3dgen/TRELLIS/datasets/ObjaverseXL_sketchfab/raw/hf-objaverse-v1/glbs/000-034/1bb177d4e6f6470ba167ef5e4d8e2596.glb"
    file_path = sys.argv[1]
    # file_path = "/baai-cwm-1/baai_cwm_ml/algorithm/hong.li/code/3dgen/TRELLIS/datasets/ObjaverseXL_sketchfab/raw/glbs/000-000/00a1a602456f4eb188b522d7ef19e81b.glb"
    # file_path = "/baai-cwm-1/baai_cwm_ml/algorithm/hong.li/code/3dgen/blender-render-toolbox-1209/assets/glbs/1a57a0d6609145b486ed5b1d3e9ec7fb.glb"
    # file_path = "/baai-cwm-1/baai_cwm_ml/algorithm/hong.li/code/3dgen/blender-render-toolbox-1209/assets/glbs/1a57a0d6609145b486ed5b1d3e9ec7fb.glb"
    # file_path = "/baai-cwm-1/baai_cwm_ml/algorithm/hong.li/code/3dgen/TRELLIS/datasets/trash/assets/5a2d6e397a8945eebf02064c63f88866.glb"
    sha256 = os.path.basename((file_path)).split('.')[0]

    start_time = time.time()

    _render(file_path=file_path, 
            sha256 = sha256, 
            # output_dir="datasets/carverse_blenderkit_60view_even_light",
            output_dir="datasets/carverse_d760_batch2",
            num_views=12,
            normal_map=False
            )

    end_time = time.time()

    print(f'Time taken: {end_time - start_time} seconds')
    # results_to_npz("datasets/tmp/renders/1bb177d4e6f6470ba167ef5e4d8e2596")
    # results_to_npz("datasets/tmp/renders/00a1a602456f4eb188b522d7ef19e81b")