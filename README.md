# 数据生成

环境准备：
```bash
conda activate blenderproc2
```

环境内装blenderproc

## 生成场景

运行如下代码
```bash
blenderproc run scene_creator/better_camera_trajectory.py  \
              /DATA_EDS2/shenlc2403/data_factory/BOP \
              lm \
              /DATA_EDS2/shenlc2403/data_factory/BlenderProc/resources/cctextures \
              datasets/primitives_v0/000000
```
其中的最后一个参数规定了输出的路径，可以是datasets/primitives_v0/000000，或其他自定义的路径

生成完的场景路径下有如下文件：
```bash
- lm.blend
- transforms.json
```

lm.blend为场景blender文件，transforms.json规定了渲染的相机位姿

## PBR渲染

渲染代码：
```bash
python dataset_toolkits/slc_render_pbr.py {path_to_blend}
```

path_to_blend填入“生成场景”步骤中生成的.blend文件
```bash
python dataset_toolkits/slc_render_pbr.py /DATA_EDS2/shenlc2403/data_factory/data_factory_blender/datasets/primitives_v0/000000/lm.blend
```

渲染后的结果会保存在
```bash
datasets/env_rotation/000000
```

这里000000即场景名

## 生成场景多样化修改指南

把objects_to_sample换成primitives或Objaverse物体
```python
# Sample objects on the given surface
placed_objects = bproc.object.sample_poses_on_surface(objects_to_sample=sampled_bop_objs + distractor_bop_objs,
                                         surface=room_planes[0],
                                         sample_pose_func=sample_initial_pose,
                                         min_distance=0.01,
                                         max_distance=0.2)
```

objects_to_sample的数据类型应当是List[MeshObject]

## 汽车数据渲染指南

汽车数据渲染时使用随机采样的相机姿态，确保完全覆盖车身全部视角

运行方法：
```bash
python dataset_toolkits/lh_render_pbr.py {path_to_model}
```

修改函数_render调用参数中的output_dir来指定保存位置：
```python
_render(file_path=file_path, 
            sha256 = sha256, 
            # output_dir="datasets/carverse_blenderkit_60view_even_light",
            # output_dir="datasets/carverse_sketchfab_new_others_60view_even_light",
            # output_dir="datasets/carverse_sketchfab_new_KOE_60view_even_light",
            output_dir="datasets/debug",
            num_views=60,
            normal_map=False
            )
```

一般保存在datasets/xxx下
