"""
Created on Sun Jul 7 11:50:22 2024
@author: ZHIYANG
"""
import open3d as o3d
import numpy as np
# Step 1: 加载点云数据

x_main, y_main, z_main = list(np.eye(3))

def process_single(mesh_path):
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

    axis = np.eye(3)[np.argmax(np.abs(main_direction))]

    breakpoint()

mesh_paths = ''
