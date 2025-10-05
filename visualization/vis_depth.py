import pyexr
import numpy as np
import cv2
import matplotlib.cm as cm

# Read EXR file with pyexr
depth_map = pyexr.open("/DATA_EDS2/shenlc2403/data_factory/data_factory_blender/datasets/glassverse_v0_120_views_hdri/007856/depth/000_depth.exr").get()

# If it's a multi-channel image, take the first channel
if len(depth_map.shape) == 3:
    depth_map = depth_map[:, :, 0]

# Handle inf/nan values
depth_map = np.where(np.isfinite(depth_map), depth_map, 0)

# Normalize to 0-255 range
depth_min = depth_map.min()
depth_max = depth_map.max()
depth_normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

# Save as PNG with same dimensions
cv2.imwrite('depth_map.png', depth_normalized)

# Apply colormap (e.g., viridis)
colormap = cm.get_cmap('Spectral')
colored = colormap(depth_normalized / 255.0)  # Normalize to 0-1 for colormap
colored_bgr = (colored[:, :, :3] * 255).astype(np.uint8)  # Remove alpha, convert to BGR
colored_bgr = cv2.cvtColor(colored_bgr, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

cv2.imwrite('depth_map_colored.png', colored_bgr)