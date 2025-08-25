import imageio
import os
import sys
import glob

extension = '.png'
# Directory containing images
image_files = glob.glob(os.path.join(sys.argv[1], '*'+extension))  # Add your image filenames here
image_files.sort(key=lambda x: int(os.path.basename(x).split(extension)[0].split('_')[-1]))

# Output video file
save_name = sys.argv[2]
video_filename = f"{save_name}.mp4"

# Get list of images
images = []
for file_name in image_files:
    if file_name.endswith(extension):  # or .jpg, .jpeg, etc.
        file_path = file_name
        images.append(imageio.imread(file_path))

# Define the FPS (frames per second)
fps = 30

os.makedirs(os.path.dirname(video_filename), exist_ok=True)
# Write images to a video file
imageio.mimsave(video_filename, images, fps=fps)