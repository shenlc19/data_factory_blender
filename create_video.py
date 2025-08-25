import imageio
import numpy as np
import cv2
import os
from pathlib import Path

def create_tilted_brush_transition(rendering, normal, progress, line_angle=45):
    """
    Create a tilted line transition effect between rendering and normal
    
    Args:
        rendering: Rendering image
        normal: Normal image 
        progress: Animation progress (0.0 to 1.0)
        line_angle: Angle of the transition line in degrees
    """
    h, w = rendering.shape[:2]
    
    # Calculate line position based on progress
    # Progress 0-0.5: brush from right to middle (show normal)
    # Progress 0.5-1.0: brush from middle to right (show rendering)
    
    if progress <= 0.5:
        # First half: revealing normal from right to middle
        brush_progress = progress * 2  # 0 to 1
        line_x = w - (w * 0.5 * brush_progress)  # From right to middle
        reveal_normal = True
    else:
        # Second half: revealing rendering from middle to right  
        brush_progress = (progress - 0.5) * 2  # 0 to 1
        line_x = w * 0.5 + (w * 0.5 * brush_progress)  # From middle to right
        reveal_normal = False
    
    # Convert angle to radians
    angle_rad = np.radians(line_angle)
    
    # Create tilted line mask
    mask = np.zeros((h, w), dtype=np.bool_)
    
    for y in range(h):
        for x in range(w):
            # Calculate distance from the tilted line
            # Line equation: (x - line_x) * cos(angle) + y * sin(angle) = 0
            distance = (x - line_x) * np.cos(angle_rad) + y * np.sin(angle_rad)
            
            if reveal_normal:
                # Show normal where distance > 0 (right side of line)
                mask[y, x] = distance > 0
            else:
                # Show rendering where distance < 0 (left side of line)
                mask[y, x] = distance < 0
    
    # Apply the mask
    if reveal_normal:
        # Start with rendering, reveal normal
        result = rendering.copy()
        result[mask] = normal[mask]
    else:
        # Start with normal, reveal rendering
        result = normal.copy()
        result[mask] = rendering[mask]
    
    return result

def load_image_sequence(folder_path, extensions=['.png', '.jpg', '.jpeg']):
    """Load images from a folder in sorted order"""
    folder = Path(folder_path)
    images = []
    
    # Get all image files and sort them
    image_files = []
    for ext in extensions:
        image_files.extend(folder.glob(f'*{ext}'))
        image_files.extend(folder.glob(f'*{ext.upper()}'))
    
    image_files.sort()
    
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    return images

def create_brush_transition_video(renderings_folder, normals_folder, output_path, fps=30, line_angle=45):
    """
    Create a video with tilted brush transition effect
    - 120 total frames
    - Only middle 60 frames (30-89) have transition effect
    - Others show pure rendering
    
    Args:
        renderings_folder: Path to rendering images
        normals_folder: Path to normal images  
        output_path: Output video file path
        fps: Frames per second
        line_angle: Angle of transition line in degrees
    """
    
    # Load image sequences
    print("Loading images...")
    renderings = load_image_sequence(renderings_folder)
    normals = load_image_sequence(normals_folder)
    
    if len(renderings) != len(normals):
        raise ValueError(f"Mismatch: {len(renderings)} renderings vs {len(normals)} normals")
    
    if len(renderings) != 120:
        print(f"Warning: Expected 120 frames, got {len(renderings)} frames")
    
    if len(renderings) == 0:
        raise ValueError("No images found")
    
    # Get dimensions and resize if needed
    h, w = renderings[0].shape[:2]
    
    frames = []
    total_frames = len(renderings)
    
    # Define transition range (middle 60 frames)
    transition_start = 30
    transition_end = 90
    transition_frames = 60
    
    print(f"Creating {total_frames} frames...")
    print(f"Transition effect: frames {transition_start} to {transition_end-1}")
    
    for i, (rendering, normal) in enumerate(zip(renderings, normals)):
        # Resize normal to match rendering if needed
        if normal.shape != rendering.shape:
            normal = cv2.resize(normal, (w, h))
        
        if transition_start <= i < transition_end:
            # Apply transition effect for middle 60 frames
            transition_progress = (i - transition_start) / (transition_frames - 1)
            frame = create_tilted_brush_transition(rendering, normal, transition_progress, line_angle)
        else:
            # Show pure rendering for other frames
            frame = rendering.copy()
        
        frames.append(frame)
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{total_frames} frames")
    
    # Save video
    print(f"Saving video to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps)
    print("Video saved successfully!")

# Example usage
if __name__ == "__main__":
    # Set your folder paths here
    import sys
    renderings_folder = sys.argv[1]  # Replace with your renderings folder path
    normals_folder = os.path.join(renderings_folder, 'normal_vis')      # Replace with your normals folder path
    model_name = os.path.basename(renderings_folder)
    subset_name = os.path.basename(os.path.dirname(renderings_folder))
    output_video = f"./{subset_name}/{model_name}.mp4"         # Output video filename
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    
    create_brush_transition_video(
        renderings_folder=renderings_folder,
        normals_folder=normals_folder, 
        output_path=output_video,
        fps=30,
        line_angle=10  # Adjust angle as needed
    )