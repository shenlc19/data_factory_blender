import pyexr
import numpy as np
import cv2

def exr_single_channel_to_array(exr_path, channel_name=None, normalize_method='minmax', 
                               gamma_correct=False, clamp_range=None):
    """
    Convert single channel EXR to normalized numpy array for visualization
    
    Args:
        exr_path: Path to the EXR file
        channel_name: Name of channel to extract (None for auto-detection)
        normalize_method: 'minmax', 'absolute', 'percentile', or 'none'
        gamma_correct: Apply gamma correction (default: False)
        clamp_range: Tuple (min, max) to clamp values before normalization
    
    Returns:
        tuple: (normalized_array, original_array, channel_info)
    """
    # Read the EXR file
    exr_data = pyexr.open(exr_path)
    available_channels = exr_data.channels
    
    # Auto-detect channel if not specified
    if channel_name is None:
        if len(available_channels) == 1:
            channel_name = available_channels[0]
        else:
            # Prefer common single-channel names
            priority_channels = ['Y', 'L', 'depth', 'Z', 'alpha', 'A', 'R', 'G', 'B']
            for ch in priority_channels:
                if ch in available_channels:
                    channel_name = ch
                    break
            
            if channel_name is None:
                channel_name = available_channels[0]  # Use first available
    
    if channel_name not in available_channels:
        raise ValueError(f"Channel '{channel_name}' not found. Available: {available_channels}")
    
    # Extract the channel data
    original_data = exr_data.get(channel_name)
    
    # Get statistics
    data_min = np.min(original_data)
    data_max = np.max(original_data)
    data_mean = np.mean(original_data)
    data_std = np.std(original_data)
    
    channel_info = {
        'channel_name': channel_name,
        'shape': original_data.shape,
        'dtype': original_data.dtype,
        'min': data_min,
        'max': data_max,
        'mean': data_mean,
        'std': data_std,
        'available_channels': available_channels
    }
    
    # Apply clamping if specified
    work_data = original_data.copy()
    if clamp_range is not None:
        work_data = np.clip(work_data, clamp_range[0], clamp_range[1])
        channel_info['clamped_min'] = clamp_range[0]
        channel_info['clamped_max'] = clamp_range[1]
    
    # Normalize based on method
    if normalize_method == 'minmax':
        # Standard min-max normalization to [0, 1]
        work_min = np.min(work_data)
        work_max = np.max(work_data)
        if work_max > work_min:
            normalized_data = (work_data - work_min) / (work_max - work_min)
        else:
            normalized_data = np.zeros_like(work_data)
    
    elif normalize_method == 'absolute':
        # Normalize by absolute maximum (preserves sign if negative values)
        abs_max = np.max(np.abs(work_data))
        if abs_max > 0:
            normalized_data = work_data / abs_max
            normalized_data = (normalized_data + 1.0) * 0.5  # Map [-1,1] to [0,1]
        else:
            normalized_data = np.zeros_like(work_data)
    
    elif normalize_method == 'percentile':
        # Use percentile-based normalization (robust to outliers)
        p1 = np.percentile(work_data, 1)
        p99 = np.percentile(work_data, 99)
        if p99 > p1:
            normalized_data = np.clip((work_data - p1) / (p99 - p1), 0, 1)
        else:
            normalized_data = np.zeros_like(work_data)
    
    elif normalize_method == 'none':
        # No normalization, just clamp to [0, 1]
        normalized_data = np.clip(work_data, 0, 1)
    
    else:
        raise ValueError(f"Unknown normalize_method: {normalize_method}")
    
    # Apply gamma correction if requested
    if gamma_correct:
        normalized_data = np.power(normalized_data, 1.0/2.2)
    
    return normalized_data, original_data, channel_info

def exr_single_channel_to_rgb(exr_path, channel_name=None, colormap='gray', **kwargs):
    """
    Convert single channel EXR to RGB array using colormap
    
    Args:
        exr_path: Path to the EXR file
        channel_name: Channel to extract
        colormap: 'gray', 'viridis', 'plasma', 'hot', 'cool', or 'jet'
        **kwargs: Arguments passed to exr_single_channel_to_array()
    
    Returns:
        tuple: (rgb_array, original_data, channel_info)
    """
    normalized_data, original_data, channel_info = exr_single_channel_to_array(
        exr_path, channel_name, **kwargs
    )
    
    # Apply colormap
    if colormap == 'gray':
        # Grayscale - replicate channel 3 times
        rgb_array = np.stack([normalized_data] * 3, axis=-1)
    
    elif colormap == 'viridis':
        # Viridis colormap approximation
        rgb_array = apply_viridis_colormap(normalized_data)
    
    elif colormap == 'plasma':
        # Plasma colormap approximation
        rgb_array = apply_plasma_colormap(normalized_data)
    
    elif colormap == 'hot':
        # Hot colormap (black -> red -> yellow -> white)
        rgb_array = apply_hot_colormap(normalized_data)
    
    elif colormap == 'cool':
        # Cool colormap (cyan -> magenta)
        rgb_array = apply_cool_colormap(normalized_data)
    
    elif colormap == 'jet':
        # Jet colormap (blue -> cyan -> yellow -> red)
        rgb_array = apply_jet_colormap(normalized_data)
    
    else:
        raise ValueError(f"Unknown colormap: {colormap}")
    
    return rgb_array, original_data, channel_info

def exr_single_channel_to_8bit(exr_path, channel_name=None, **kwargs):
    """
    Convert single channel EXR to 8-bit grayscale array
    
    Returns:
        tuple: (uint8_array, original_data, channel_info)
    """
    normalized_data, original_data, channel_info = exr_single_channel_to_array(
        exr_path, channel_name, **kwargs
    )
    
    uint8_array = (normalized_data * 255).astype(np.uint8)
    return uint8_array, original_data, channel_info

# Colormap functions
def apply_viridis_colormap(data):
    """Approximate viridis colormap"""
    r = np.clip(0.267 + 0.975*data - 0.334*data**2, 0, 1)
    g = np.clip(0.004 + 1.404*data - 0.569*data**2, 0, 1)  
    b = np.clip(0.329 + 1.105*data - 1.103*data**2 + 0.674*data**3, 0, 1)
    return np.stack([r, g, b], axis=-1)

def apply_plasma_colormap(data):
    """Approximate plasma colormap"""
    r = np.clip(0.050 + 2.810*data - 2.055*data**2 + 0.195*data**3, 0, 1)
    g = np.clip(0.010 + 0.425*data + 1.851*data**2 - 1.286*data**3, 0, 1)
    b = np.clip(0.527 + 1.579*data - 2.794*data**2 + 1.688*data**3, 0, 1)
    return np.stack([r, g, b], axis=-1)

def apply_hot_colormap(data):
    """Hot colormap: black -> red -> yellow -> white"""
    r = np.clip(3 * data, 0, 1)
    g = np.clip(3 * data - 1, 0, 1)
    b = np.clip(3 * data - 2, 0, 1)
    return np.stack([r, g, b], axis=-1)

def apply_cool_colormap(data):
    """Cool colormap: cyan -> magenta"""
    r = data
    g = 1 - data
    b = np.ones_like(data)
    return np.stack([r, g, b], axis=-1)

def apply_jet_colormap(data):
    """Jet colormap: blue -> cyan -> yellow -> red"""
    r = np.where(data < 0.375, 0,
         np.where(data < 0.625, 4 * data - 1.5,
         np.where(data < 0.875, 1, -4 * data + 4.5)))
    
    g = np.where(data < 0.125, 0,
         np.where(data < 0.375, 4 * data - 0.5,
         np.where(data < 0.625, 1, -4 * data + 2.5)))
    
    b = np.where(data < 0.125, 4 * data + 0.5,
         np.where(data < 0.375, 1,
         np.where(data < 0.625, -4 * data + 2.5, 0)))
    
    return np.stack([np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)], axis=-1)

def print_channel_stats(channel_info):
    """Print detailed statistics about the channel"""
    print(f"Channel: {channel_info['channel_name']}")
    print(f"Shape: {channel_info['shape']}")
    print(f"Data type: {channel_info['dtype']}")
    print(f"Value range: {channel_info['min']:.6f} to {channel_info['max']:.6f}")
    print(f"Mean: {channel_info['mean']:.6f}")
    print(f"Std deviation: {channel_info['std']:.6f}")
    print(f"Available channels: {channel_info['available_channels']}")
    if 'clamped_min' in channel_info:
        print(f"Clamped to: {channel_info['clamped_min']} to {channel_info['clamped_max']}")

# Example usage
if __name__ == "__main__":
    exr_file = "/DATA_EDS2/shenlc2403/data_factory/data_factory_blender/datasets/primitives_v0_env_cam_rotation/01916/Metallic/000_Metallic.exr"
    
    # Method 1: Basic normalization
    normalized, original, info = exr_single_channel_to_array(exr_file)
    print_channel_stats(info)
    print(f"Normalized range: {normalized.min():.3f} to {normalized.max():.3f}")
    
    # Method 2: Get 8-bit version
    uint8_data, _, _ = exr_single_channel_to_8bit(exr_file)
    print(f"8-bit array shape: {uint8_data.shape}, range: {uint8_data.min()} to {uint8_data.max()}")

    cv2.imwrite('test_single_channel_exr.png', uint8_data)
    
    # # Method 3: Apply colormap
    # rgb_viridis, _, _ = exr_single_channel_to_rgb(exr_file, colormap='viridis')
    # print(f"RGB viridis shape: {rgb_viridis.shape}")
    
    # # Method 4: Custom normalization with clamping
    # custom_norm, _, _ = exr_single_channel_to_array(
    #     exr_file, 
    #     normalize_method='percentile',
    #     clamp_range=(-10, 10)  # Clamp extreme values
    # )