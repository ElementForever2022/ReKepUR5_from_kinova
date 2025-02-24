import numpy as np
import cv2
import os
import glob

def display_npy_videos(data_path):
    # Check if the directory exists
    if not os.path.exists(data_path):
        print(f"Directory {data_path} does not exist.")
        return

    # Get sorted lists of color and depth files
    color_files = sorted(glob.glob(os.path.join(data_path, 'color_*.npy')))
    depth_files = sorted(glob.glob(os.path.join(data_path, 'depth_*.npy')))

    # Check if files exist
    if len(color_files) == 0 or len(depth_files) == 0:
        print(f"No .npy files found in {data_path}.")
        return
    output_dir = os.path.join(os.path.dirname(data_path), 'rgbd_frames')
    color_output_dir = os.path.join(output_dir, 'color')
    depth_output_dir = os.path.join(output_dir, 'depth')
    
    os.makedirs(color_output_dir, exist_ok=True)
    os.makedirs(depth_output_dir, exist_ok=True)

    # Loop through the frames
    for i, (color_file, depth_file) in enumerate(zip(color_files, depth_files)):
        # Load the color and depth images
        color_image = np.load(color_file)
        depth_image = np.load(depth_file)

        # Normalize depth for better visualization (optional)
        depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_normalized = np.uint8(depth_image_normalized)
        depth_colormap = cv2.applyColorMap(depth_image_normalized, cv2.COLORMAP_JET)

        # Save the color and depth images in their respective subfolders
        cv2.imwrite(os.path.join(color_output_dir, f'rgb_frame_{i:04d}.png'), color_image)
        cv2.imwrite(os.path.join(depth_output_dir, f'depth_frame_{i:04d}.png'), depth_colormap)

        print(f'Saved frame {i}')

    print(f"Color frames saved in {color_output_dir}")
    print(f"Depth frames saved in {depth_output_dir}")

data_path = '/home/tonyw/VLM/ReKep/data/rgbd/825312071880'
display_npy_videos(data_path)
