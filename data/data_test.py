import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_data(data_folder):
    """
    Visualize the data in the folder.
    """
    # get all the images in the folder
    images = [f for f in os.listdir(data_folder) if f.endswith('.png')]
    # visualize the images
    for image in images:
        img = cv2.imread(os.path.join(data_folder, image))
        print(f"Image {image} shape: {img.shape}")

if __name__ == "__main__":
    data_folder = "/home/tonyw/VLM/ReKep/data/"
    visualize_data(data_folder)

