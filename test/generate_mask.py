

import os
import numpy as np
from PIL import Image
import pdb
# from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Load the image
img_path = '/home/tonyw/VLM/ReKep/data/pen.png'
image = Image.open(img_path)
image = np.array(image.convert("RGB"))

# Initialize the SAM2 predictor
# predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
mask_generator = SAM2AutomaticMaskGenerator.from_pretrained("facebook/sam2-hiera-large")
# Set the image
# predictor.set_image(image)

# Generate masks for the entire image
# masks, scores, _ = predictor.predict(
#     point_coords=None,
#     point_labels=None,
#     multimask_output=True,
# )
masks = mask_generator.generate(image)
print(len(masks))
print(masks[0].keys())


# Automated Mask (Instance Segmentation) Generation with SAM
# To generate masks automatically, use the SamAutomaticMaskGenerator. This utility generates a list of dictionaries describing individual segmentations. Each dict in the result list has the following format:

#     segmentation - [np.ndarray] - the mask with (W, H) shape, and bool type, where W and H are the width and height of the original image, respectively
#     area - [int] - the area of the mask in pixels
#     bbox - [List[int]] - the boundary box detection in xywh format
#     predicted_iou - [float] - the model's own prediction for the quality of the mask
#     point_coords - [List[List[float]]] - the sampled input point that generated this mask
#     stability_score - [float] - an additional measure of mask quality
#     crop_box - List[int] - the crop of the image used to generate this mask in xywh format


pdb.set_trace()
# Create output directory
img_name = os.path.splitext(os.path.basename(img_path))[0]
output_dir = f'/home/tonyw/VLM/ReKep/data/mask/{img_name}_sam2'
os.makedirs(output_dir, exist_ok=True)

# Convert masks to uint8 and stack them
masks_uint8 = np.stack([mask['segmentation'].astype(np.uint8) for mask in masks], axis=-1)

# Save masks as a single .npy file
masks_path = os.path.join(output_dir, f'{img_name}_masks.npy')
np.save(masks_path, masks_uint8)

print(f"Saved {masks_uint8.shape[-1]} masks to {masks_path}")