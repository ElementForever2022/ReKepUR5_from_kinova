import os
from PIL import Image
import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load the image
img_path = '/home/tonyw/VLM/ReKep/data/pen.png'
image = Image.open(img_path)
image = np.array(image.convert("RGB"))

# Initialize the SAM2 predictor
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

# Set the image
predictor.set_image(image)

# Generate masks for the entire image
masks, scores, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    multimask_output=True,
)

# Create output directory
img_name = os.path.splitext(os.path.basename(img_path))[0]
output_dir = f'./data/mask/{img_name}_sam2'
os.makedirs(output_dir, exist_ok=True)

# Save masks
for i, (mask, score) in enumerate(zip(masks, scores)):
    mask_binary = (mask > 0.5).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask_binary)
    
    # Save as PNG (you can change this to other formats if needed)
    mask_path = os.path.join(output_dir, f'mask_{i:03d}_score_{score:.3f}.png')
    mask_image.save(mask_path)

print(f"Saved {len(masks)} masks to {output_dir}")