
# https://github.com/orgs/ultralytics/discussions/14830
# 
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import torch
import ultralytics 
model = ultralytics.SAM("sam2_s.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Segment with point prompt
results = model(ultralytics.ASSETS / "bus.jpg", points=[150, 150], labels=[1], device=device)
