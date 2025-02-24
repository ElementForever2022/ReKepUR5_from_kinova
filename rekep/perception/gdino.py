from dds_cloudapi_sdk import Config, Client, DetectionTask, TextPrompt, DetectionModel, DetectionTarget
import os
import cv2

API_TOKEN = "6af95839327bbdd9ad310cafd8f097d6"
MODEL = "GDino1_5_Pro"
DETECTION_TARGETS = ["Mask", "BBox"]

class GroundingDINO:
    def __init__(self):
        config = Config(API_TOKEN)
        self.client = Client(config)

    def detect_objects(self, image_path, input_prompts):
        # print(f"Debug: Input image path: {image_path}")
        # image = cv2.imread(image_path)
        # print(f"Debug: Input image shape: {image.shape}")
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        image_url = self.client.upload_file(image_path)
        
        task = DetectionTask(
            image_url=image_url,
            prompts=[TextPrompt(text=pt) for pt in input_prompts],
            targets=[getattr(DetectionTarget, target) for target in DETECTION_TARGETS],
            model=getattr(DetectionModel, MODEL),
        )
        self.client.run_task(task)
        return task.result

    def rle2rgba(self, rle_mask):
        # Create a dummy task with minimal required arguments
        dummy_task = DetectionTask(
            image_url="dummy",
            prompts=[TextPrompt(text="dummy")],
            targets=[DetectionTarget.Mask],
            model=getattr(DetectionModel, MODEL)
        )
        return dummy_task.rle2rgba(rle_mask)