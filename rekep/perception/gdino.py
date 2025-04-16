# from dds_cloudapi_sdk import Config, Client, DetectionTask, TextPrompt, DetectionModel, DetectionTarget
from dds_cloudapi_sdk import Config, Client
import os
import cv2


# Create a task with proper parameters.
from dds_cloudapi_sdk.tasks.v2_task import V2Task


# API_TOKEN = "6af95839327bbdd9ad310cafd8f097d6"
API_TOKEN = "bd31258f745ca3235c19a68133c6db37"
# 获取TOKEN：
# https://cloud.deepdataspace.com/dashboard/token-key

# MODEL = "GDino1_5_Pro"

# DETECTION_TARGETS = ["Mask", "BBox"]

class GroundingDINO:
    def __init__(self):
        config = Config(API_TOKEN)
        self.client = Client(config)

    # def detect_objects(self, image_path, input_prompts):
    #     # print(f"Debug: Input image path: {image_path}")
    #     # image = cv2.imread(image_path)
    #     # print(f"Debug: Input image shape: {image.shape}")
    #     # cv2.imshow('image', image)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     image_url = self.client.upload_file(image_path)
        
    #     task = DetectionTask(
    #         image_url=image_url,
    #         prompts=[TextPrompt(text=pt) for pt in input_prompts],
    #         targets=[getattr(DetectionTarget, target) for target in DETECTION_TARGETS],
    #         model=getattr(DetectionModel, MODEL),
    #     )
    #     self.client.run_task(task)
    #     return task.result
    
    # 0416 测试v2api
    def detect_objects(self, image_path, input_prompts):
        # print(f"Debug: Input image path: {image_path}")
        # image = cv2.imread(image_path)
        # print(f"Debug: Input image shape: {image.shape}")
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        image_url = self.client.upload_file(image_path)
        
        # 打印image_path
        print(f"Debug: image_path: {image_path}")
        
        # 打印input_prompts
        # print(f"Debug: input_prompts: {input_prompts}")
        
        # 如果 input_prompts 是列表，则将其转换为字符串
        if isinstance(input_prompts, list):
            prompt_text = ".".join(input_prompts)
        else:
            prompt_text = input_prompts
        # v2的调用要求传入的prompt是字符串而不是列表

        # 打印处理后的prompt_text
        # print(f"Debug: prompt_text_after_process: {prompt_text}")

        task = V2Task(api_path="/v2/task/grounding_dino/detection", api_body={
            "model": "GroundingDino-1.6-Pro",
            "image": image_url,
            "prompt": {
                "type":"text",
                "text":prompt_text
            },
            "targets": ["bbox"],
        })

        # 打印image_url
        # print(f"Debug: image_url: {image_url}")

        self.client.run_task(task)
        return task.result
    

    # def __detect_objects_with_photo(self, color_frame, input_prompts):


    # def rle2rgba(self, rle_mask):
    #     # Create a dummy task with minimal required arguments
    #     dummy_task = DetectionTask(
    #         image_url="dummy",
    #         prompts=[TextPrompt(text="dummy")],
    #         targets=[DetectionTarget.Mask],
    #         model=getattr(DetectionModel, MODEL)
    #     )
    #     return dummy_task.rle2rgba(rle_mask)