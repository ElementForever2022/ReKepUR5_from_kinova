import os
import torch
import numpy as np
import argparse
import pyrealsense2 as rs
import supervision as sv
import cv2

from rekep.keypoint_proposal import KeypointProposer
from rekep.constraint_generation import ConstraintGenerator
from rekep.utils import (
    bcolors,
    get_config,
)
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from hydra import compose, initialize
from hydra.utils import instantiate

#sam的权重文件和配置文件的路径
checkpoint = "//home//ur5//rekep//ReKepUR5_from_kinova//sam2model//sam2.1_hiera_base_plus.pt"
model_cfg = "//home//ur5//rekep//ReKepUR5_from_kinova//sam2model//sam2.1_hiera_b+.yaml"
# checkpoint =  "//home//kinova//Model//sam2.1_hiera_base_plus.pt"
# model_cfg = "//home//kinova//Model//sam2.1_hiera_base_plus.yaml"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# from rekep.perception.realsense import initialize_realsense
from rekep.perception.gdino import GroundingDINO

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


import time

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper


@timer_decorator
class MainVision:
    def __init__(self, visualize=False):
        global_config = get_config(config_path="./configs/config.yaml")
        self.config = global_config['main']
        self.visualize = visualize
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])

        self.keypoint_proposer = KeypointProposer(global_config['keypoint_proposer'])
        self.constraint_generator = ConstraintGenerator(global_config['constraint_generator'])
        # Init with local bbox
        sam_model = build_sam2(model_cfg, checkpoint).to(device)
        self.mask_generator = SAM2ImagePredictor(sam_model)

        self.intrinsics = self.load_camera_intrinsics()

    def load_camera_intrinsics(self):
        # D435i 的默认内参（你可以根据实际情况修改这些值）
        class RS_Intrinsics:
            def __init__(self):
                self.fx = 386.738  # focal length x
                self.fy = 386.738  # focal length y
                self.ppx = 319.5   # principal point x
                self.ppy = 239.5   # principal point y
                
        intrinsics = RS_Intrinsics()
        depth_scale = 0.001  # D435i默认深度比例，1mm

        return intrinsics, depth_scale

    @timer_decorator
    def depth_to_pointcloud(self, depth):
        # TODO: check if this is correct
        intrinsics, depth_scale = self.intrinsics

        height, width = depth.shape
        nx = np.linspace(0, width-1, width)
        ny = np.linspace(0, height-1, height)
        u, v = np.meshgrid(nx, ny)
        x = (u.flatten() - intrinsics.ppx) / intrinsics.fx
        y = (v.flatten() - intrinsics.ppy) / intrinsics.fy

        z = depth.flatten() * depth_scale
        x = np.multiply(x, z)
        y = np.multiply(y, z)

    
        points = np.stack((x, y, z), axis = -1)
        return points    
    
    def get_bboxes(self, rgb, obj_list):
        gdino = GroundingDINO()
        rgb_path = 'data/temp_rgb.png' # save rgb to png at data temperarily for upload
        # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(rgb_path, bgr)
        results = gdino.detect_objects(rgb_path, obj_list)
        return results

    @timer_decorator
    def perform_task(self, instruction,obj_list, data_path, frame_number):
        # BUG: name for  color is not consistent
        color_path = os.path.join(data_path, f'color_{frame_number:06d}.npy')
        depth_path = os.path.join(data_path, f'depth_{frame_number:06d}.npy')

        #png格式的rgb图
        png_path = os.path.join(data_path, f'color_{frame_number:06d}.png')

        if not os.path.exists(color_path) or not os.path.exists(depth_path):
            raise FileNotFoundError(f"Color or depth frame not found for frame {frame_number}")

        rgb = np.load(color_path)
        depth = np.load(depth_path)

        print(f"Debug: Input image shape: {rgb.shape}") # (480, 640, 3)
        print(f"Debug: Input depth shape: {depth.shape}") # (480, 640)  

        # detect objects
        gdino = GroundingDINO()

        #png格式的rgb图
        # rgb_path = '/home/ur5/rekep/Rekep4Real/d435i/camera_shot/color_shot_0.png' # save rgb to png at data temperarily for upload
        rgb_path = png_path
        #使用photo.py生成的npy格式的rgb图转换成的png格式的rgb图

        #通道转换
        # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(rgb_path, bgr)

        if isinstance(obj_list, str):
            obj_list = obj_list.split(',')  # 如果输入是逗号分隔的字符串
    
        results = gdino.detect_objects(rgb_path, obj_list)
        self._show_objects(rgb, results.objects)
        # print(f"Debug: Detected {len(results)} objects")
        boxes = []
        for obj in results.objects:
            print(f"class: {obj.category}, conf: {obj.score:.2f}, bbox: {obj.bbox}")
            boxes.append(obj.bbox)
        print(f"Debug: obj_list: {obj_list}")
        print(f"Debug: Boxes: {boxes}")
        # import pdb; pdb.set_trace()
        # Generate masks
        # masks_dict = self.mask_generator.generate(rgb)
        self.mask_generator.set_image(rgb)
        with torch.no_grad():
            masks, scores, logits = self.mask_generator.predict(box=boxes, multimask_output=False)
        # masks = [m['segmentation'] for m in masks_dict]
        print(f"Debug: Generated {len(masks)} masks")
        print(f"Debug: masks shape: {masks[0].shape}")
        print(f"Debug: Type of masks: {type(masks)}")
        # TODO: Add point cloud data from DepthPro model 

        # Generate point cloud from depth image
        points = self.depth_to_pointcloud(depth)
        print(f"Debug: Generated point cloud with shape: {points.shape}")
        # ====================================
        # = Keypoint Proposal and Constraint Generation
        # ====================================
        keypoints, projected_img = self.keypoint_proposer.get_keypoints(rgb, points, masks)
        print(f'{bcolors.HEADER}Got {len(keypoints)} proposed keypoints{bcolors.ENDC}')
        if self.visualize:
            self._show_image(projected_img,rgb)
        metadata = {'init_keypoint_positions': keypoints, 'num_keypoints': len(keypoints)}
        rekep_program_dir = self.constraint_generator.generate(projected_img, instruction, metadata)
        print(f'{bcolors.HEADER}Constraints generated and saved in {rekep_program_dir}{bcolors.ENDC}')
        
    def _show_objects(self, rgb, results):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb)
        plt.axis('on')
        plt.title('Detected Objects')
        for obj in results:
            plt.text(obj.bbox[0], obj.bbox[1], obj.category, color='red', fontsize=12)
            plt.box(obj.bbox)
        plt.savefig('data/gdino_objects.png', bbox_inches='tight', dpi=300)
        plt.close()
        
    def _show_image(self, idx_img, rgb, masks = None,bboxes = None):
        # Save the annotated image with keypoints
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.imshow(idx_img)
        plt.axis('on')
        plt.title('Annotated Image with Keypoints')
        plt.savefig('data/rekep_with_keypoints.png', bbox_inches='tight', dpi=300)
        plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--instruction', type=str, required=True, help='Instruction for the task')
    parser.add_argument('--obj_list', type=str, required=True, help='String List of objects to detect')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the directory containing color and depth frames')
    parser.add_argument('--frame_number', type=int, required=True, help='Frame number to process')
    parser.add_argument('--visualize', action='store_true', help='Visualize the keypoints on the image')
    args = parser.parse_args()

    main = MainVision(visualize=args.visualize)
    main.perform_task(instruction=args.instruction, obj_list=args.obj_list, data_path=args.data_path, frame_number=args.frame_number)

# python main_vision.py --instruction "help me take that bottle of water" --obj_list 'bottle' --data_path /home/kinova/Rekep4Real/data --frame_number 2