"""
calculate the intrinsics of the camera from multiple chessboard images
"""

# import necessary packages
import numpy as np # for array operations
from realsense_camera import RealsenseCamera # to use cameras
from camera_manager import CameraManager # to use cameras
from debug_decorators import print_debug,debug_decorator # for debugging messages
import cv2 # for image processing
import pathlib # for path operations
import datetime # for time operations
import os # for os operations

class CalculateIntrinsics:
    def __init__(self, pc_id:int, camera_position:str='global', use_cache:bool=False) -> None:
        """
        initialize the class

        inputs:
            - camera_manager: the camera manager
            - pc_id: the id of the pc
            - camera_position: the position of the camera['global', 'wrist']
        """
        self.pc_id = pc_id # initialize the pc id
        self.camera_position = camera_position # initialize the camera position
        self.camera_manager = CameraManager(pc_id=pc_id) # initialize the camera manager

        # resolution of the camera
        self.height = self.camera_manager.height
        self.width = self.camera_manager.width

    def shoot_images(self, shoot_key:str='s', exit_key:str='q', empty_cache_key:str='r', save_dir:str='./intrinsics_images'):
        """
        shoot images from the camera
        """
        # get current path
        current_path = pathlib.Path(__file__).parent.resolve()
        save_path = pathlib.Path(current_path, save_dir)

        # check if the save directory exists
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
            print_debug(f"save path {save_path} does not exist, and has been created", color_name='COLOR_GREEN')
        else:
            print_debug(f"save path {save_path} already exists", color_name='COLOR_BLUE')

        # keys
        try:
            shoot_key = shoot_key.lower()
            exit_key = exit_key.lower()
            empty_cache_key = empty_cache_key.lower()
        except Exception as e:
            raise Exception(f'Unknown Exception: {e}')

        # get the camera
        camera = self.camera_manager.get_camera(self.camera_position)

        # keep looping until the specific exit key is pressed
        self.shoot_loop(camera=camera, shoot_key=shoot_key, exit_key=exit_key, empty_cache_key=empty_cache_key, save_path=save_path)

    @debug_decorator(
        head_message='shooting images...',
        tail_message='shooting images finished',
        color_name='COLOR_CYAN',
        bold=True
    )
    def shoot_loop(self, camera:RealsenseCamera, save_path:str, shoot_key:str='s', exit_key:str='q', empty_cache_key:str='r') -> None:
        """
        shoot images from the camera
        """
        # file tree width
        file_tree_width = self.width

        # debug message
        message_position = (self.width//10, self.height//7)
        message = f"Press '{shoot_key}' to shoot images, '{exit_key}' to exit, '{empty_cache_key}' to empty cache"
        print_debug(message, color_name='COLOR_CYAN')

        # screen message: "press 'somekey' to exit"
        # calculate text size and baseline
        messages = [f'Press "{exit_key.upper()}" to exit', f'Press "{shoot_key.upper()}" to shoot images', f'Press "{empty_cache_key.upper()}" to empty cache']
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        message_color=(255,255,0) # cyan text
        message_thickness = 1
        # message background
        padding = 5
        background_color = (0,0,0) # black background

        file_tree_position = (self.width, self.height//20)
        file_tree_fontScale = 0.5

        
        # keep looping until the specific exit key is pressed
        while True:

            # create a "black screen"
            screen = np.zeros((self.height, self.width + file_tree_width, 3), dtype=np.uint8)


            # get color and depth image(_depth_image and _aligned_depth_frame are to be discarded)
            color_image = camera.get_color_image()

            # "stick" the images onto the screen
            screen[:,:self.width, :] = color_image # color image

            # screen message demo
            (text_width, text_height), baseline = cv2.getTextSize(messages[0], fontFace, fontScale, message_thickness)
            top_left = (message_position[0] - padding, message_position[1] - 2*text_height - 2*padding)
            bottom_right = (message_position[0] + text_width + padding, message_position[1] + padding - text_height)
            for i,message in enumerate(messages):
                # calculate text size and baseline
                (text_width, text_height), baseline = cv2.getTextSize(message, fontFace, fontScale, message_thickness)
                # rectangular background bound
                top_left = (top_left[0], top_left[1] + text_height + padding*2)
                bottom_right = (top_left[0] + text_width + padding, top_left[1] + text_height + padding*2)
                cv2.rectangle(screen,top_left, bottom_right, background_color, -1)
                # demo "press 'somekey' to exit" on the screen
                cv2.putText(screen, f'{message}', (message_position[0], message_position[1] + i*(text_height + padding*2)), 
                            fontFace = fontFace, fontScale = fontScale,color=message_color, thickness=message_thickness)
            # end of screen message demo

            # file tree demo
            img_path_list = ['    '+pathlib.Path(img_path).name for img_path in pathlib.Path(save_path).glob('*')] # get all files under save path
            # add title to the file tree
            img_path_list.insert(0, f'intrinsics_images: ({len(img_path_list)} images)')
            (text_width, text_height), baseline = cv2.getTextSize(img_path_list[0], fontFace, fontScale, message_thickness)
            
            for i,img_path in enumerate(img_path_list):
                # demo all the files in the file tree
                cv2.putText(screen, f'{img_path}', (file_tree_position[0], file_tree_position[1] + i*(text_height + padding*2)), 
                            fontFace = fontFace, fontScale = file_tree_fontScale,color=message_color, thickness=message_thickness)
            # demo the screen
            cv2.imshow('camera_'+camera.serial_number+' Realtime Viewer', screen)

            ord_key_pressed = cv2.waitKey(1) & 0xFF

            if ord_key_pressed == ord(exit_key):
                cv2.destroyWindow('camera_'+camera.serial_number+' Realtime Viewer')
                break
            elif ord_key_pressed == ord(shoot_key):
                # shoot image
                color_image = camera.get_color_image()
                curr_time = datetime.datetime.now()
                img_name = f"{curr_time.year}_{curr_time.month}_{curr_time.day}-{curr_time.hour}_{curr_time.minute}_{curr_time.second}_{curr_time.microsecond//1000}.png"
                img_path = pathlib.Path(save_path, img_name)
                cv2.imwrite(img_path, color_image)
                print_debug(f"image {img_name} has been saved to {img_path}", color_name='COLOR_GREEN')
            elif ord_key_pressed == ord(empty_cache_key):
                # empty cache
                os.system(f"rm -rf {save_path}/*")
                print_debug(f"cache {save_path} has been emptied", color_name='COLOR_GREEN')


    def calculate_intrinsics(self):
        pass

if __name__ == '__main__':
    calculate_intrinsics = CalculateIntrinsics(pc_id=2, camera_position='global', use_cache=False)
    calculate_intrinsics.shoot_images(shoot_key='s', exit_key='q', empty_cache_key='r', save_dir='./intrinsics_images')
