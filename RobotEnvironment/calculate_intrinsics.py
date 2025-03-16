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
    def __init__(self, pc_id:int, chessboard_shape:tuple[int], square_size:float ,camera_position:str='global', save_dir:str="./intrinsics_images") -> None:
        """
        initialize the class

        inputs:
            - pc_id: the id of the pc
            - chessboard_shape: (x,y) of chessboard CROSS-CORNERS, like a (9,6)-blocked chessboard oughted to be (8,5) as input
            - square_size: float, side length of a square(in METERS)
            - camera_position: the position of the camera['global', 'wrist']
            - save_dir: the directory path where the captured images will be saved, default is './intrinsics_images'
        """
        self.pc_id = pc_id # initialize the pc id
        self.camera_position = camera_position # initialize the camera position
        self.camera_manager = CameraManager(pc_id=pc_id) # initialize the camera manager

        # resolution of the camera
        self.height = self.camera_manager.height
        self.width = self.camera_manager.width

        # get save directory and save path
        self.current_path = pathlib.Path(__file__).parent.resolve()
        self.save_dir = save_dir
        self.save_path = pathlib.Path(self.current_path, self.save_dir)

        # chessboard shape
        self.chessboard_shape = chessboard_shape
        # square size
        self.square_size = square_size

        # print debug information
        print_debug(f'camera position:{self.camera_position}', color_name='COLOR_YELLOW')
        print_debug(f'pc id:{self.pc_id}', color_name='COLOR_YELLOW')
        print_debug(f'camera resolution: {self.width}x{self.height}', color_name='COLOR_YELLOW')
        print_debug(f'save path: {self.save_path}', color_name='COLOR_YELLOW')
        print_debug(f'chessboard shape: {self.chessboard_shape}', color_name='COLOR_YELLOW')
        print_debug(f'square size: {self.square_size} meter', color_name='COLOR_YELLOW')

        # end of initialization

    def shoot_images(self, shoot_key:str='s', exit_key:str='q', empty_cache_key:str='r'):
        """
        shoot images from the camera

        intputs:
            - shoot_key: the key to press to shoot an image, default is 's'
            - exit_key: the key to press to exit the shooting loop, default is 'q'
            - empty_cache_key: the key to press to empty the cache directory, default is 'r'
            - save_dir: the directory path where the captured images will be saved, default is './intrinsics_images'
        """
        # check if the save directory exists
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True, exist_ok=True)
            print_debug(f"save path {self.save_path} does not exist, and has been created", color_name='COLOR_GREEN')
        else:
            print_debug(f"save path {self.save_path} already exists", color_name='COLOR_BLUE')

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
        self.shoot_loop(camera=camera, shoot_key=shoot_key, exit_key=exit_key, empty_cache_key=empty_cache_key)

    @debug_decorator(
        head_message='shooting images...',
        tail_message='shooting images finished',
        color_name='COLOR_CYAN',
        bold=True
    )
    def shoot_loop(self, camera:RealsenseCamera, shoot_key:str='s', exit_key:str='q', empty_cache_key:str='r') -> None:
        """
        shoot images from the camera
        
        inputs:
            - camera: the camera instance of RealsenseCamera
            - shoot_key: the key to press to shoot an image, default is 's'
            - exit_key: the key to press to exit the shooting loop, default is 'q'
            - empty_cache_key: the key to press to empty the cache directory, default is 'r'
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

            # "stick" the annotated image onto the screen
            _ret, _corners2, annotated_color_image = self.__detect_chessboard(
                                        color_image,
                                        self.chessboard_shape)
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
            img_path_list = ['    '+pathlib.Path(img_path).name for img_path in pathlib.Path(self.save_path).glob('*')] # get all files under save path
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
                img_path = pathlib.Path(self.save_path, img_name)
                cv2.imwrite(img_path, color_image)
                print_debug(f"image {img_name} has been saved to {img_path}", color_name='COLOR_GREEN')
            elif ord_key_pressed == ord(empty_cache_key):
                # empty cache
                os.system(f"rm -rf {self.save_path}/*")
                print_debug(f"cache {self.save_path} has been emptied", color_name='COLOR_GREEN')


    def calculate_intrinsics(self):
        """
        calculate the intrinsics of the camera according to current save path
        """
        # get all files under save path
        img_path_list = [str(pathlib.Path(img_path)) for img_path in pathlib.Path(self.save_path).glob('*')]

        if len(img_path_list)<5:
            print_debug('not enough images! at least 5 of them')
            return
        
        # store all 3D and 2D points
        obj_points_list = [] # 3d points in real world space
        img_points_list = [] # 2d pixel points in image plane

        # get the 3D points in real world space
        # (they are the same in the whole directory)
        obj_points = np.zeros((self.chessboard_shape[0]*self.chessboard_shape[1], 3), np.float32)
        obj_points[:,:2] = np.mgrid[0:self.chessboard_shape[0], 0:self.chessboard_shape[1]].T.reshape(-1, 2)*self.square_size

        # process all the images under save_path
        for img_path in img_path_list:
            # get the image ndarray
            color_image = cv2.imread(img_path)
            # detect corners
            ret, corners_subpixel, _annotated_img = self.__detect_chessboard(color_image, self.chessboard_shape)

            # if corners found, then add them to the list
            if ret:
                # add 3d points to the list
                obj_points_list.append(obj_points)
                # add 2d points to the list
                img_points_list.append(corners_subpixel)
        
        # then use OpenCV API to calculate intrinsics
        color_image_shape = (cv2.imread(img_path_list[0])).shape
        ret, calculated_intrinsics_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points_list, img_points_list, (color_image_shape[1], color_image_shape[0]), None, None
        )
        # return calculated intrinsics matrix
        return calculated_intrinsics_matrix
          


    @staticmethod
    def __detect_chessboard(color_image: np.ndarray, chessboard_shape:tuple[int]):
        """
        detect chessboard corners

        inputs:
            - color_image: the color image of the chessboard
            - chessboard_shape: the shape of the chessboard, like a (9,6)-blocked chessboard oughted to be (8,5) as input
        outputs:
            - ret: bool, whether corners detected or not
            - corners_subpixel: np.ndarray, the subpixel corners of the chessboard
            - annotated_color_image: np.ndarray, the corner annotated color image
        """
        # convert to gray image
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # detect chessboard corners
        ret, corners = cv2.findChessboardCorners(gray_image, chessboard_shape, None)

        # if corners detected, then annotate
        annotated_color_image = color_image.copy() # will be returned whatever
        if ret:
            # corner subpixel find termination criteria
            criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
            # find subpixel corner position
            corners_subpixel = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)

            # begin to annotate and draw the corners
            cv2.drawChessboardCorners(annotated_color_image, chessboard_shape, corners, ret)

        # return the results
        return ret, corners_subpixel, annotated_color_image

if __name__ == '__main__':
    calculate_intrinsics = CalculateIntrinsics(pc_id=2, chessboard_shape=(8,6), square_size=0, camera_position='global')
    calculate_intrinsics.shoot_images(shoot_key='s', exit_key='q', empty_cache_key='r')
