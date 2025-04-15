"""
calculate the intrinsics of the camera from multiple chessboard images
"""

# import necessary packages
import cv2 # for image processing
import pathlib # for path operations
import datetime # for time operations
import os # for os operations
import numpy as np # for array operations

# import the necessary modules
from realsense_camera import RealsenseCamera # to use cameras
from camera_manager import CameraManager # to use cameras
from debug_decorators import print_debug,debug_decorator # for debugging messages
from visualizer import Visualizer # for visualizing the images

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

    def shoot_images(self, shoot_key:str='s', exit_key:str='q', empty_cache_key:str='r', calculate_intrinsics_key:str='i'):
        """
        shoot images from the camera

        intputs:
            - shoot_key: the key to press to shoot an image, default is 's'
            - exit_key: the key to press to exit the shooting loop, default is 'q'
            - empty_cache_key: the key to press to empty the cache directory, default is 'r'
            - calculate_intrinsics_key: the key to press to calculate the intrinsics, default is 'i'
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
            calculate_intrinsics_key = calculate_intrinsics_key.lower()
        except Exception as e:
            raise Exception(f'Unknown Exception: {e}')

        # get the camera
        camera = self.camera_manager.get_camera(self.camera_position)

        # keep looping until the specific exit key is pressed
        self.shoot_loop(camera=camera, shoot_key=shoot_key, exit_key=exit_key, empty_cache_key=empty_cache_key, calculate_intrinsics_key=calculate_intrinsics_key)

    @debug_decorator(
        head_message='shooting images...',
        tail_message='shooting images finished',
        color_name='COLOR_CYAN',
        bold=True
    )
    def shoot_loop(self, camera:RealsenseCamera, shoot_key:str='s', exit_key:str='q', empty_cache_key:str='r', calculate_intrinsics_key:str='i') -> None:
        """
        shoot images from the camera
        
        inputs:
            - camera: the camera instance of RealsenseCamera
            - shoot_key: the key to press to shoot an image, default is 's'
            - exit_key: the key to press to exit the shooting loop, default is 'q'
            - empty_cache_key: the key to press to empty the cache directory, default is 'r'
            - calculate_intrinsics_key: the key to press to calculate the intrinsics, default is 'i'
        """
        # file tree width
        
        file_tree_width = self.width

        # debug message
        message_position = (self.width//10, self.height//7)
        message = f"Press '{shoot_key}' to shoot images, '{exit_key}' to exit, '{empty_cache_key}' to empty cache"
        print_debug(message, color_name='COLOR_CYAN')

        # screen message: "press 'somekey' to exit"
        # calculate text size and baseline
        messages = [f'Press "{exit_key.upper()}" to exit', 
                    f'Press "{shoot_key.upper()}" to shoot images',
                    f'Press "{empty_cache_key.upper()}" to empty cache',
                    f'Press "{calculate_intrinsics_key.lower()}" to calculate intrinsics']
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
            # screen[:,:self.width, :] = color_image # color image
            screen[:,:self.width, :] = annotated_color_image # annotated color image

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
            img_path_list = ['    '+pathlib.Path(img_path).name for img_path in pathlib.Path(self.save_path).glob('*.png')] # get all files under save path
            # add title to the file tree
            img_path_list.insert(0, f'intrinsics_images: ({len(img_path_list)} images)')
            (text_width, text_height), baseline = cv2.getTextSize(img_path_list[0], fontFace, fontScale, message_thickness)
            
            for i,img_path in enumerate(img_path_list):
                # demo all the files in the file tree
                cv2.putText(screen, f'{img_path}', (file_tree_position[0], file_tree_position[1] + i*(text_height + padding*2)), 
                            fontFace = fontFace, fontScale = file_tree_fontScale,color=message_color, thickness=message_thickness)
            # demo the screen
            cv2.imshow('camera_'+camera.serial_number+' Realtime Viewer', screen)


            # get the key pressed
            ord_key_pressed = cv2.waitKey(1) & 0xFF
            # key event
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
            elif ord_key_pressed == ord(calculate_intrinsics_key):
                # calculate intrinsics
                calculated_intrinsics_matrix = self.calculate_intrinsics()
                print_debug(f"intrinsics have been calculated:\n{calculated_intrinsics_matrix}", color_name='COLOR_GREEN')
                print_debug(f"camera intrinsics:\n{camera.intrinsics_matrix}", color_name='COLOR_GREEN')

                # save the intrinsics to a file
                with open(pathlib.Path(self.save_path, 'intrinsics.txt'), 'w') as f:
                    # write the intrinsics matrix as list of lists
                    f.write(str(calculated_intrinsics_matrix.tolist()))
                print_debug(f"intrinsics have been saved to {pathlib.Path(self.save_path, 'intrinsics.txt')}", color_name='COLOR_GREEN')


    def calculate_intrinsics(self):
        """
        calculate the intrinsics of the camera according to current save path
        """
        # get all files under save path
        img_path_list = [str(pathlib.Path(img_path)) for img_path in pathlib.Path(self.save_path).glob('*.png')]

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
        else:
            corners_subpixel = None
        # return the results
        return ret, corners_subpixel, annotated_color_image

# intrinsics calculator
class IntrinsicsCalculator(Visualizer):
    def __init__(self, pc_id:int, chessboard_shape:tuple[int], square_size:float ,camera_position:str='global', save_dir:str="./intrinsics_images", window_name:str='intrinsics_calculator') -> None:
        """
        initialize the intrinsics calculator
        
        inputs:
            - pc_id: the id of the pc
            - chessboard_shape: the shape of the chessboard, like a (9,6)-blocked chessboard oughted to be (8,5) as input
            - square_size: float, side length of a square(in METERS)
            - camera_position: the position of the camera['global', 'wrist']
            - save_dir: the directory path where the captured images will be saved, default is './intrinsics_images'
            - window_name: the name of the window, default is 'intrinsics_calculator'
        """
        super().__init__(height=480, width_left=640, width_right=640, window_name=window_name)

        # initialize the camera manager
        self.pc_id = pc_id
        self.camera_position = camera_position
        self.camera_manager = CameraManager(pc_id=pc_id)

        # initialize the camera
        self.camera = self.camera_manager.get_camera(self.camera_position)

        # resolution of the camera
        self.height = self.camera.height
        self.width = self.camera.width

        # initialize the chessboard shape
        self.chessboard_shape = chessboard_shape
        # initialize the square size
        self.square_size = square_size

        # initialize the save directory
        self.current_path = pathlib.Path(__file__).parent.resolve()
        self.save_dir = save_dir
        self.save_path = pathlib.Path(self.current_path, self.save_dir)

        # check if the save directory exists
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True, exist_ok=True)
            print_debug(f"save path {self.save_path} does not exist, and has been created", color_name='COLOR_GREEN')
        else:
            print_debug(f"save path {self.save_path} already exists", color_name='COLOR_BLUE')

        # set the keys
        self.set_keys(shoot_key='s', exit_key='q', empty_cache_key='r', calculate_intrinsics_key='i')

        # print debug information
        print_debug(f'camera position:{self.camera_position}', color_name='COLOR_YELLOW')
        print_debug(f'pc id:{self.pc_id}', color_name='COLOR_YELLOW')
        print_debug(f'camera resolution: {self.width}x{self.height}', color_name='COLOR_YELLOW')
        print_debug(f'save path: {self.save_path}', color_name='COLOR_YELLOW')
        print_debug(f'chessboard shape: {self.chessboard_shape}', color_name='COLOR_YELLOW')
        print_debug(f'square size: {self.square_size} meter', color_name='COLOR_YELLOW')
    
    def set_keys(self, shoot_key:str='s', exit_key:str='q', empty_cache_key:str='r', calculate_intrinsics_key:str='i'):
        """
        set the keys
        """
        # convert to lower case
        try:
            shoot_key = shoot_key.lower()
            exit_key = exit_key.lower()
            empty_cache_key = empty_cache_key.lower()
            calculate_intrinsics_key = calculate_intrinsics_key.lower()
        except Exception as e:
            raise Exception(f'Unknown Exception: {e}')
        # set the keys
        self.shoot_key = shoot_key
        self.exit_key = exit_key
        self.empty_cache_key = empty_cache_key
        self.calculate_intrinsics_key = calculate_intrinsics_key

        # print debug information
        print_debug(f'shoot key:{self.shoot_key.upper()}', color_name='COLOR_YELLOW')
        print_debug(f'exit key:{self.exit_key.upper()}', color_name='COLOR_YELLOW')
        print_debug(f'empty cache key:{self.empty_cache_key.upper()}', color_name='COLOR_YELLOW')
        print_debug(f'calculate intrinsics key:{self.calculate_intrinsics_key.upper()}', color_name='COLOR_YELLOW')

        # connect the keys to the functions
        self.add_keys(keys=[self.shoot_key,
                             self.exit_key,
                             self.empty_cache_key,
                             self.calculate_intrinsics_key], 
                       events=[self.__shoot,
                               self.__exit,
                               self.__empty_cache,
                               self._calculate_intrinsics])
        # end of key initialization

    @staticmethod
    def _detect_chessboard(color_image: np.ndarray, chessboard_shape:tuple[int]):
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
        else:
            corners_subpixel = None
        # return the results
        return ret, corners_subpixel, annotated_color_image
    
    def shoot_loop(self):
        """
        shoot images from the camera and save them to the save directory
        """
        # key message
        key_message = [f"Press '{self.shoot_key}' to shoot images,",
                    f"Press '{self.exit_key}' to exit,",
                    f"Press '{self.empty_cache_key}' to empty cache,",
                    f"Press '{self.calculate_intrinsics_key}' to calculate intrinsics"]
        key_message_position = (self.width//10, self.height//7)
        key_message_color = (255,255,0) # cyan text
        key_message_thickness = 1
        key_message_padding = 5
        key_message_background_color = (0,0,0) # black background

        # file tree message
        file_tree_position = (0, self.height//20)
        file_tree_fontScale = 0.5

        # keep looping until the specific exit key is pressed
        self.keep_looping = True
        while self.keep_looping:
            # get current color image
            color_image = self.camera.get_color_image()
            # "stick" the annotated image onto the screen
            _ret, _corners2, annotated_color_image = self._detect_chessboard(
                                        color_image,
                                        self.chessboard_shape)
            
            # add the annotated image to the screen
            self.set_screen_middle(annotated_color_image)
            # add key message to the screen
            self.add_words(words=key_message, screen_switch='left', 
                       position=key_message_position, color=key_message_color, 
                       thickness=key_message_thickness, padding=key_message_padding,
                        background_color=key_message_background_color)
            
            # add file tree to the screen
            file_tree_message = ['intrinsics_images:']
            file_tree_message.extend(['    '+pathlib.Path(img_path).name for img_path in pathlib.Path(self.save_path).glob('*.png')])
            self.add_words(words=file_tree_message, screen_switch='right', 
                       position=file_tree_position, font_scale=file_tree_fontScale)
        
            # show the screen
            self.show() # render and show the screen
        # end of shoot loop 

    def __shoot(self):
        """
        shoot image from the camera and save it to the save directory
        """
        color_image = self.camera.get_color_image()
        curr_time = datetime.datetime.now()
        img_name = f"{curr_time.year}_{curr_time.month}_{curr_time.day}-{curr_time.hour}_{curr_time.minute}_{curr_time.second}_{curr_time.microsecond//1000}.png"
        img_path = pathlib.Path(self.save_path, img_name)
        cv2.imwrite(img_path, color_image)
        print_debug(f"image {img_name} has been saved to {img_path}", color_name='COLOR_GREEN')

    def __exit(self):
        """
        exit the program
        """
        print_debug('exit', color_name='COLOR_GREEN')
        self.keep_looping = False
        self.close()

    def __empty_cache(self):
        """
        empty the cache directory
        """
        os.system(f"rm -rf {self.save_path}/*")
        print_debug(f"cache {self.save_path} has been emptied", color_name='COLOR_GREEN')


    def _calculate_intrinsics(self, save:bool=True):
        """
        calculate the intrinsics of the camera according to current save path and save it to a file
        """
        print_debug('calculating intrinsics', color_name='COLOR_GREEN')
        # get all files under save path
        img_path_list = [str(pathlib.Path(img_path)) for img_path in pathlib.Path(self.save_path).glob('*.png')]

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
            ret, corners_subpixel, _annotated_img = self._detect_chessboard(color_image, self.chessboard_shape)

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
        print_debug(f"intrinsics have been calculated:\n{calculated_intrinsics_matrix}", color_name='COLOR_GREEN')
        print_debug(f"camera intrinsics:\n{self.camera.intrinsics_matrix}", color_name='COLOR_GREEN')

        # save the intrinsics to a file
        if save:
            with open(pathlib.Path(self.save_path, 'intrinsics.txt'), 'w') as f:
                # write the intrinsics matrix as list of lists
                f.write(str(calculated_intrinsics_matrix.tolist()))
            print_debug(f"intrinsics have been saved to {pathlib.Path(self.save_path, 'intrinsics.txt')}", color_name='COLOR_GREEN')
        return calculated_intrinsics_matrix
if __name__ == '__main__':
    # calculate_intrinsics = CalculateIntrinsics(pc_id=2, chessboard_shape=(5,8), square_size=0.0261111, camera_position='global')
    # calculate_intrinsics.shoot_images(shoot_key='s', exit_key='q', empty_cache_key='r', calculate_intrinsics_key='i')
    ic = IntrinsicsCalculator(pc_id=2, chessboard_shape=(5,8), square_size=0.0261111, camera_position='global')
    ic.shoot_loop()