"""
define a class of camera manager that manages multiple cameras
"""

# import necessary libs
from realsense_camera import RealsenseCamera # to use cameras
import pandas as pd # read camera csv
import pyrealsense2 as rs # official lib of RealSense Camneras
import time # wait for cameras to initialize
import numpy as np # to handle numpy arrays
import cv2 # OpenCV image process
import pathlib # to get the path of the current file

from debug_decorators import print_debug,debug_decorator


class CameraManager(object):
    """
    Camera Manager
    """
    @debug_decorator(
        'creating camera manager',
        'YOUR CAMERAS HAVE SUCCESSFULLY INITIALIZED',
        'COLOR_GREEN'
    )
    def __init__(self, pc_id:int, width:int=640, height:int=480, fps:int=30) -> None:
        """
        Manage multiple cameras according to the id of pc

        inputs:
            pc_id:int, must be 1 or 2
        """
        # check if pc_id is valid
        if pc_id != 1 and pc_id!= 2:
            raise Exception('ID of pc must be 1 or 2!!!')
        self.pc_id = pc_id # id of pc

        # frame resolution
        self.height = height
        self.width = width
        self.fps = fps
        
        # load registered_camera.csv to get
        # bi-directional relationship of serial number and device information
        # self.camera_register_path = './registered_camera.csv'
        self.camera_register_path = pathlib.Path(__file__).parent / 'registered_camera.csv'
        print_debug(f'Loading camera register from {self.camera_register_path}')
        camera_register_df = pd.read_csv(self.camera_register_path) # dataframe of camera register
        # create bi-directional dict: serial number <=> camera name
        camera_id_col = camera_register_df['camera_id']
        camera_name_col = camera_register_df['camera_name']
        self.id2name_dict = {id:name for id,name in zip(camera_id_col,camera_name_col)}
        self.name2id_dict = {name:id for name,id in zip(camera_name_col,camera_id_col)}
        
        # get all connected cameras 
        context = rs.context()
        # serial numbers of all connected cameras
        connected_devices = [int(d.get_info(rs.camera_info.serial_number)) for d in context.devices]
        # judge whether all cameras are in the register
        for device in connected_devices:
            if device not in self.id2name_dict.keys():
                raise Exception(f'Camera {device} is not registered')
        # get camera names and ids related to the pc
        self.possible_camera_names = [camera_name for camera_name in camera_name_col if camera_name.startswith(f'pc{pc_id}_')]
        self.possible_camera_ids = [self.name2id_dict[camera_name] for camera_name in self.possible_camera_names]
        # get camera names connected to the pc
        self.connected_camera_ids = [camera_id for camera_id in self.possible_camera_ids if camera_id in connected_devices]
        self.connected_camera_names = [self.id2name_dict[camera_id] for camera_id in self.connected_camera_ids]
        print_debug(f'connected cameras: {", ".join(self.connected_camera_names)}',color_name='COLOR_YELLOW')

        # initialize cameras
        camera_positions = ['global', 'wrist']
        print_debug(f'Initializing all cameras: {", ".join(camera_positions)}')
        self.camera_dict = {} # to store cameras
        self.notification_dict = {} # to store notifications of disconnected cameras
        for camera_position in camera_positions:
            camera_name = f'pc{pc_id}_{camera_position}_camera'
            if camera_name in self.connected_camera_names:
                print_debug(f'initializing {camera_name}...')
                self.camera_dict[camera_position] = RealsenseCamera(self.name2id_dict[camera_name], self.width, self.height, self.fps)
            else:
                print_debug(f'{camera_name} is not connected',color_name='COLOR_RED')
                self.notification_dict[camera_position] = 'not notificated'
                self.camera_dict[camera_position] = None
        time.sleep(2.5) # wait for cameras to initialize

        # end of initialization

    def get_camera(self, camera_position:str) -> RealsenseCamera:
        """
        get camera according to the position
        
        inputs:
            camera_position:str, position of camera: ['global', 'wrist']
        
        outputs:
            camera: RealsenseCamera, camera obj at the position
        """
        if camera_position not in self.camera_dict.keys():
            print_debug(f'The camera at position "{camera_position}" is not connected. Connected cameras: {self.connected_camera_names}',color_name='COLOR_RED')
            return None
        else:
            return self.camera_dict[camera_position]
    
    def get_global_color_image(self) -> np.ndarray:
        """
        get global camera color image

        outputs:
            color_image: np.ndarray (H,W,3), color image
        """
        if self.camera_dict['global'] is None:
            if self.notification_dict['global'] == 'not notificated':
                print_debug(f'The global camera is not connected',color_name='COLOR_RED')
                self.notification_dict['global'] = 'notified'
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return self.camera_dict['global'].get_color_image()
    
    def get_wrist_color_image(self) -> np.ndarray:
        """
        get wrist camera color image

        outputs:
            color_image: np.ndarray (H,W,3), color image
        """
        if self.camera_dict['wrist'] is None:
            if self.notification_dict['wrist'] == 'not notificated':
                print_debug(f'The wrist camera is not connected',color_name='COLOR_RED')
                self.notification_dict['wrist'] = 'notified'
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return self.camera_dict['wrist'].get_color_image()

    def get_global_depth_image(self) -> np.ndarray:
        """
        get global camera depth image

        outputs:
            depth_image: np.ndarray (H,W), depth image
        """
        if self.camera_dict['global'] is None:
            if self.notification_dict['global'] == 'not notificated':
                print_debug(f'The global camera is not connected',color_name='COLOR_RED')
                self.notification_dict['global'] = 'notified'
            return np.zeros((self.height, self.width), dtype=np.float32)
        return self.camera_dict['global'].get_depth_image()
    
    def get_wrist_depth_image(self) -> np.ndarray:
        """
        get wrist camera depth image

        outputs:
            depth_image: np.ndarray (H,W), depth image
        """
        if self.camera_dict['wrist'] is None:
            if self.notification_dict['wrist'] == 'not notificated':
                print_debug(f'The wrist camera is not connected',color_name='COLOR_RED')
                self.notification_dict['wrist'] = 'notified'
            return np.zeros((self.height, self.width), dtype=np.float32)
        return self.camera_dict['wrist'].get_depth_image()
    
    def get_global_colorized_depth_image(self) -> np.ndarray:
        """
        get global camera colorized depth image

        outputs:
            colorized_depth_image: np.ndarray (H,W,3), colorized depth image
        """
        if self.camera_dict['global'] is None:
            if self.notification_dict['global'] == 'not notificated':
                print_debug(f'The global camera is not connected',color_name='COLOR_RED')
                self.notification_dict['global'] = 'notified'
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return self.camera_dict['global'].get_colorized_depth_image()
    
    def get_wrist_colorized_depth_image(self) -> np.ndarray:
        """
        get wrist camera colorized depth image

        outputs:
            colorized_depth_image: np.ndarray (H,W,3), colorized depth image
        """
        if self.camera_dict['wrist'] is None:
            if self.notification_dict['wrist'] == 'not notificated':
                print_debug(f'The wrist camera is not connected',color_name='COLOR_RED')
                self.notification_dict['wrist'] = 'notified'
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return self.camera_dict['wrist'].get_colorized_depth_image()
    
    @debug_decorator(
        head_message='launching realtime viewer...',
        tail_message='viewer destroyed',
        color_name='COLOR_CYAN',
        bold=True
    )
    def launch_realtime_viewer(self, exit_key:str='q') -> None:
        """
        launch realtime viewer of cameras

        inputs:
            exit_key: str, key to exit the viewer
        """
        # detect possible errors/exceptions
        try:
            exit_key = exit_key.lower()
        except Exception as e:
            raise Exception(f'Unknown Exception: {e}')

        # create a "black screen"
        screen = np.zeros((self.height*2, self.width*2, 3), dtype=np.uint8)

        # position of message "press 'somekey' to exit"
        message_position = (self.width//10*9, self.height*2//7)

        # customize window size
        window_name = 'Realtime Viewer'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.width*3//2, self.height*3//2)

        # keep looping until the specific exit key is pressed
        while True:
            global_color_image = self.get_global_color_image()
            global_colorized_depth_image = self.get_global_colorized_depth_image()
            wrist_color_image = self.get_wrist_color_image()
            wrist_colorized_depth_image = self.get_wrist_colorized_depth_image()

            # "stick" the images onto the screen
            screen[:self.height, :self.width, :] = global_color_image # global color image on the upper left
            screen[:self.height, self.width:, :] = global_colorized_depth_image # global depth image on the upper right
            screen[self.height:, :self.width, :] = wrist_color_image # wrist color image on the lower left
            screen[self.height:, self.width:, :] = wrist_colorized_depth_image # depth image on the right

            # screen message: "press 'somekey' to exit"
            # calculate text size and baseline
            message = f'Press "{exit_key.upper()}" to exit'
            fontFace = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            message_color=(255,255,0) # cyan text
            message_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(message, fontFace, fontScale, message_thickness)
            # rectangular background bound
            padding = 5
            top_left = (message_position[0] - padding, message_position[1] - text_height - padding)
            bottom_right = (message_position[0] + text_width + padding, message_position[1] + padding)
            background_color = (0,0,0) # black background
            cv2.rectangle(screen,top_left, bottom_right, background_color, -1)
            # demo "press 'somekey' to exit" on the screen
            cv2.putText(screen, f'Press "{exit_key.upper()}" to exit', message_position, 
                        fontFace = fontFace, fontScale = fontScale,color=message_color, thickness=message_thickness)
            # end of screen message config

            # demo the screen
            cv2.imshow(window_name, screen)

            if cv2.waitKey(1) & 0xFF == ord(exit_key):
                cv2.destroyWindow(window_name)
                break



if __name__ == '__main__':
    cm = CameraManager(2)
    cm.launch_realtime_viewer()