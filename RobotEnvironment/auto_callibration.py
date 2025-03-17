# automatically callibrate the camera extrinsics

# import the necessary packages
import pathlib
import cv2
import numpy as np
import os
import sys
import time
import json

# import necessary modules
from camera_manager import CameraManager # to get the camera
from calculate_intrinsics import IntrinsicsCalculator # to visualize the images and calculate the intrinsics
from debug_decorators import debug_decorator,print_debug # to print debug information

class AutoCallibrator(IntrinsicsCalculator):
    def __init__(self,pc_id:int, camera_position: str, chessboard_shape: tuple, square_size: float, save_dir: str='./auto_callibration', window_name:str='auto_callibration')->None:
        """
        initialize the auto callibrator

        inputs:
            - pc_id: int, the id of the pc
            - camera_position: str, the position of the camera, ['left', 'right']
            - chessboard_shape: tuple, the shape of the chessboard
            - square_size: float, the size of the square
            - save_dir: str, the directory to save the results
            - window_name: str, the name of the window, default is 'auto_callibration'
        """
        super().__init__(pc_id=pc_id, chessboard_shape=chessboard_shape, square_size=square_size, camera_position=camera_position, save_dir='./intrinsics_images', window_name=window_name)
        self.set_auto_callibrator_keys()

        # initialize the position of original point of the chessboard
        self.position_x = None
        self.position_y = None
        self.camera_intrinsics = self.camera.intrinsics_matrix
        self.calculated_intrinsics_matrix = self._calculate_intrinsics(save=False)

    def set_auto_callibrator_keys(self, callibrate_key:str='c', set_x_key:str='x', set_y_key:str='y')->None:
        """
        set the keys

        inputs:
            - callibrate_key: str, the key to calculate the extrinsics
        """
        # set the keys
        self.callibrate_key = callibrate_key
        self.set_x_key = set_x_key
        self.set_y_key = set_y_key

        # print debug information
        print_debug(f'callibrate key:{self.callibrate_key.upper()}', color_name='COLOR_YELLOW')
        print_debug(f'set x key:{self.set_x_key.upper()}', color_name='COLOR_YELLOW')
        print_debug(f'set y key:{self.set_y_key.upper()}', color_name='COLOR_YELLOW')

        # add the keys
        self.add_keys(keys=[self.callibrate_key,
                            self.set_x_key,
                            self.set_y_key],
                    events=[self.__callibrate,
                            self.__set_position_x,
                            self.__set_position_y])

    def _calculate_intrinsics(self, save:bool=True)->None:
        """
        calculate the intrinsics of the camera
        """
        result = super()._calculate_intrinsics(save=save)
        self.calculated_intrinsics_matrix = result
        return result

    def callibration_loop(self)->None:
        """
        the loop to callibrate the camera
        """
        # key message
        key_message = [f"Press {self.callibrate_key.upper()} to callibrate the camera",
                       f"Press {self.shoot_key.upper()} to shoot images",
                       f"Press {self.exit_key.upper()} to exit",
                       f"Press {self.empty_cache_key.upper()} to empty cache",
                       f"Press {self.calculate_intrinsics_key.upper()} to calculate intrinsics",
                       f"Press {self.set_x_key.upper()} to set x position",
                       f"Press {self.set_y_key.upper()} to set y position"]
        key_message_position = (self.width//20, self.height//10)
        key_message_color = (255,255,0) # cyan text
        key_message_thickness = 2
        key_message_padding = 5
        key_message_background_color = (0,0,0) # black background

        # intrinsics message
        intrinsics_message_position = (self.width//10, self.height//7)
        intrinsics_message_color = (255,255,0) # cyan text
        intrinsics_message_thickness = 2
        intrinsics_message_padding = 5
        intrinsics_message_background_color = (0,0,0) # black background

        # the loop to callibrate the camera
        self.keep_looping = True
        while self.keep_looping:
            # get the image
            color_image = self.camera.get_color_image()

            # add the annotated image to the screen
            _ret, _corners2, annotated_color_image = self._detect_chessboard(
                                        color_image,
                                        self.chessboard_shape)
            self.set_screen_left(annotated_color_image)

            # add key message to the screen
            self.add_words(words=key_message, screen_switch='left', 
                       position=key_message_position, color=key_message_color, 
                       thickness=key_message_thickness, padding=key_message_padding,
                        background_color=key_message_background_color)
            
            # intrinsics message
            if self.calculated_intrinsics_matrix is None:
                self.calculated_intrinsics_matrix = np.array([[0,0,0],[0,0,0],[0,0,0]])
            if self.position_x is None:
                demo_x_position = 0.0
            else:
                demo_x_position = self.position_x
            if self.position_y is None:
                demo_y_position = 0.0
            else:
                demo_y_position = self.position_y
            intrinsics_message = [f"camera api intrinsics: ",
                              f"[[{self.camera_intrinsics[0,0]:.2f}, {self.camera_intrinsics[0,1]:.2f}, {self.camera_intrinsics[0,2]:.2f}]",
                              f"[{self.camera_intrinsics[1,0]:.2f}, {self.camera_intrinsics[1,1]:.2f}, {self.camera_intrinsics[1,2]:.2f}]",
                              f"[{self.camera_intrinsics[2,0]:.2f}, {self.camera_intrinsics[2,1]:.2f}, {self.camera_intrinsics[2,2]:.2f}]]",
                              f"",
                              f"calculated intrinsics: ",
                              f"[[{self.calculated_intrinsics_matrix[0,0]:.2f}, {self.calculated_intrinsics_matrix[0,1]:.2f}, {self.calculated_intrinsics_matrix[0,2]:.2f}]",
                              f"[{self.calculated_intrinsics_matrix[1,0]:.2f}, {self.calculated_intrinsics_matrix[1,1]:.2f}, {self.calculated_intrinsics_matrix[1,2]:.2f}]",
                              f"[{self.calculated_intrinsics_matrix[2,0]:.2f}, {self.calculated_intrinsics_matrix[2,1]:.2f}, {self.calculated_intrinsics_matrix[2,2]:.2f}]]",
                              f"",
                              f"position: x={demo_x_position:.2f} y={demo_y_position:.2f}"
                              ]
            # add intrinsics message to the screen
            self.add_words(words=intrinsics_message, screen_switch='right', 
                       position=intrinsics_message_position, color=intrinsics_message_color, 
                       thickness=intrinsics_message_thickness, padding=intrinsics_message_padding,
                        background_color=intrinsics_message_background_color)

            self.show() # render and show the screen
        # end of callibrate loop
    
    def __callibrate(self)->None:
        """
        the event to callibrate the camera
        """
        # check if the position of the original point of the chessboard is set
        if self.position_x is None or self.position_y is None:
            print_debug('Please set the position of the original point of the chessboard', color_name='COLOR_RED')
            return
        
        # callibrate the camera
        print_debug('callibrating the camera', color_name='COLOR_GREEN')
    
    def __set_position_x(self)->None:
        """
        set the position in x axis(in METERS) of the original point of the chessboard
        """
        self.position_x = float(input('Please input the position in x axis(in METERS) of the original point of the chessboard: '))

    def __set_position_y(self)->None:
        """
        set the position in y axis(in METERS) of the original point of the chessboard
        """
        self.position_y = float(input('Please input the position in y axis(in METERS) of the original point of the chessboard: '))





if __name__ == '__main__':
    auto_callibration = AutoCallibrator(
        pc_id=2,
        camera_position='global',
        chessboard_shape=(5,8),
        square_size=0.02611,
        save_dir='./auto_callibration',
        window_name='auto_callibration'
    )
    auto_callibration.callibration_loop()