# robot environment

# import necessary libs
import pyrealsense2 as rs # to control the camera
import numpy as np
import os

import time

import logging

import sys
import logging

# import necessary modules
from visualizer import Visualizer
from auto_callibration import AutoCallibrator
from camera_manager import CameraManager
from debug_decorators import debug_decorator, print_debug

class RobotEnvironment(Visualizer):
    """
    the class to control the robot environment
    """
    def __init__(self, pc_id: int = 0):
        """
        initialize the robot environment
        """
        super().__init__(window_name='robot_environment')
        if pc_id not in [1, 2]:
            raise ValueError(f'pc_id must be 1 or 2, but got {pc_id}')
        self.pc_id = pc_id
        self.camera_manager = CameraManager(pc_id=pc_id)
        self.camera = self.camera_manager.get_camera('global')

        # MUST WARM UP BEFORE IT MOVES
        self.warmed_up = False

        # set the keys
        self.keys = []
        self.events = []
        self.__set_keys()

    def __warm_up(self):
        """
        warm up the robot
        """
        # warm up the robot
        if self.warmed_up:
            return
        self.warmed_up = True
        print_debug('warming up the robot...', color_name='COLOR_WHITE')
        time.sleep(1)
        print_debug('robot warmed up!', color_name='COLOR_WHITE')

    def run_loop(self):
        """
        the main loop of the robot environment
        """
        while True:
            self.set_screen_left(self.camera.get_color_image())

            self.show()

    def __set_keys(self,
                   quit_key: str = 'q',
                   ):
        """
        set up the keys
        """
        self.keys.extend([quit_key])
        self.events.extend([self.__quit])

    def __quit(self):
        """
        quit the robot environment
        """
        self.close()
        sys.exit(0)

if __name__ == '__main__':
    robot_environment = RobotEnvironment(pc_id=2)
    robot_environment.run_loop()

