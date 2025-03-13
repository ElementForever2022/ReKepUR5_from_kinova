"""
Defination of RealsenseCamera class
"""

# import necessary libs
import pyrealsense2 as rs # camera lib
import time # to sleep for certain period

class RealsenseCamera(object):
    """
    our camera is Realsense d435i
    the class is to initialize the camera, take photos, and destroy pipeline of d435i
    """
    def __init__(self, width:int=640, height:int=480, fps:int=30) -> None:
        """
        initialize the camera pipeline
        input:
            width(int): width of the photo
            height(int): height of the photo
            fps(int): frame rate of photo pipeline
        output:
            None
        """
        # print log
        print('[DEBUG]starting camera...',end='')
        
        # frame resolution
        self.width = width
        self.height = height

        # configuration of camera
        self.pipeline = rs.pipeline() # photo pipeline
        self.config = rs.config() # basic configuration
        # configure the settings of color images and depth images
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, fps)

        # infrared images
        # self.config.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, fps)
        # self.config.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, fps)

        # start image pipeline
        self.pipeline.start(self.config)
        # wait for 2 sec until the stable state of the camera
        time.sleep(2)

        # print the log and conclude the initialization
        print('done')
    
    def __del__():
        print('[DEBUG]')