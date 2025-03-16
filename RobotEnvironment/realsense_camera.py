"""
Defination of RealsenseCamera class
"""

# import necessary libs
import pyrealsense2 as rs # camera lib
import time # to sleep for certain period

import numpy as np # matrix calculation
import cv2 # OpenCV image process

from debug_decorators import debug_decorator # method debug decorator

class RealsenseCamera(object):
    """
    our camera is Realsense d435i
    the class is to initialize the camera, take photos, and destroy pipeline of d435i
    """

    @debug_decorator(
        head_message='initializing realsense camera...',
        tail_message='realsense camera initialized!',
        color_name='COLOR_WHITE',
        bold=True
    )
    def __init__(self, serial_number:int, width:int=640, height:int=480, fps:int=30) -> None:
        """
        initialize the camera pipeline
        input:
            serial_number(int): serial number of the camera
            width(int): width of the photo
            height(int): height of the photo
            fps(int): frame rate of photo pipeline
        output:
            None
        """
        # serial number of the camera
        self.serial_number = str(serial_number)

        # frame resolution
        self.width = width
        self.height = height
        self.fps = fps
        # configuration of camera
        self.pipeline = rs.pipeline() # photo pipeline
        self.config = rs.config() # basic configuration
        # configure the settings of color images and depth images
        self.config.enable_device(self.serial_number)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, fps)

        # infrared images
        # self.config.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, fps)
        # self.config.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, fps)

        # image aligner
        self.align = rs.align(rs.stream.color)

        # start image pipeline
        self.pipeline.start(self.config)

        # camera intrinsics
        self.intrinsics = None # will be fetch later

        # wait for 2 sec until the stable state of the camera
        time.sleep(2)

        # get intrinsics from RealSense aligned depth frame obj
        _color_image, _depth_image, _, aligned_depth_frame = self.__get_frame()
        self.intrinsics = aligned_depth_frame.profile.as_video_stream_profile().get_intrinsics()
        # register intrinsics' attrs
        self.fx = self.intrinsics.fx # x direction focal length
        self.fy = self.intrinsics.fy # y direction focal length
        self.ppx = self.intrinsics.ppx # principal point(optical center) x coordinate
        self.ppy = self.intrinsics.ppy # principal point(optical center) y coordinate
        # calculate intrinsics matrix
        self.intrinsics_matrix = np.array([
            [self.fx, 0      , self.ppx],
            [0      , self.fy, self.ppy],
            [0      , 0      , 1       ]
        ])
        # end of initialization
    


    @debug_decorator(
        head_message='turning off realsense camera...',
        color_name='COLOR_WHITE',
        bold=True
    )
    def __del__(self) -> None:
        """
        destruction method (de-constructor/destructor)
        """
        # cut off the pipeline
        self.pipeline.stop()
        # end of deletion

    @debug_decorator(
        head_message='launching realtime viewer...',
        tail_message='viewer destroyed',
        color_name='COLOR_CYAN',
        bold=True
    )
    def launch_realtime_viewer(self,exit_key:str='q') -> None:
        """
        launch realtime viewer of camera

        inputs:
            exit_key: str, key to exit the viewer
        """
        # detect possible errors/exceptions
        try:
            exit_key = exit_key.lower()
        except Exception as e:
            raise Exception(f'Unknown Exception: {e}')
        
        # create a "black screen"
        screen = np.zeros((self.height, self.width*2, 3), dtype=np.uint8)
        # position of message "press 'somekey' to exit"
        message_position = (self.width//10*9, self.height//7)

        # keep looping until the specific exit key is pressed
        while True:
            # get color and depth image(_depth_image and _aligned_depth_frame are to be discarded)
            color_image, _depth_image, colorized_depth_image, _aligned_depth_frame = self.__get_frame()
            
            # "stick" the images onto the screen
            screen[:,:self.width, :] = color_image # color image on the left
            screen[:,self.width:, :] = colorized_depth_image # depth image on the right
            
            # screen message: "press 'somekey' to exit"
            # calculate text size and baseline
            message = f'Press "{exit_key.upper()}" to exit'
            fontFace = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            message_color=(255,255,0) # cyan text
            message_thickness = 1
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
            cv2.imshow('camera_'+self.serial_number+' Realtime Viewer', screen)

            if cv2.waitKey(1) & 0xFF == ord(exit_key):
                cv2.destroyWindow('camera_'+self.serial_number+' Realtime Viewer')
                break


    def __get_frame(self) -> tuple:
        """
        immediately get frames from current pipeline

        outputs:
            - color_image: pure color image
            - depthx_image: original depth data image(not pseudo color)
            - colorizer_depth: pseudo color depth image for visualization
            - aligned_depth_frame: Depth Frame obj by RealSense, for intrinsics and other information
        """
        # get immediate frames(color and depth)
        frames = self.pipeline.wait_for_frames()
        # pseudo color image obj
        colorizer = rs.colorizer()

        # aligner
        align_to = rs.stream.color
        align = rs.align(align_to)
        aligned_frames = align.process(frames) # images aligned by aligner
        # aligner frames(color and depth)
        color_frame = aligned_frames.get_color_frame()
        aligned_depth_frame = aligned_frames.get_depth_frame()

        # cvt frame into image(array/matrix)
        color_image = np.asanyarray(color_frame.get_data())
        depthx_image = np.asanyarray(aligned_depth_frame.get_data())  # corresponding original depth image
        colorizer_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

        # return frames/images
        return color_image, depthx_image, colorizer_depth, aligned_depth_frame
    
    def get_color_image(self) -> np.ndarray:
        """
        get current color image (H,W,3)
        
        outputs:
        - color_image: np.ndarray (H,W,3), current color image
        """
        # get current frames(everything BUT color_image will be discarded)
        color_image, _depthx_image, _colorizer_depth, _aligned_depth_frame = self.__get_frame()
        return color_image
    
    def get_depth_image(self) -> np.ndarray:
        """
        get current depth image (H,W), depth in mm
        
        outputs:
        - depth_image: np.ndarray (H,W), current depth image
        """
        # get current frames(everything BUT depth_image will be discarded)
        _color_image, depth_image, _colorizer_depth, _aligned_depth_frame = self.__get_frame()
        return depth_image

    def get_colorized_depth_image(self) -> np.ndarray:
        """
        get current colorized depth image (H,W,3)
        
        outputs:
        - colorized_depth_image: np.ndarray (H,W,3), current colorized depth image
        """
        _color_image, _depth_image, colorized_depth_image, _aligned_depth_frame = self.__get_frame()
        return colorized_depth_image

if __name__ == '__main__':
    cam = RealsenseCamera()
    # cam.launch_realtime_viewer()
    img = cam.get_depth_image()
    print(img)