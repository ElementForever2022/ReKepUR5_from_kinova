# automatically callibrate the camera extrinsics

# import the necessary packages
import pathlib
import cv2
import numpy as np
import os
import sys
import time
import json
from scipy.spatial.transform import Rotation as R

# import necessary modules
from camera_manager import CameraManager # to get the camera
from calculate_intrinsics import IntrinsicsCalculator # to visualize the images and calculate the intrinsics
from debug_decorators import debug_decorator,print_debug # to print debug information

class AutoCallibrator(IntrinsicsCalculator):
    def __init__(self,pc_id:int, camera_position: str, chessboard_shape: tuple, square_size: float, save_file: str='./auto_callibration.json', window_name:str='auto_callibration')->None:
        """
        initialize the auto callibrator

        inputs:
            - pc_id: int, the id of the pc
            - camera_position: str, the position of the camera, ['left', 'right']
            - chessboard_shape: tuple, the shape of the chessboard, like a (9,6)-blocked chessboard oughted to be (8,5) as input
            - square_size: float, the size of the square, in meters, usually 0.0261
            - save_file: str, the file to save the results
            - window_name: str, the name of the window, default is 'auto_callibration'
        """
        super().__init__(pc_id=pc_id, chessboard_shape=chessboard_shape, square_size=square_size, camera_position=camera_position, save_dir='./intrinsics_images', window_name=window_name)
        self.set_auto_callibrator_keys()

        # initialize the position of original point of the chessboard
        # self.position_x = None
        # self.position_y = None
        self.position_x = -0.201
        self.position_y = -0.5515
        self.camera_intrinsics = self.camera.intrinsics_matrix
        self.calculated_intrinsics_matrix = self._calculate_intrinsics(save=False)

        # initialize the orientation of the chessboard
        self.chessboard_rvecs = np.array([-180,0,-180])
        print_debug(f'chessboard rvecs: {self.chessboard_rvecs}', color_name='COLOR_YELLOW')

        # initialize the robot2camera matrix
        self.robot2camera = None
        self.save_file = save_file

        # initialize the clicked point coordinates
        self.click_pixel_coords = None # the pixel coordinates of the clicked point: (u,v)
        self.click_camera_coords = None # the camera coordinates of the clicked point: (x,y,z)
        self.click_robot_coords = None # the robot coordinates of the clicked point: (x,y,z)

        self.from_calculated_intrinsics = False

    def set_auto_callibrator_keys(self, callibrate_key:str='c', set_x_key:str='x', set_y_key:str='y', callibate_from_calculated_intrinsics_key:str='v', save_matrix_key:str='z')->None:
        """
        set the keys

        inputs:
            - callibrate_key: str, the key to calculate the extrinsics
        """
        # set the keys
        self.callibrate_key = callibrate_key
        self.set_x_key = set_x_key
        self.set_y_key = set_y_key
        self.callibate_from_calculated_intrinsics_key = callibate_from_calculated_intrinsics_key
        self.save_matrix_key = save_matrix_key

        # print debug information
        print_debug(f'callibrate key:{self.callibrate_key.upper()}', color_name='COLOR_YELLOW')
        print_debug(f'set x key:{self.set_x_key.upper()}', color_name='COLOR_YELLOW')
        print_debug(f'set y key:{self.set_y_key.upper()}', color_name='COLOR_YELLOW')
        print_debug(f'callibrate from calculated intrinsics key:{self.callibate_from_calculated_intrinsics_key.upper()}', color_name='COLOR_YELLOW')
        print_debug(f'save matrix key:{self.save_matrix_key.upper()}', color_name='COLOR_YELLOW')

        # add the keys
        self.add_keys(keys=[self.callibrate_key,
                            self.set_x_key,
                            self.set_y_key,
                            self.callibate_from_calculated_intrinsics_key,
                            self.save_matrix_key],
                    events=[self.__callibrate,
                            self.__set_position_x,
                            self.__set_position_y,
                            self.__callibrate_from_calculated_intrinsics,
                            self.__save_matrix])
        
        # set mouse callback
        cv2.setMouseCallback(self.window_name, self.__mouse_callback)

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
                       f"Press {self.set_y_key.upper()} to set y position",
                       f"Press {self.callibate_from_calculated_intrinsics_key.upper()} to callibrate from calculated intrinsics",
                       f"Press {self.save_matrix_key.upper()} to save the matrix"]
        key_message_position = (self.width//20, self.height//10)
        key_message_color = (255,255,0) # cyan text
        key_message_thickness = 2
        key_message_padding = 5
        key_message_background_color = (0,0,0) # black background

        # intrinsics message
        intrinsics_message_position = (self.width//10, self.height//10)
        intrinsics_message_color = (255,255,0) # cyan text
        intrinsics_message_thickness = 2
        intrinsics_message_padding = 5
        intrinsics_message_background_color = (0,0,0) # black background

        # clicked point message
        clicked_message_position = (self.width//10, 2*self.height//3)

        # the loop to callibrate the camera
        self.keep_looping = True
        while self.keep_looping:
            # get the image
            color_image = self.camera.get_color_image()

            # add the annotated image to the screen
            _ret, _corners2, annotated_color_image = self._detect_chessboard(
                                        color_image,
                                        self.chessboard_shape)
            self.set_screen_middle(annotated_color_image)

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
                              f" [{self.camera_intrinsics[1,0]:.2f}, {self.camera_intrinsics[1,1]:.2f}, {self.camera_intrinsics[1,2]:.2f}]",
                              f" [{self.camera_intrinsics[2,0]:.2f}, {self.camera_intrinsics[2,1]:.2f}, {self.camera_intrinsics[2,2]:.2f}]]",
                              f"",
                              f"calculated intrinsics: ",
                              f"[[{self.calculated_intrinsics_matrix[0,0]:.2f}, {self.calculated_intrinsics_matrix[0,1]:.2f}, {self.calculated_intrinsics_matrix[0,2]:.2f}]",
                              f" [{self.calculated_intrinsics_matrix[1,0]:.2f}, {self.calculated_intrinsics_matrix[1,1]:.2f}, {self.calculated_intrinsics_matrix[1,2]:.2f}]",
                              f" [{self.calculated_intrinsics_matrix[2,0]:.2f}, {self.calculated_intrinsics_matrix[2,1]:.2f}, {self.calculated_intrinsics_matrix[2,2]:.2f}]]",
                              f"",
                              f"chessboard position: ",
                              f"    x={demo_x_position:.2f} y={demo_y_position:.2f}",
                              f"chessboard orientation: ",
                              f"    x={self.chessboard_rvecs[0]:.2f} y={self.chessboard_rvecs[1]:.2f} z={self.chessboard_rvecs[2]:.2f}"
                              f"",
                              f"from calculated intrinsics: {self.from_calculated_intrinsics}"
                              ]
            # add intrinsics message to the screen
            self.add_words(words=intrinsics_message, screen_switch='right', 
                       position=intrinsics_message_position, color=intrinsics_message_color, 
                       thickness=intrinsics_message_thickness, padding=intrinsics_message_padding,
                        background_color=intrinsics_message_background_color,
                        font_scale=0.5)

            # add clicked point message to the screen
            if self.click_pixel_coords is not None:
                self.add_words(words=[f"clicked at ({self.click_pixel_coords[0]}, {self.click_pixel_coords[1]})",
                                      f"camera coordinates: {self.click_camera_coords[0]:.2f}, {self.click_camera_coords[1]:.2f}, {self.click_camera_coords[2]:.2f}",
                                      f"robot coordinates: {self.click_robot_coords[0]:.2f}, {self.click_robot_coords[1]:.2f}, {self.click_robot_coords[2]:.2f}"], screen_switch='right', 
                       position=clicked_message_position, color=intrinsics_message_color, 
                       thickness=intrinsics_message_thickness, padding=intrinsics_message_padding,
                        background_color=intrinsics_message_background_color,
                        font_scale=0.5)

            self.show() # render and show the screen
        # end of callibrate loop
    
    def __callibrate_from_calculated_intrinsics(self)->None:
        """
        the event to callibrate the camera from calculated intrinsics
        """
        self.from_calculated_intrinsics = not self.from_calculated_intrinsics

    def __save_matrix(self)->None:
        """
        the event to save the matrix
        """
        if self.robot2camera is None:
            print_debug('Please callibrate the camera first', color_name='COLOR_RED')
            return
        print_debug('saving the matrix', color_name='COLOR_GREEN')
        current_dir = pathlib.Path(__file__).parent
        template_file = current_dir / 'templatefile.json'
        with open(template_file, 'r') as f:
            template_data = json.load(f)
        # template_data['misc']['world2robot_homo'] = self.robot2camera.tolist()
        # save the inverse matrix
        template_data['misc']['world2robot_homo'] = np.linalg.inv(self.robot2camera).tolist()
        with open(current_dir / self.save_file, 'w') as f:
            json.dump(template_data, f, indent=4, ensure_ascii=False, sort_keys=True)
        print_debug(f'matrix saved to {current_dir / self.save_file}', color_name='COLOR_GREEN')
    
    
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

        # calculate the robo[os.path.join(directory, name) for name in image_names] t2board matrix
        r2b_rvecs = self.chessboard_rvecs
        r2b_tvecs = np.array([self.position_x, self.position_y, 0])
        r2b_rotation_matrix = R.from_euler('xyz', r2b_rvecs, degrees=True).as_matrix()
        r2b_rotation_matrix = np.linalg.inv(r2b_rotation_matrix)
        r2b_translation_vector = r2b_tvecs.ravel()
        r2b_matrix = np.eye(4)
        r2b_matrix[:3, :3] = r2b_rotation_matrix
        r2b_matrix[:3, 3] = r2b_translation_vector
        self.robot2board = np.linalg.inv(r2b_matrix)

        # calculate the board2camera matrix
        current_color_image = self.camera.get_color_image()
        obj_points_list = [] # 3D points of the chessboard
        img_points_list = [] # 2D points of the chessboard
        gray = cv2.cvtColor(current_color_image, cv2.COLOR_BGR2GRAY)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_shape, None)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points_list.append(corners2)
        obj_points = np.zeros((self.chessboard_shape[1] * self.chessboard_shape[0], 3), np.float32)
        obj_points[:, :2] = np.mgrid[0:self.chessboard_shape[0], 0:self.chessboard_shape[1]].T.reshape(-1, 2) * self.square_size
        obj_points_list.append(obj_points)

        img_list = [os.path.join(self.save_path, name) for name in os.listdir(self.save_path)]
        for i in range(len(obj_points_list)):
            if img_list[i].split('.')[-1] != 'png':
                continue
            img = cv2.imread(img_list[i])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_shape, None)
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points_list.append(corners2)
                obj_points_list.append(obj_points)

        # calculate the board2camera matrix
        _ret, _camera_matrix, _dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points_list, img_points_list, gray.shape[::-1], None, None
        )

        b2c_rotation_matrix,_ = cv2.Rodrigues(rvecs[0])
        b2c_translation_vector = tvecs[0].ravel()
        b2c_matrix = np.eye(4)
        b2c_matrix[:3, :3] = b2c_rotation_matrix
        b2c_matrix[:3, 3] = b2c_translation_vector
        self.board2camera = b2c_matrix

        # calculate the robot2camera matrix
        self.robot2camera = self.board2camera@self.robot2board
        print_debug(f'robot2camera: {self.robot2camera}', color_name='COLOR_GREEN')
    
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

    def __mouse_callback(self, event, x, y, flags, param)->None:
        """
        the mouse callback, when the left button is clicked, 
        the clicked point coordinates will be set
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < self.width_left or x >= self.width_left+self.width_middle or y < 0 or y >= self.height:
                # out of the screen
                return
            if self.robot2camera is None:
                print_debug('please callibrate the camera first', color_name='COLOR_RED')
                return
            if self.from_calculated_intrinsics and self.calculated_intrinsics_matrix is None:
                print_debug('please calculate the intrinsics first', color_name='COLOR_RED')
                return
            # get the depth
            x  = x-self.width_left
            depth = self.camera.get_depth_image()[y, x]/1000
            # get the robot coordinates
            if self.from_calculated_intrinsics:
                camera_coords, robot_coords = self.pixel_to_robot(x, y, depth, self.calculated_intrinsics_matrix, self.robot2camera)
            else:
                camera_coords, robot_coords = self.pixel_to_robot(x, y, depth, self.camera.intrinsics_matrix, self.robot2camera)

            # set the clicked point coordinates
            self.click_pixel_coords = (x, y)
            self.click_camera_coords = camera_coords
            self.click_robot_coords = robot_coords


    @staticmethod
    def pixel_to_robot(u, v, Z, intrinsic_matrix, extrinsic_matrix)->None:
        """
        the robot to camera transformation
        
        inputs:
            - u: int, the x coordinate of the point
            - v: int, the y coordinate of the point
            - Z: int, the depth of the point(mm)
            - intrinsic_matrix: np.ndarray (3,3), the intrinsic matrix of the camera
            - extrinsic_matrix: np.ndarray (4,4), the extrinsic matrix of the camera
        
        outputs:
            - camera_coords: np.ndarray (3,), the coordinates of the point in the camera frame
            - robot_coords: np.ndarray (3,), the coordinates of the point in the robot frame
        """
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
        X_c = (u - cx) * Z / fx
        Y_c = (v - cy) * Z / fy
    
        camera_coords = np.array([X_c, Y_c, Z, 1]).reshape(4,1)
        robot_coords = np.linalg.inv(extrinsic_matrix) @ camera_coords

        return camera_coords[:3,0], robot_coords[:3,0]





if __name__ == '__main__':
    auto_callibration = AutoCallibrator(
        pc_id=2,
        camera_position='global',
        chessboard_shape=(5,8),
        square_size=0.02611,
        save_file='./auto_callibration.json',
        window_name='auto_callibration'
    )
    auto_callibration.callibration_loop()