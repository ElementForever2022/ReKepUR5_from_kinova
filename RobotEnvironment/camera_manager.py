"""
define a class of camera manager that manages multiple cameras
"""

# import necessary libs
from realsense_camera import RealsenseCamera # to use cameras
import pandas as pd # read camera csv
import pyrealsense2 as rs # official lib of RealSense Camneras

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
    def __init__(self, pc_id:int):
        """
        Manage multiple cameras according to the id of pc

        inputs:
            pc_id:int, must be 1 or 2
        """
        if pc_id != 1 and pc_id!= 2:
            raise Exception('ID of pc must be 1 or 2!!!')
        
        # load registered_camera.csv to get
        # bi-directional relationship of serial number and device information
        self.camera_register_path = './registered_camera.csv'
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
        print(self.possible_camera_names)
        print(self.possible_camera_ids)
        # get camera names connected to the pc
        self.connected_camera_ids = [camera_id for camera_id in self.possible_camera_ids if camera_id in connected_devices]
        self.connected_camera_names = [self.id2name_dict[camera_id] for camera_id in self.connected_camera_ids]
        print(self.connected_camera_ids)
        print(self.connected_camera_names)




if __name__ == '__main__':
    # df  = pd.read_csv('registered_camera.csv')
    # print(df)
    # all connected cameras
    # context = rs.context()
    # # serial numbers of all connected cameras
    # connected_devices = [d.get_info(rs.camera_info.serial_number) for d in context.devices]
    # print('Connected devices:', connected_devices)
    cm = CameraManager(2)