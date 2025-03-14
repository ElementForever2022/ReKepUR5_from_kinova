"""
define a class of camera manager that manages multiple cameras
"""

# import necessary libs
from realsense_camera import RealsenseCamera # to use cameras
import pandas as pd # read camera csv

class CameraManager(object):
    """
    Camera Manager
    """
    
    def __init__(self, pc_id:int):
        """
        Manage multiple cameras according to the id of pc

        inputs:
            pc_id:int, must be 1 or 2
        """
        if pc_id != 1 and pc_id!= 2:
            raise Exception('ID of pc must be 1 or 2!!!')
        


if __name__ == '__main__':
    df  = pd.read_csv('registered_camera.csv')
    print(df)