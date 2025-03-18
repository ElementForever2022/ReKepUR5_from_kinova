# to control the gripper

# import the necessary libraries
import time # for time
import threading # for multi-threading
import serial # for serial communication

# import necessary modules
from motor_controller import MotorController # for motor control

class Gripper(MotorController):
    """
    a class to control the gripper
    """
    def __init__(self, pc_id: int):
        """
        initialize the gripper

        inputs:
            pc_id: int, the id of the pc [1,2]
        """
        super().__init__(pc_id)
        self._add_realtime_properties(['gripper_state'], [0.0])

        self.ser = serial.Serial(port='/dev/ttyUSB0', baudrate=9600, timeout=1)

        # constants
        self.MAX_SPEED_42 = (0X01,0X40) # max speed 42rad/s
        self.AVER_SPEED_10 = (0X00,0XC8) # average speed 10rad/s

        # gripper state
        self.MOTOR_OPEN_LIST = (0x02,self.BYTE_OPEN,0x20,0x43,0x14,*self.AVER_SPEED_10) # release the gripper
        self.MOTOR_CLOSE_LIST = (0x02,self.BYTE_CLOSE,0x20,0x43,0x14,*self.AVER_SPEED_10) # close the gripper

        # gripper state
        self.gripper_state = 'open' # enum: open or close

        # begin to update the realtime property
        self.ready = True

    def update(self):
        """
        implement the update method
        update the realtime property of the gripper
        """
        self['gripper_state'] = 1.0

if __name__ == '__main__':
    gripper = Gripper(1)
    time.sleep(10)