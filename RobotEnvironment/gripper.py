# to control the gripper

# import the necessary libraries
import time # for time
import threading # for multi-threading
import serial # for serial communication
import os # for file operation
import pathlib # for file path

# import necessary modules
from motor_controller import MotorController # for motor control
from debug_decorators import print_debug,debug_decorator # for debug

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
        self.BYTE_OPEN = 0x00
        self.BYTE_CLOSE = 0x01

        # gripper state
        self.MOTOR_OPEN_LIST = (0x02,self.BYTE_OPEN,0x20,0x43,0x14,*self.AVER_SPEED_10) # release the gripper
        self.MOTOR_CLOSE_LIST = (0x02,self.BYTE_CLOSE,0x20,0x43,0x14,*self.AVER_SPEED_10) # close the gripper

        # gripper state
        # self.gripper_state = 'OPEN' # enum: OPEN or CLOSE

        # begin to update the realtime property
        self.ready = True
        self.state_file = 'gripper_state'
        self.save_path = pathlib.Path(__file__).parent.resolve()/self.state_file
        with open(self.save_path, 'r') as f:
            self.task_end = f.read() # the start state of the task


    @staticmethod
    def rad2byte(rad):
        """
        convert the angle to the byte
        output as the high and low bits
        """
        output_dec = round(rad/3.14*180*10)
        output_hex = format(output_dec, '04X') # 长度为4字符，使用0填充，大写16进制
        high_byte = output_hex[:2]
        low_byte = output_hex[2:]
        return int(high_byte, 16), int(low_byte, 16)
    @staticmethod
    def gripper_state2rad(state):
        """
        convert the gripper state to the rad
        """
        return state * 3.14*2*5.2

    def update(self):
        """
        implement the update method
        update the realtime property of the gripper
        """
        self['gripper_state'] = self.task_end
        with open('gripper_state', 'w') as f:
            f.write(self.task_end)


    def __open_byte(self, bytes,speedType='MAX'):
        """
        open the gripper for the given bytes and speed type
        """
        if speedType == 'MAX':
            self.ser.write((0x02,self.BYTE_OPEN,0x20,*bytes,*self.MAX_SPEED_42))
        elif speedType == 'AVER':
            self.ser.write((0x02,self.BYTE_OPEN,0x20,*bytes,*self.AVER_SPEED_10))
    
    def __close_byte(self, bytes,speedType='MAX'):
        """
        close the gripper for the given bytes and speed type
        """
        if speedType == 'MAX':
            self.ser.write((0x02,self.BYTE_CLOSE,0x20,*bytes,*self.MAX_SPEED_42))
        elif speedType == 'AVER':
            self.ser.write((0x02,self.BYTE_CLOSE,0x20,*bytes,*self.AVER_SPEED_10))
    
    def open(self):
        """
        open the gripper
        """
        if self['gripper_state'] == 'CLOSE': # if the gripper is close

            if self.ready: # if the working state is ready, the gripper can work
                self.task_start = self['gripper_state']
                self.task_end = 'OPEN'
                self.task_start_time = time.time() # the start time of the task
                self.__open_byte(self.rad2byte(self.gripper_state2rad(0.8)))
                print_debug('the gripper is opened')
            else:
                return
        elif self['gripper_state'] == 'HALF':
            if self.ready: # if the working state is ready, the gripper can work
                self.task_start = self['gripper_state']
                self.task_end = 'OPEN'
                self.task_start_time = time.time() # the start time of the task
                self.__open_byte(self.rad2byte(self.gripper_state2rad(0.4)),speedType='AVER')
                print_debug('the gripper is opened')
            else:
                return
        else:
            return
        
    
    def close(self):
        """
        close the gripper
        """
        if self['gripper_state'] == 'OPEN': # if the gripper is open

            if self.ready: # if the working state is ready, the gripper can work
                self.task_start = self['gripper_state']
                self.task_end = 'CLOSE'
                self.task_start_time = time.time() # the start time of the task
                self.__close_byte(self.rad2byte(self.gripper_state2rad(0.8)))
                print_debug('the gripper is closed')
            else:
                return
        elif self['gripper_state'] == 'HALF':
            if self.ready: # if the working state is ready, the gripper can work
                # open and close the gripper
                self.open()
                time.sleep(2)
                self.close()
            else:
                return
        else:
            return
        
    def grasp(self):
        """
        grasp the object
        """
        if self['gripper_state'] == 'OPEN':
            if self.ready:
                self.task_start = self['gripper_state']
                self.task_end = 'HALF'
                self.task_start_time = time.time() # the start time of the task
                self.__close_byte(self.rad2byte(self.gripper_state2rad(0.4)))
                print_debug('the gripper is grasped')
            else:
                return
        elif self['gripper_state'] == 'CLOSE':
            if self.ready:
                # open and grasp the gripper
                self.open()
                time.sleep(2)
                self.grasp()
            else:
                return
        else:
            return


if __name__ == '__main__':
    gripper = Gripper(2)
    time.sleep(3)
    gripper.close()
    time.sleep(3)
    gripper.open()
    time.sleep(3)
    gripper.open()
    time.sleep(3)
    gripper.close()
    time.sleep(3)
    gripper.grasp()
    time.sleep(3)
    gripper.close()
    time.sleep(3)