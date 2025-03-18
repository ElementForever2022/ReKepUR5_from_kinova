# the code to control the UR5 robot

# import the necessary packages
import logging # for logging
import pathlib # for file path manipulation
import time # for timing and sleeping
import sys # for system exit
import threading # for multi-threading

# import the necessary modules
import Servoj_RTDE_UR5.rtde.rtde as rtde
import Servoj_RTDE_UR5.rtde.rtde_config as rtde_config
from debug_decorators import debug_decorator, print_debug

class UR5Robot:
    """
    the class to control the UR5 robot
    """
    @debug_decorator(
        head_message='Initializing the UR5 robot...',
        tail_message='UR5 robot initialized!',
        color_name='COLOR_YELLOW',
        bold=True
    )
    def __init__(self, pc_id:int=0):
        """
        initialize the UR5 robot
        """
        if pc_id not in [1, 2]:
            raise ValueError(f'pc_id must be 1 or 2, but got {pc_id}')
        # set up the robot connection configuration
        self.pc_id: int = pc_id # the id of the pc(1 or 2)
        self.robot_host: str = '192.168.1.201' if pc_id == 1 else '192.168.0.201' # the host(ip address  ) of the robot
        self.robot_port: int = 30004 # the port of the robot
        self.robot_connection = None
        print_debug(f'UR5 Robot Configuration:')
        print_debug(f'    PC: {self.pc_id}, Host: {self.robot_host}, Port: {self.robot_port}')

        # set up the robot control configuration
        self.FREQUENCY=500
        self.config_filename=pathlib.Path(__file__).parent/'Servoj_RTDE_UR5/control_loop_configuration.xml'
        self.joint_num = 6

        # set up the logging
        logging.getLogger().setLevel(logging.INFO)

        # read the configuration file
        conf = rtde_config.ConfigFile(self.config_filename)
        self.state_names, self.state_types = conf.get_recipe('state')
        self.setp_names, self.setp_types = conf.get_recipe('setp')
        self.watchdog_names, self.watchdog_types = conf.get_recipe('watchdog')

        # connect to the robot
        self.is_connected = False
        self.__connect()


    @debug_decorator(
        head_message='Connecting to the robot...',
        tail_message='Connected to the robot!',
        color_name='COLOR_YELLOW',
        bold=True
    )
    def __connect(self):
        """
        connect to the robot
        """
        self.robot_connection = rtde.RTDE(self.robot_host, self.robot_port)
        connection_state = self.robot_connection.connect()
        while connection_state != 0:
            time.sleep(0.5)
            connection_state = self.robot_connection.connect()
        print_debug(f'Connection State: {connection_state}')


        # initialize the robot
        self.robot_connection.get_controller_version()
        self.robot_connection.send_output_setup(self.state_names, self.state_types)
        self.setp = self.robot_connection.send_input_setup(self.setp_names, self.setp_types)
        self.watchdog = self.robot_connection.send_input_setup(self.watchdog_names, self.watchdog_types)


        # initialize the robot registers
        self.setp.input_double_register_0 = 0
        self.setp.input_double_register_1 = 0
        self.setp.input_double_register_2 = 0
        self.setp.input_double_register_3 = 0
        self.setp.input_double_register_4 = 0
        self.setp.input_double_register_5 = 0
        self.setp.input_bit_registers0_to_31 = 0

        # start the robot
        if not self.robot_connection.send_start():
            # if the robot is not started, exit the program
            sys.exit()

        # set up the watchdog
        self.last_time_comunicate = time.time() - 20
        def communicate():
            while True:
                if time.time() - self.last_time_comunicate > 5:
                    # watchdog triggered
                    curr_pose = self.get_tcp_pose()
                    self.move(curr_pose, 0.1)
                    self.last_time_comunicate = time.time()
                time.sleep(1)
        watchdog_thread = threading.Thread(target=communicate)
        watchdog_thread.daemon = True # daemon thread
        watchdog_thread.start() # start the thread

        # set the robot connection state
        self.is_connected = True




if __name__ == '__main__':
    ur5_robot = UR5Robot(pc_id=2)
    print(ur5_robot.state_names)
    print(ur5_robot.state_types)
    print(ur5_robot.setp_names)
    print(ur5_robot.setp_types)
    print(ur5_robot.watchdog_names)
    print(ur5_robot.watchdog_types)
