import pyrealsense2 as rs # to control the camera
import numpy as np
import os

import time

import logging

import sys
import logging
import Servoj_RTDE_UR5.rtde.rtde as rtde
import Servoj_RTDE_UR5.rtde.rtde_config as rtde_config
import time
from Servoj_RTDE_UR5.min_jerk_planner_translation import PathPlanTranslation

import serial # 控制夹爪使用的串口通信

from docs.test_6drot import compute_ortho6d_from_rotation_matrix
from docs.test_6drot import convert_euler_to_rotation_matrix

from docs.test_6drot import compute_rotation_matrix_from_ortho6d
from docs.test_6drot import convert_rotation_matrix_to_euler
from docs.test_6drot import compute_ortho6d_from_rotation_matrix    
from docs.test_6drot import convert_euler_to_rotation_matrix

import torch

import cv2
import datetime
from pynput import keyboard

import threading

from .gripper import Gripper

from .debug_decorators import debug_decorator, print_debug
from .camera_manager import CameraManager
# Manager to control the cameras
class Cameras:
    def __init__(self):
        # directory to save the images
        save_dir = os.path.expanduser('/home/ur5/rdtkinova/cam_and_arm/save_image')
        os.makedirs(save_dir, exist_ok=True) 

        # 获取所有连接的设备
        context = rs.context()
        # 所有可连接设备的serial number
        connected_devices = [d.get_info(rs.camera_info.serial_number) for d in context.devices]
        print('Connected devices:', connected_devices)

        # 将serial number映射到index
        self.serial_number2index = {serial_number: index for index, serial_number in enumerate(connected_devices)}
        # 将view映射到serial number
        self.view2serial_number = {
            'global': connected_devices[0],
            'wrist': connected_devices[1]
        }
        # 将serial number映射到view
        self.serial_number2view = {
            connected_devices[0]: 'global',
            connected_devices[1]: 'wrist'
        }
        
        # initialize cameras
        print('Initializing cameras...')
        self.global_camera = Camera(self.view2serial_number['global'])
        self.wrist_camera = Camera(self.view2serial_number['wrist'])
        time.sleep(2.5) # 等待相机初始化完成
        print('done')
        # 将view映射到camera对象
        self.view2camera = {
            'global': self.global_camera,
            'wrist': self.wrist_camera
        }

# camera class
class Camera:
    def __init__(self, serial_number):
        self.serial_number = serial_number

        self.pipeline = rs.pipeline()
        # config = rs.config()
        # config.enable_device(self.serial_number)
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.config = rs.config()
        self.config.enable_device(self.serial_number)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = 0.001
        self.intrinsics = self.get_intrinsics()

        self.robot_tcp = np.zeros(6) # 机器人的tcp传到这里

    def get_intrinsics(self):
        # D435i 的默认内参（你可以根据实际情况修改这些值）
        class RS_Intrinsics:
            def __init__(self):
                self.fx = 386.738  # focal length x
                self.fy = 386.738
                self.ppx = 319.5
                self.ppy = 239.5
        intrinsics = RS_Intrinsics()
        return intrinsics
        
    def get_rgb_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data()) 
        return color_image

    def get_depth_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        return depth_image
    
    def realtime_shoot(self, shoot_button='s', quit_button='q', save_dir='./chessboards/', retries=5):
        """
        实时获取摄像机图像，并且按下特定的按键可以拍照或退出
        """
        threshold = 100 # 图片二值化阈值

        try:
            while True:
                # 定义字体、大小、颜色和粗细
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                color = (255, 255, 255)  # 白色 (B, G, R)
                thickness = 1

                for _ in range(retries):
                    try:
                        frames = self.pipeline.wait_for_frames()
                        aligned_frames = self.align.process(frames)
                        depth_frame = aligned_frames.get_depth_frame()
                        color_frame = aligned_frames.get_color_frame()
                        if not depth_frame or not color_frame:
                            raise RuntimeError("Could not acquire depth or color frame.")
                        depth_image = np.asanyarray(depth_frame.get_data())
                        color_image = np.asanyarray(color_frame.get_data())
                        break
                        
                        # print(f'color image shape: {color_image.shape}')
                        # print(f'depth image shape: {depth_image.shape}')
                        # return color_image, depth_image
                    except RuntimeError as e:
                        print(f"Error: {e}. Retrying...")
                        time.sleep(1)
                else:
                    raise RuntimeError(f"Failed to acquire frames after {retries} retries.")
                # 投射到cv2窗口上面的画面(非保存的画面)
                screen = np.zeros((color_image.shape[0], color_image.shape[1]*2, 3),dtype=np.uint8)
                # 左边部分是原图
                screen[:,:color_image.shape[1],:] = color_image.copy()
                # 实时找到画面的棋盘格
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                # 对图片进行二值化
                gray[gray<threshold] = 0
                gray[gray>=threshold] = 255
                # 将灰度图转换回彩色图(看起来还是黑白的)
                gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                # gray_3channel = color_image.copy()
                gray_3channel_demo = gray_3channel.copy()
                
                pattern_size = (7,7) # 棋盘格大小是8*8
                # pattern_size = (11,8)
                corner_x, corner_y = 0,0
                ret, corners = cv2.findChessboardCornersSB(gray_3channel, pattern_size, None) # 实时找到画面8*8的棋盘格
                if ret:
                    # 如果检测到棋盘，把棋盘标出来
                    cv2.drawChessboardCorners(screen, pattern_size, corners, ret)
                    cv2.drawChessboardCorners(gray_3channel_demo, pattern_size, corners, ret)

                    # 在标出来棋盘的同时，把角点的序数标出来
                    corner_x = round(corners[0][0][0])
                    corner_y = round(corners[0][0][1])
                    try:
                        depth = depth_image[corner_x, corner_y]
                    except IndexError:
                        depth = -1
                    for index_corner, corner in enumerate(corners):
                        corner_position = np.array(corner[0], dtype=np.int32)
                        if index_corner == 0:
                            cv2.putText(gray_3channel_demo, f'{depth:.2f}',corner_position, font, font_scale, (0,0,255), thickness+1)
                        cv2.putText(screen, f'{index_corner}',corner_position, font, font_scale/2, (0,0,255), thickness)


                # 在屏幕右边绘制用于处理的图片
                screen[:,color_image.shape[1]:,:] = gray_3channel_demo

                # 对画面进行一些修饰
                position_line1 = (30, 20)  # (x, y)坐标
                position_line2 = (30, 50)  # (x, y)坐标
                position_line3 = (30, 80)  # (x, y)坐标
                position_line4 = (30, 110)  # (x, y)坐标
                position_line5 = (30, 140)
                position_line6 = (30, 170)
                
                if ret:
                    color = (0,0,255) # 如果找到了棋盘，则字体为红色
                thickness = 1
                cv2.rectangle(screen, (position_line1[0],10), (screen.shape[1]//2,position_line6[-1]), (0,0,0), thickness=-1)
                cv2.putText(screen,f'press {shoot_button} to save image', position_line1, font, font_scale, color, thickness)
                cv2.putText(screen,f'press {quit_button} to quit', position_line2, font, font_scale, color, thickness)
                cv2.putText(screen,f'Bi-value threshold: {threshold}, press p to increase and press m to decrease', position_line3, font, font_scale, color, thickness)
                cv2.putText(screen,f'RGB resolution: {color_image.shape}', position_line4, font, font_scale, color, thickness)
                cv2.putText(screen,f'depth resolution: {depth_image.shape}', position_line5, font, font_scale, color, thickness)
                
                demo_tcp = '[{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}]'.format(
                    self.robot_tcp[0],self.robot_tcp[1],self.robot_tcp[2],self.robot_tcp[3],self.robot_tcp[4],self.robot_tcp[5]
                )
                cv2.putText(screen,f'current tcp: {demo_tcp}, left_up:({corner_x},{corner_y})', position_line6, font, font_scale, color, thickness)
                
                # 实时展示画面
                cv2.imshow('realtime camera', screen)
                key_presseed = cv2.waitKey(1)
                if key_presseed & 0xff==ord('q'):
                    # 按下q就退出
                    break
                elif key_presseed & 0xff==ord('s') and ret:
                    # 按下s就保存
                    current_time = datetime.datetime.now()
                    timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
                    with open(f'{save_dir}/timestamps','a') as f:
                        f.write(f'{timestamp}\n')
                    cv2.imwrite(f'{save_dir}rgb_{timestamp}.png', gray_3channel) # 要保存的是可以识别出棋盘的画面
                    np.save(f'{save_dir}depth_{timestamp}.npy', depth_image)
                    np.save(f'{save_dir}tcp_{timestamp}.npy',self.robot_tcp)
                    print('Image saved!')
                elif key_presseed & 0xff==ord('p'):
                    threshold += 1
                elif key_presseed & 0xff==ord('m'):
                    threshold -= 1
                else:
                    # 其他按键保留
                    continue
        finally:
            cv2.destroyAllWindows()


class RobotEnv:
    def __init__(self,pc_id:int, warm_up:bool=True):
        
         # 初始化真实机械臂环境


        # 连接机器人    
        if pc_id not in [1,2]:
            raise ValueError(f'pc_id must be 1 or 2, but got {pc_id}')
        self.pc_id = pc_id
        if self.pc_id == 2:
            self.ip = '192.168.0.201'
            self.port = 30004
        elif self.pc_id == 1:
            self.ip = '192.168.1.201'
            self.port = 30004

        print('prepare to warm up!!!')
        if warm_up:
            os.system('python warmup.py')

        # 初始化机械臂
        self.robot = ur5Robot(self.ip, self.port)
        
        # 其他初始化代码
        
        # initialize camera
        # self.cameras = Cameras()
        # self.view2camera = self.cameras.view2camera # 将view映射到camera对象

        # self.global_camera = self.view2camera['global']
        # self.wrist_camera = self.view2camera['wrist']
        self.cameras = CameraManager(self.pc_id)
        self.global_camera = self.cameras.get_camera('global')
        self.wrist_camera = self.cameras.get_camera('wrist')

        # 实时将tcp_pose更新
        self.tcp_pose = np.zeros(6)
        def update_tcp_thread():
            while True:
                current_tcp = self.robot.get_tcp_pose()
                self.tcp_pose = current_tcp
                self.global_camera.robot_tcp = current_tcp
                # self.wrist_camera.robot_tcp = current_tcp
                time.sleep(0.1)
        self.tcp_manager_thread = threading.Thread(target=update_tcp_thread)
        self.tcp_manager_thread.daemon = True
        self.tcp_manager_thread.start()

        self.alphas = [np.pi/2,0,0,np.pi/2,-np.pi/2,0]

        pass

    def __del__(self):
        # self.robot.__del__()
        pass
    
    def get_joint_positions(self):
        """
        得到当前机器人系统的各关节角度 in rad
        """
        return np.array(self.robot.get_joint_positions())

    def get_tip_direction(self, local_vector=np.array([0,0,1])):
        """
        得到当前机器人的工具尖端方向向量(norm=1)
        """
        def at2rotmat(alpha,theta):
            """
            根据机器人的alpha和theta得到变换矩阵
            """
            return np.array([
                [np.cos(theta),-np.sin(theta)*np.cos(alpha),np.sin(theta)*np.sin(alpha)],
                [np.sin(theta),np.cos(theta)*np.cos(alpha),-np.cos(theta)*np.sin(alpha)],
                [0,np.sin(alpha),np.cos(alpha)]
            ])
        # 得到当前各关节角度
        six_joints_np = self.get_joint_positions()
        thetas = [theta for theta in six_joints_np]

        # 得到各关节的方向转换矩阵
        Ts = [
            at2rotmat(alpha, theta) for alpha,theta in zip(self.alphas,thetas)
        ]
        transformation_matrix = np.eye(3)
        for T_mat in Ts:
            transformation_matrix = transformation_matrix@T_mat
        
        local_vector_col = np.array([local_vector]).T # 1D to 2D
        rotated_vector = transformation_matrix@local_vector_col
        rotated_vector = rotated_vector[:,0] # 2D to 1D

        return rotated_vector


    def reset(self, seed=None):
        # 重置环境到初始状态

        #当前末端执行器位姿
        tcp_pose=self.robot.get_tcp_pose()
        print(f"Initial tcp_pose: {tcp_pose}")

        #取出tcp_pose的3位末端旋转
        rot_eff=tcp_pose[3:]
        print(f"Initial rot_eff: {rot_eff}")

        #将rot_eff的格式从list转换为np
        rot_eff=np.array(rot_eff)
        print(f"Initial rot_eff np: {rot_eff}")

        #将3位欧拉形式的末端旋转升一维以方便计算
        rot_eff = rot_eff[None, :]    
        print(f"Initial rot_eff up 1 dimension: {rot_eff}")
    
        #转换为旋转矩阵
        rotmat = convert_euler_to_rotation_matrix(rot_eff)
        #转换为ortho6d
        ortho6d = compute_ortho6d_from_rotation_matrix(rotmat)
        print(f"Initial ortho6d up 1 dimension: {ortho6d}")

        #将ortho6d的维度降一维
        ortho6d = ortho6d.squeeze(0)
        print(f"Initial ortho6d down 1 dimension: {ortho6d}")

        #将当前末端执行器位姿和ortho6d合并为target_pose
        target_pose=np.concatenate([tcp_pose[:3], ortho6d])
        print(f"Initial target_pose: {target_pose}")

        #当前夹爪状态
        gripper_state=self.robot.gripper.gripper_state
        print(f"Initial gripper_state: {gripper_state}")

        #将target_pose和gripper_state合并为initial_observation
        initial_observation =np.concatenate([target_pose, np.array([gripper_state])])
        print(f"Initial action: {initial_observation}")

        initial_observation = torch.tensor(initial_observation, dtype=torch.float32)

        # 返回初始观测值和其他可能的初始化信息
        additional_info = None  # 可能的其他信息 (外面的调用不需要)
        
        return initial_observation, additional_info

    def get_state(self):
        # 获取当前状态
        # 返回值: 当前状态
        pass

    def sample_action(self):
        # 采样一个动作
        # 返回值: 动作
        pass
    def move_to_ee_pos(self, target_ee_pos, move_time=3):
        """
        将ee(end effector)移动到某个特定的位姿
        target_ee_pos: dict, 格式为
        {
            "x",
            "y",
            "z",
            "theta_x",
            "theta_y",
            "theta_z"
        }
        """
        target_pose = np.array([
            target_ee_pos['x'],
            target_ee_pos['y'],
            target_ee_pos['z'],
            target_ee_pos['theta_x'],
            target_ee_pos['theta_y'],
            target_ee_pos['theta_z']
        ]) # 将dict转换为array
        self.robot.move(target_pose, move_time)


    def step(self, action, move_time, initial_step=False):
        #输入：action，模型生成的。3位末端位置，6位末端旋转，1位夹爪状态
    

        # 将action拆解为3部分，末端位置，末端旋转（ortho6d），夹爪状态
        pos_eff = action[:3] # 末端位置
        rot_eff = action[3:9] # 末端旋转
        gripper_state = action[9] # 夹爪状态



        #6位末端旋转-》3位末端旋转（UR5要使用）
        ortho6d = rot_eff#仅做测试，代替从模型的输入
        ortho6d = ortho6d[None, :] # 升高维度
        print(f"6D Rotation: {ortho6d}")
        rotmat_recovered = compute_rotation_matrix_from_ortho6d(ortho6d)
        euler_recovered = convert_rotation_matrix_to_euler(rotmat_recovered)
        print(f"Recovered Euler angles: {euler_recovered}")


        # 合并3位末端位置和3位末端旋转为target_pose
        target_pose = np.concatenate([pos_eff, rot_eff])


        # 根据末端执行器位姿（3位末端位置，3位末端旋转），移动机械臂
        if initial_step:
            print("Initial move")
            print(target_pose)
            time.sleep(3)
        print(f"current_pose: {self.robot.get_joint_positions()}")
        self.robot.move(target_pose, move_time)

        # 根据夹爪状态，控制夹爪
        self.robot.gripper_control(gripper_state)


        # # 获取并返回观测值(关节状态proprio)
        # proprio = self.robot.get_joint_positions()
        # 获取tcp位置
        tcp_pose = self.robot.get_tcp_pose()
        tcp_pose = np.array(tcp_pose)
        tcp_xyz = tcp_pose[:3]
        tcp_rpy = tcp_pose[3:]
        gripper_state = self.robot.gripper.gripper_state
        # 将tcp_rpy转换为旋转矩阵   
        tcp_rpy = tcp_rpy[None, :]
        tcp_rotmat = convert_euler_to_rotation_matrix(tcp_rpy)
        # 将旋转矩阵转换为ortho6d
        tcp_6d = compute_ortho6d_from_rotation_matrix(tcp_rotmat)
        # 将tcp_xyz, tcp_6d, gripper_state合并为proprio
        print(f"tcp_xyz: {tcp_xyz}")
        print(f"tcp_6d: {tcp_6d}")
        print(f"gripper_state: {np.array([gripper_state])}")
        proprio = np.concatenate([tcp_xyz, tcp_6d[0],np.array([gripper_state])])
        #返回观测值
        # obs = {
        #     'agent': {
        #         'qpos': None  # 需要根据具体环境实现，应该是一个包含关节位置的数组
        #     }
        # }
        # obs = np.concatenate([proprio.copy(), np.array([gripper_state])])
        obs = proprio.copy()

        #没用，不用管
        reward = 0.0  # 奖励值，需要根据具体环境实现 （没用）
        #没用
        #是否在这一步终止执行
        terminated = False  
        truncated = False  
        # 没用
        #提示是否成功
        #'success'的初始值是False，判断成功后设为True
        info = {
            'success': False  
        }


        # 目前只有obs有用
        return obs, reward, terminated, truncated, info

    def render(self):
        # 渲染当前环境状态

        # 返回值: 一张图像 来自global相机
        return self.global_camera.get_rgb_frame()

    def shootImage(self):
        """
        # 拍摄图像

        # 返回值: 一张图像 np格式 来自global相机
        """
        return self.global_camera.get_rgb_frame()
    
    def get_tcp_pose(self):
        return self.robot.get_tcp_pose()
    
    def realtime_shoot(self):
        def button_detect():
            def on_press(key):
                try:
                    if key.char == 'q':
                        return False
                    elif key.char == 's':
                        print('save image')

                        tcp = self.robot.get_tcp_pose()
                        print(f'current tcp: {tcp}')

                        current_time = datetime.datetime.now()
                        timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
                        np.save(f'./chessboards/tcp_{timestamp}.npy', tcp)
                    else:
                        pass
                except AttributeError:
                    print(f'special key {key} is pressed')
            def on_release(key):
                if key.char == 'q':
                    # Stop listener
                    print('q released')
                    return False

            # Collect events until released
            with keyboard.Listener(
                    on_press=on_press,
                    on_release=on_release) as listener:
                listener.join()

        shoot_thread = threading.Thread(target=button_detect)
        shoot_thread.daemon = True # 设定为守护进程
        # shoot_thread.start()
        self.cameras.global_camera.realtime_shoot()
        # self.cameras.wrist_camera.realtime_shoot()

class ur5Robot:
    def __init__(self, ip='192.168.1.201', port=30004, FREQUENCY=500,
                 config_filename='Servoj_RTDE_UR5/control_loop_configuration.xml'):
        # UR5机械臂的基本参数
        

        # 连接机器人
        self.ip = ip
        self.port = port

        # 夹爪
        self.gripper = Gripper(2)

        
        #?没用
        self.joint_num = 6  # UR5有6个关节

        # 当前关节位置  
        self.current_joint_positions = None
        # 当前末端执行器位姿
        self.current_tcp_pose = None

        
        # 频率
        self.FREQUENCY = FREQUENCY

        # 日志
        logging.getLogger().setLevel(logging.INFO)

        # 读取配置文件
        conf = rtde_config.ConfigFile(config_filename)
        self.state_names, self.state_types = conf.get_recipe('state')
        self.setp_names, self.setp_types = conf.get_recipe('setp')
        self.watchdog_names, self.watchdog_types = conf.get_recipe('watchdog')

        # 连接状态
        self.__connect()
        # self.is_connected = False  
         
    def __del__(self):
        self.gripper.__del__()
        print('robot destroyed')


    def __connect(self):
        # 连接
        self.con = rtde.RTDE(self.ip, self.port)
        connection_state = self.con.connect()
        while connection_state != 0:
            time.sleep(0.5)
            connection_state = self.con.connect()
        print("---------------Successfully connected to the robot-------------\n")

        #通信
        self.con.get_controller_version()
        self.con.send_output_setup(self.state_names, self.state_types)  
        self.setp = self.con.send_input_setup(self.setp_names, self.setp_types)
        self.watchdog = self.con.send_input_setup(self.watchdog_names, self.watchdog_types)

        #初始化寄存器
        self.setp.input_double_register_0 = 0
        self.setp.input_double_register_1 = 0
        self.setp.input_double_register_2 = 0
        self.setp.input_double_register_3 = 0
        self.setp.input_double_register_4 = 0
        self.setp.input_double_register_5 = 0
        self.setp.input_bit_registers0_to_31 = 0

        #连不上就退出
        if not self.con.send_start():
            sys.exit() 
        
        self.is_connected = True

        # 创建一个单独的线程, 每10s与看门狗对话一次
        self.last_time_comunicate = time.time() - 20
        # self.home = [-0.503, -0.0088, 0.31397, 1.266, -2.572, -0.049]
        def communicate():
            while True:
                if time.time() - self.last_time_comunicate > 5:
                    # 此时触发看门狗
                    print('watchdog triggered')

                    # 获取当前位置
                    curr_pose = self.get_tcp_pose()

                    # 自己移动到自己的位置
                    self.move(curr_pose,0.1)
                    
                    # 更新时钟
                    self.last_time_comunicate = time.time()
                time.sleep(1)
        watchdog_thread = threading.Thread(target=communicate)
        watchdog_thread.daemon = True # 守护进程
        watchdog_thread.start() # 开启该进程



    def move(self, target_pose, trajectory_time=0.1):
        """
        笛卡尔空间直线运动
        Args:
            target_pose: 目标位姿 [x, y, z, rx, ry, rz]
            trajectory_time: 运动时间 (s)

            speed: 运动速度 (m/s)
            acceleration: 加速度 (m/s^2)
        """
        if not self.is_connected:
            self.__connect()

        #获取当前末端执行器位姿
        state = self.con.receive()
        tcp = state.actual_TCP_pose
        orientation_const = tcp[3:]

        #打印
        print("Current TCP pose:", tcp)
        print("-------Executing servoJ to point 1 -----------\n")
                
        #指定伺服模式
        self.watchdog.input_int_register_0 = 2
        self.con.send(self.watchdog)
        state = self.con.receive()
        print("Watchdog sent")
        # time.sleep(3)

        # 路径规划器
        planner = PathPlanTranslation(tcp, target_pose, trajectory_time)

        # 动
        t_start = time.time()
        while time.time() - t_start < trajectory_time:
            state = self.con.receive()
            # print(f'state. runtime_state: {state.runtime_state}')
            t_current = time.time() - t_start

            if state.runtime_state > 1 and t_current <= trajectory_time:
            # if t_current <= trajectory_time:
                position_ref, lin_vel_ref, acceleration_ref = planner.trajectory_planning(t_current)
                pose = position_ref.tolist() + orientation_const
                self.list_to_setp(self.setp, pose)
                # print(f"Sending setp: {pose}")
                self.con.send(self.setp)

        # 打印
        print(f"It took {time.time()-t_start}s to execute the servoJ to point 1")
        print('Final TCP pose:', self.con.receive().actual_TCP_pose)

        # 更新上次通信时间
        self.last_time_comunicate = time.time()
        # self.stop()

        pass
    
    @staticmethod
    def list_to_setp(setp, list):
        """
        将列表转换为setp, 输入的setp是自己的setp属性
        """
        for i in range(6):
            setp.__dict__[f"input_double_register_{i}"] = list[i]
        return setp

    def stop(self):
        """
        立即停止机械臂运动
        """

        self.con.send_pause()
        self.con.disconnect()

        pass


    def gripper_control(self, gripper_state):
        """
        控制夹爪 1=grasp(半夹), 0=close, -1=open/release
        """
        dist_1 = abs(gripper_state - 1)
        dist_0 = abs(gripper_state - 0)
        dist_b1 = abs(gripper_state + 1)
        min_dist = min([dist_1, dist_0, dist_b1])
        if min_dist==dist_1:
            self.gripper.grasp()
        elif min_dist==dist_0:
            self.gripper.close()
        elif min_dist==dist_b1:
            self.gripper.open()
        else:
            print(f"Invalid gripper state:{gripper_state}")

#？？这个类下面的其他方法有没有用？

    # def move_j(self, joint_positions, speed=1.0, acceleration=1.0):
    #     """
    #     关节空间运动
    #     Args:
    #         joint_positions: 目标关节角度 [j1, j2, j3, j4, j5, j6]
    #         speed: 运动速度比例 (0.0-1.0)
    #         acceleration: 加速度比例 (0.0-1.0)
    #     """
    #     pass



    # def move_p(self, target_pose, speed=0.25, acceleration=0.5, blend_radius=0.05):
    #     """
    #     笛卡尔空间点到点运动（带圆弧过渡）
    #     Args:
    #         target_pose: 目标位姿 [x, y, z, rx, ry, rz]
    #         speed: 运动速度 (m/s)
    #         acceleration: 加速度 (m/s^2)
    #         blend_radius: 圆弧过渡半径 (m)
    #     """
    #     pass

    def get_joint_positions(self):
        """
        获取当前关节角度
        Returns:
            list: 当前关节角度 [j1, j2, j3, j4, j5, j6]
        """
        if not self.is_connected:
            self.__connect()
        state = self.con.receive() # 获取当前状态
        current_joint_positions = state.actual_q # 获取当前关节角度
        return current_joint_positions

    def get_tcp_pose(self):
        """
        获取当前末端执行器位姿
        Returns:
            list: 当前TCP位姿 [x, y, z, rx, ry, rz]
        """
        if not self.is_connected:
            self.__connect()
        state = self.con.receive() # 获取当前状态
        current_tcp_pose = state.actual_TCP_pose # 获取当前末端执行器位姿
        return current_tcp_pose

    def control_gripper(self, open_state, speed=1.0):
        """
        控制夹爪
        Args:
            open_state: 0.0表示打开，1.0表示关闭
            speed: 夹爪速度 (0-1)
        """
        prev_state = self.gripper_state # 获取当前夹爪状态
        self.gripper_state = open_state # 设置夹爪新状态

        # 夹爪状态发生变化时，发送命令
        if prev_state != self.gripper_state:
            if self.gripper_state == 0.0:
                self.ser.write(self.MOTOR_OPEN_LIST)
            else:
                self.ser.write(self.MOTOR_CLOSE_LIST)


    def set_tcp(self, tcp_offset):
        """
        设置工具中心点偏移
        Args:
            tcp_offset: TCP偏移量 [x, y, z, rx, ry, rz]
        """
        pass

    def set_payload(self, weight, cog=None):
        """
        设置负载
        Args:
            weight: 负载重量(kg)
            cog: 重心位置 [x, y, z]
        """
        pass

# class Gripper:
#     def __init__(self, port='/dev/ttyUSB0', baudrate=9600, timeout=1):
#         self.ser = serial.Serial(port, baudrate, timeout=timeout) # control the gripper

#         # 夹爪速度
#         self.MAX_SPEED_42 = (0X01,0X40) # 最快速度为42rad/s， 也就是01A0
#         self.AVER_SPEED_10 = (0X00,0XC8) # 平均速度为10rad/s， 也就是00C8

#         # 夹爪打开/关闭的命令
#         self.BYTE_OPEN = 0x00
#         self.BYTE_CLOSE = 0x01
#         # self.MOTOR_OPEN_LIST = (0x02,self.BYTE_OPEN,0x20,0x49,0x20,*self.AVER_SPEED_10) #机械爪松开(最快速度为42rad/s， 也就是01A0)
#         # self.MOTOR_CLOSE_LIST = (0x02,self.BYTE_CLOSE,0x20,0x49,0x20,*self.AVER_SPEED_10) #机械爪闭合

#         # 机械爪90%张开闭合
#         self.MOTOR_OPEN_LIST = (0x02,self.BYTE_OPEN,0x20,0x43,0x14,*self.AVER_SPEED_10) #机械爪松开(最快速度为42rad/s， 也就是01A0)
#         self.MOTOR_CLOSE_LIST = (0x02,self.BYTE_CLOSE,0x20,0x43,0x14,*self.AVER_SPEED_10) #机械爪闭合
        
#         # 夹爪状态
#         self.gripper_state = 0.0 # 0.0表示夹爪打开，1.0表示夹爪关闭
#         # 工作状态
#         self.working_state = 'ready' # 工作状态：ready表示待机，working表示工作
#         # 任务状态
#         self.task_start = None # 任务开始的闭合程度
#         self.task_end = None # 任务计划结束的闭合程度

#         # 任务时间管理
#         self.task_start_time = None # 任务开始时间

#         # 默认先打开夹爪
#         self.ser.write(self.MOTOR_OPEN_LIST)
#         # 等待夹爪打开
#         time.sleep(2)
#         print("Gripper initialized")
    
#     def __del__(self):
#         print('gripper destroyed')
#         self.ser.write(self.MOTOR_CLOSE_LIST)
#         self.ser.close()

#     @staticmethod
#     def rad2byte(rad):
#         """
#         将角度转换为字节
#         输出为高低位的10进制
#         """
#         output_dec = round(rad/3.14*180*10)
#         output_hex = format(output_dec, '04X') # 长度为4字符，使用0填充，大写16进制
#         high_byte = output_hex[:2]
#         low_byte = output_hex[2:]
#         return int(high_byte, 16), int(low_byte, 16)
#     @staticmethod
#     def gripper_state2rad(state):
#         """
#         将夹爪状态转换为弧度
#         """
#         return state * 3.14*2*5.2

#     def open(self):
#         self.__update_gripper_state()
#         if self.gripper_state == 1.0: # 如果夹爪状态为1.0，则夹爪已经闭合

#             if self.working_state == 'ready': # 如果工作状态为ready，则夹爪可以工作
#                 self.working_state = 'working'
#                 self.task_start = self.gripper_state
#                 self.task_end = 0.0
#                 self.task_start_time = time.time() # 任务开始时间
#                 print('opening')
#                 # self.ser.write(self.MOTOR_OPEN_LIST)
#                 self.__open_byte(self.rad2byte(self.gripper_state2rad(0.8)))
#                 # time.sleep(2)
#             else:
#                 return
#         elif self.gripper_state == 0.5:
#             if self.working_state == 'ready': # 如果工作状态为ready，则夹爪可以工作
#                 self.working_state = 'working'
#                 self.task_start = self.gripper_state
#                 self.task_end = 0.0
#                 self.task_start_time = time.time() # 任务开始时间
#                 print('opening')
#                 # self.ser.write(self.MOTOR_OPEN_LIST)
#                 self.__open_byte(self.rad2byte(self.gripper_state2rad(0.4)),speedType='AVER')
#                 # time.sleep(2)
#             else:
#                 return
#         else:
#             # 如果夹爪状态为0.0，则夹爪已经打开
#             print('gripper is already opened')
#             return
        
#     def grasp(self):
#         self.__update_gripper_state()
#         if self.gripper_state == 0.0: # 如果夹爪状态为0.0，则夹爪已经打开
#             if self.working_state == 'ready': # 如果工作状态为ready，则夹爪可以工作
#                 self.working_state = 'working'
#                 self.task_start = self.gripper_state
#                 self.task_end = 0.5
#                 self.task_start_time = time.time() # 任务开始时间
#                 print('grasping from open')
#                 # self.ser.write(self.MOTOR_OPEN_LIST)
#                 self.__close_rad(self.gripper_state2rad(0.4),speedType='AVER')
#                 # time.sleep(2)
#             else:
#                 return
#         elif self.gripper_state == 1.0:
#             if self.working_state == 'ready':
#                 self.working_state = 'working'
#                 self.task_start = self.gripper_state
#                 self.task_end = 0.5
#                 self.task_start_time = time.time() # 任务开始时间
#                 print('grasping from close')
#                 # self.ser.write(self.MOTOR_OPEN_LIST)
#                 self.__open_rad(self.gripper_state2rad(0.4),speedType='AVER')
#                 # time.sleep(2)
#             else:
#                 return
#         else:
#             # 如果夹爪状态为0.0，则夹爪已经打开
#             return


#     def close(self):
#         self.__update_gripper_state()
#         if self.gripper_state == 0.0: # 如果夹爪状态为1.0，则夹爪已经闭合
#             if self.working_state == 'ready': # 如果工作状态为ready，则夹爪可以工作
#                 self.working_state = 'working'
#                 self.task_start = self.gripper_state
#                 self.task_end = 1.0
#                 self.task_start_time = time.time() # 任务开始时间
#                 print('closing')
#                 # self.ser.write(self.MOTOR_CLOSE_LIST)
#                 self.__close_byte(self.rad2byte(self.gripper_state2rad(0.8)))
#                 # time.sleep(2)
#             else:
#                 return
#         elif self.gripper_state == 0.5:
#             if self.working_state == 'ready': # 如果工作状态为ready，则夹爪可以工作
#                 self.working_state = 'working'
#                 self.task_start = self.gripper_state
#                 self.task_end = 1.0
#                 self.task_start_time = time.time() # 任务开始时间
#                 print('closing')
#                 # self.ser.write(self.MOTOR_CLOSE_LIST)
#                 self.__close_byte(self.rad2byte(self.gripper_state2rad(0.4)),speedType='AVER')
#                 # time.sleep(2)
#             else:
#                 return
#         else:
#             print('gripper is already closed')
#             return
        
#     def __update_gripper_state(self):
#         """
#         更新夹爪状态
#         """
#         current_time = time.time()
#         if self.task_start_time is not None:
#             if current_time - self.task_start_time >= 3:
#                 self.working_state = 'ready'
#                 if self.task_end == 0.0:
#                     self.gripper_state = 0.0
#                 elif self.task_end == 0.5:
#                     self.gripper_state = 0.5
#                 else:
#                     self.gripper_state = 1.0
#                 self.task_start = None
#                 self.task_end = None
#                 self.task_start_time = None
        
    
#     def __close_rad(self, rad,speedType='MAX'):
#         """
#         闭合指定角度
#         Args:
#             rad: 角度
#             speedType: 速度类型，'MAX'表示最大速度，'AVER'表示平均速度
#         """
#         high_byte, low_byte = self.rad2byte(rad)    
#         if speedType == 'MAX':
#             self.ser.write((0x02,self.BYTE_CLOSE,0x20,high_byte,low_byte,*self.MAX_SPEED_42))
#         elif speedType == 'AVER':
#             self.ser.write((0x02,self.BYTE_CLOSE,0x20,high_byte,low_byte,*self.AVER_SPEED_10))

#     def __open_rad(self, rad,speedType='MAX'):
#         """
#         打开指定角度
#         Args:
#             rad: 角度
#             speedType: 速度类型，'MAX'表示最大速度，'AVER'表示平均速度
#         """
#         high_byte, low_byte = self.rad2byte(rad)
#         if speedType == 'MAX':
#             self.ser.write((0x02,self.BYTE_OPEN,0x20,high_byte,low_byte,*self.MAX_SPEED_42))
#         elif speedType == 'AVER':
#             self.ser.write((0x02,self.BYTE_OPEN,0x20,high_byte,low_byte,*self.AVER_SPEED_10))

#     def __close_byte(self, bytes,speedType='MAX'):
#         """
#         闭合指定字节
#         Args:
#             bytes: 字节
#             speedType: 速度类型，'MAX'表示最大速度，'AVER'表示平均速度
#         """
#         if speedType == 'MAX':
#             self.ser.write((0x02,self.BYTE_CLOSE,0x20,*bytes,*self.MAX_SPEED_42))
#         elif speedType == 'AVER':
#             self.ser.write((0x02,self.BYTE_CLOSE,0x20,*bytes,*self.AVER_SPEED_10))

#     def __open_byte(self, bytes,speedType='MAX'):
#         """
#         打开指定字节
#         Args:
#             bytes: 字节
#             speedType: 速度类型，'MAX'表示最大速度，'AVER'表示平均速度
#         """
#         if speedType == 'MAX':
#             self.ser.write((0x02,self.BYTE_OPEN,0x20,*bytes,*self.MAX_SPEED_42))
#         elif speedType == 'AVER':
#             self.ser.write((0x02,self.BYTE_OPEN,0x20,*bytes,*self.AVER_SPEED_10))



if __name__ == "__main__":
    # setp2 = [-0.077, -0.636, 0.541, 2.778, -0.994, 0.047]
    # setp1 = [-0.553, -0.0188, 0.36397, 1.266, -2.572, -0.049]
    # setp1 = [-0.077, -0.636, 0.541, 0,0,0,0,0,0,1]
    # setp2 = [-0.553, -0.0188, 0.36397, 0,0,0,0,0,0,0]
    # setp3 = [-0.07701603, -0.63599335, 0.54099801, 0.54088514,
    #           0.03234404, 0.84047435, -0.23676042, -0.95299949, 0.18904093, 0]

    # setp1 = np.array(setp1)
    # setp2 = np.array(setp2)
    # setp3 = np.array(setp3)

    # action = [-0.503, -0.0088, 0.31397, -0.84110996 ,0.0412474 , 0.53928906 ,-0.49911578 ,0.32493579 ,-0.80330577,1]
    # action = np.array(action)+0.03
    # # observed_state, _reward, terminated, truncated, info = robotEnv.step(action)
    # # robot = ur5Robot('192.168.0.201')
    # # robot.move(setp1,trajectory_time=8)
    # # robot.move(setp2,trajectory_time=8)
    # robot_environment = RobotEnv()
    # robot_environment.reset()
    # # robot_environment.step(action)
    # robot_environment.step(setp1, 3)

    camera_manager = Cameras()
    print(camera_manager.wrist_camera)

    
    # robot = ur5Robot('192.168.0.201')
    # robot.move(np.array([-1.6307831e-04, -8.9645386e-05, -6.9618225e-05, -5.8174133e-05,
    #             -9.1075897e-05,  2.4795532e-05,  1.4495850e-04, -3.2615662e-04,
    #             -1.5068054e-04]))

    # target_pose = [-0.553, -0.0188, 0.36397, 1.266, -2.572, -0.049]
    # robot.move(target_pose,trajectory_time=8)


    # robot = ur5Robot('192.168.0.201')
    # target_pose = [-0.503, -0.0088, 0.31397, 1.266, -2.572, -0.049]
    # robot.move(target_pose,trajectory_time=8)

    # gripper = Gripper()
    # print(gripper.rad2byte(5.3*3.14*2*0.9))
    # time.sleep(5)
    # gripper.grasp()
    # print(0)


    # # high_byte, low_byte = gripper.rad2byte(1800)
    # time.sleep(5)
    # gripper.close()
    # print(1)
    # time.sleep(5)
    # gripper.open()
    # print(2)
    # time.sleep(5)
    # gripper.close()
    # print(3)
    # time.sleep(5)
    # gripper.open()
    # print(4)
    # time.sleep(5)

    # for i in range(10):
    #     gripper.close()
    #     time.sleep(5)
    #     gripper.grasp()
    #     time.sleep(5)
    #     gripper.open()
    #     time.sleep(5)
    #     gripper.grasp()
    #     time.sleep(5)
    #     gripper.close()
    #     time.sleep(5)

    # target_pose = [-0.503, -0.0088, 0.31397, 1.266, -2.572, -0.049]
    # robot.move(target_pose,trajectory_time=8)
