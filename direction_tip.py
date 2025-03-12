import robotEnv

import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

from datetime import datetime

gripper_length = 1
local_tool_vec = np.array([[0],[0],[gripper_length]])

def rotmatx(rx):
    return np.array([[1, 0, 0],
                     [0, np.cos(rx), -np.sin(rx)],
                     [0, np.sin(rx), np.cos(rx)]])
def rotmaty(ry):
    return np.array([[np.cos(ry), 0, np.sin(ry)],
                     [0, 1, 0],
                     [-np.sin(ry), 0, np.cos(ry)]])
def rotmatz(rz):
    return np.array([[np.cos(rz), -np.sin(rz), 0],
                     [np.sin(rz), np.cos(rz), 0],
                     [0, 0, 1]])
def rxryrz2mat(rxryrz,s='xyz'):
    ret_mat = np.eye(3)
    for axis in s[::-1]:
        if axis=='x':
            ret_mat = rotmatx(rxryrz[0])@ret_mat
        elif axis=='y':
            ret_mat = rotmaty(rxryrz[1])@ret_mat
        elif axis=='z':
            ret_mat = rotmatz(rxryrz[2])@ret_mat
    return ret_mat

def rxryrz2xyz_self(rxryrz:np.ndarray,s='xyz'):

    # rot = R.from_euler(s,rxryrz)
    # rotation_matrix = rot.as_matrix()
    # rotation_matrix = np.linalg.inv(rotation_matrix)
    rotation_matrix = rxryrz2mat(rxryrz,s)

    rotated_vector = rotation_matrix@local_tool_vec # 目前为列向量

    rotated_vector = rotated_vector[:,0] # 2D to 1D

    return rotated_vector
def rxryrz2xyz_scipy(rxryrz:np.ndarray,s='xyz'):
    rot = R.from_euler(s,rxryrz)
    rotation_matrix = rot.as_matrix()
    # rotation_matrix = np.linalg.inv(rotation_matrix)
    # rotation_matrix = rxryrz2mat(rxryrz,s)

    rotated_vector = rotation_matrix@local_tool_vec # 目前为列向量

    rotated_vector = rotated_vector[:,0] # 2D to 1D

    return rotated_vector

def at2rotmat(alpha,theta):
    """
    根据机器人的alpha和theta得到变换矩阵
    """
    return np.array([
        [np.cos(theta),-np.sin(theta)*np.cos(alpha),np.sin(theta)*np.sin(alpha)],
        [np.sin(theta),np.cos(theta)*np.cos(alpha),-np.cos(theta)*np.sin(alpha)],
        [0,np.sin(alpha),np.cos(alpha)]
    ])

def kinematics(six_joints):
    """
    ur5机械臂的正向D-H运动学, joints in rad
    """
    thetas = [theta for theta in six_joints]
    alphas = [np.pi/2,0,0,np.pi/2,-np.pi/2,0]
    Ts = [
        at2rotmat(alpha, theta) for alpha,theta in zip(alphas,thetas)
    ]
    ret_mat = np.eye(3)
    for T_mat in Ts:
        ret_mat = ret_mat@T_mat
    ret_mat = ret_mat
    return ret_mat



def main():
    # 创建机器人环境对象
    robot_env = robotEnv.RobotEnv()

    while True:
        # 获取腕部相机画面
        frame = robot_env.wrist_camera.get_rgb_frame()

        # 获取机器人tcp
        tcp = robot_env.get_tcp_pose()

        # 获取rxryrz
        rxryrz = tcp[3:]
        # 根据rxryrz计算针尖的方向
        tip_direction = rxryrz2xyz_scipy(rxryrz)
        joint_positions = robot_env.robot.get_joint_positions()
        joint_positions = np.array(joint_positions)
        kine_matrix = kinematics(joint_positions)
        tip_direction = kine_matrix@local_tool_vec
        tip_direction = tip_direction[:,0] # 2D to 1D

        # 将方向实时显示在屏幕上
        cv2.rectangle(frame,(50,180),(300,250),(0,0,0),-1)
        # cv2.putText(frame,f'rx ry rz=[{rxryrz[0]:.2f},{rxryrz[1]:.2f},{rxryrz[2]:.2f}]',(50,200),fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        #              fontScale = 0.5,color=(255,255,0))
        # cv2.putText(frame,f'x y z=[{tip_direction[0]:.2f},{tip_direction[1]:.2f},{tip_direction[2]:.2f}]',(50,230),fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        #              fontScale = 0.5,color=(255,255,0))
        cv2.putText(frame,f'joints=[{joint_positions[0]:.2f},{joint_positions[1]:.2f},{joint_positions[2]:.2f}{joint_positions[3]:.2f},{joint_positions[4]:.2f},{joint_positions[5]:.2f}]',(50,200),fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                     fontScale = 0.5,color=(255,255,0))
        cv2.putText(frame,f'x y z=[{tip_direction[0]:.2f},{tip_direction[1]:.2f},{tip_direction[2]:.2f}]',(50,230),fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                     fontScale = 0.5,color=(255,255,0))

        cv2.imshow('realtime tcp direction',frame)
        key = cv2.waitKey(1)
        if key==ord('Q'):
            cv2.destroyAllWindows()
            break


        # print('-'*10+f'{datetime.now()}'+'-'*10)
        # print('current tcp:',tcp)
        # print('tip direction:',tip_direction)
        
        # input()
        # print()


if __name__ == '__main__':
    main()
    # duo_rxryrz = [
    #     [-2.74,1.32,-0.77],
    #     [2.74,-1.32,0.77]
    # ]
    # ss = [
    #     'xyz','xzy','yxz','yzx','zxy','zyx'
    # ]
    # for s in ss:
    #     print(f's={s}')
    #     for rxryrz in duo_rxryrz:
    #         rxryrz = np.array(rxryrz)
    #         xyz_self = rxryrz2xyz_self(rxryrz,s)
    #         xyz_scipy = rxryrz2xyz_scipy(rxryrz,s)
    #         print(f'rotated vector self:{xyz_self}')
    #         print(f'rotated vector scipy:{xyz_scipy}')
    #     print()
