import torch
import numpy as np
import json
import os
import sys
import pdb 
from scipy.spatial.transform import Rotation as R

import argparse
from rekep.environment import R2D2Env

from rekep.ik_solver import FrankaIKSolver
# from rekep.ik_solver_ur5 import UR5eIKSolver
from rekep.iksolver_xjc import UR5IKSolver

from rekep.subgoal_solver import SubgoalSolver
from rekep.path_solver import PathSolver
import rekep.transform_utils as T
from rekep.visualizer import Visualizer
# from kinova import KinovaRobot
from robotEnv import RobotEnv

from rekep.utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)

from r2d2_vision import R2D2Vision

import datetime

'''
metadata.json
{
    "init_keypoint_positions": [
        [-0.1457058783982955, -0.47766187961876, 0.98],
        [-0.0144477656708159, 0.012521396914707113, 0.745],
        [0.14099338570298237, 0.5722672713826932, 1.283],
        [0.2693722882157947, -0.3018593983523729, 1.047],
        [0.43524427390119413, -0.04595746991503292, 0.6970000000000001]
    ],
    "num_keypoints": 5,
    "num_stages": 4,
    "grasp_keypoints": [1, -1, 2, -1],
    "release_keypoints": [-1, 1, -1, 2]
}
'''
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="xFormers is not available")

import time

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper

@timer_decorator
class MainR2D2:
    def __init__(self, visualize=False):
        global_config = get_config(config_path="./configs/config.yaml")
        self.config = global_config['main']
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.visualize = visualize
        # set random seed
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])

        # self.vision = R2D2Vision(visualize=self.visualize)
        # self.kinova = KinovaRobot()
        self.robot_env = RobotEnv()
        self.env = R2D2Env(global_config['env'])

        self.gripper_length = 0.180 # 工具长度
        print('gripper length:', self.gripper_length,'m')
        self.mat_gripper2ee = np.zeros(4)

        
        #替换
        # ik_solver = FrankaIKSolver(
        #     reset_joint_pos= self.env.reset_joint_pos,
        #     world2robot_homo= self.env.world2robot_homo,
        # )
        # ik_solver = UR5eIKSolver(
        #     reset_joint_pos= self.env.reset_joint_pos,
        #     world2robot_homo= self.env.world2robot_homo,
        # )
        # ik_solver = UR5IKSolver(
        #     reset_joint_pos= self.env.reset_joint_pos,
        #     world2robot_homo= self.env.world2robot_homo,
        # )

        # initialize solvers
        self.subgoal_solver = SubgoalSolver(global_config['subgoal_solver'],
                                            #  ik_solver,
                                               self.env.reset_joint_pos)
        self.path_solver = PathSolver(global_config['path_solver'],
                                    #    ik_solver,
                                         self.env.reset_joint_pos)
        # self.visualizer = Visualizer(global_config['visualizer'])
        with open('./robot_state.json', 'r') as f:
            robot_state = json.load(f)
            # 获取坐标系变换矩阵
            self.mat_world2robot = np.array(robot_state['misc']['world2robot_homo']) # 从世界坐标系(cam)变换到机器人坐标系(base)
            self.mat_robot2world = np.linalg.inv(self.mat_world2robot) # 从机器人坐标系变换到世界坐标系

        # if visualize:
        #     self.visualizer = Visualizer(global_config['visualizer'])
        #     self.data_path = "D:\ReKep-main\data"

    @timer_decorator
    def perform_task(self, instruction, obj_list=None, rekep_program_dir=None):
        # ====================================
        # = keypoint proposal and constraint generation
        # ====================================
        # obj_list = ['scissors']
        
        data_path = "/home/franka/R2D2_3dhat/images/current_images"
        
        if rekep_program_dir is None:
            pass
            # realworld_rekep_program_dir = self.vision.perform_task(instruction, obj_list, data_path, 3)
        else:
            realworld_rekep_program_dir = rekep_program_dir
        # ====================================
        self._execute(realworld_rekep_program_dir)

    def pose7d_world2robot(self,pose7d):
        """
        将[x,y,z,quat]从世界/相机坐标系转换为机器人坐标系
        """
        # vec2mat
        xyz = pose7d[0:3]
        orientation_quat = pose7d[3:]
        mat_pose_world = np.eye(4)
        mat_pose_world[:3,3] = xyz
        mat_pose_world[:3,:3] = R.from_quat(orientation_quat).as_matrix()

        # transformation matrix
        transformation_matrix = self.mat_world2robot
        mat_pose_robot = transformation_matrix@mat_pose_world

        # mat2vec
        robot_xyz = mat_pose_robot[:3,3]
        robot_quat = R.from_matrix(mat_pose_robot[:3,:3]).as_quat()
        robot_pose7d = np.hstack((robot_xyz, robot_quat))
        return robot_pose7d
    def pose7d_robot2world(self,pose7d):
        """
        将[x,y,z,quat]从机器人坐标系转换为世界/相机坐标系
        """
        # vec2mat
        xyz = pose7d[0:3]
        orientation_quat = pose7d[3:]
        mat_pose_robot = np.eye(4)
        mat_pose_robot[:3,3] = xyz
        mat_pose_robot[:3,:3] = R.from_quat(orientation_quat).as_matrix()

        # transformation matrix
        transformation_matrix = self.mat_robot2world
        mat_pose_world = transformation_matrix@mat_pose_robot

        # mat2vec
        world_xyz = mat_pose_world[:3,3]
        world_quat = R.from_matrix(mat_pose_world[:3,:3]).as_quat()
        world_pose7d = np.hstack((world_xyz, world_quat))
        return world_pose7d

    @timer_decorator
    def _execute(self, rekep_program_dir):
        # Load program info and constraints
        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
            self.program_info = json.load(f)

        # Register initial keypoints
        self.env.register_keypoints(self.program_info['init_keypoint_positions'])
        
        # Load all stage constraints
        self.constraint_fns = self._load_constraints(rekep_program_dir)
        
        # bookkeeping of which keypoints can be moved in the optimization
        self.keypoint_movable_mask = np.zeros(self.program_info['num_keypoints'] + 1, dtype=bool)
        self.keypoint_movable_mask[0] = True  # first keypoint is always the ee, so it's movable

        # Generate action sequences for all stages
        self.all_actions = []
        # pdb.set_trace()
        # Process each stage sequentially
        # if 1:
        # Read stage from robot state file
        with open('./robot_state.json', 'r') as f:
            robot_state = json.load(f)
            stage = robot_state.get('rekep_stage', 1)  # !!! @Tianyou Default to stage 1 if not found

        # store robot state in rekep_program_dir
        with open(os.path.join(rekep_program_dir, f'robot_state_{stage}.json'), 'w') as f:
            json.dump(robot_state, f, indent=4)
                # Get current state
        while int(stage) <= self.program_info['num_stages']:
            # scene_keypoints = self.env.get_keypoint_positions()
            # self.keypoints = np.concatenate([[self.env.get_ee_pos()], # 加一个方括号是为了把1D的array升成2D
            #                                  self.env.get_keypoint_positions()], axis=0)
            self.keypoints = np.concatenate([
                [self._kinova_get_ee_pos_world()[:3]],
                self.env.get_keypoint_positions()
            ],axis=0)
            print(f"stage {int(stage)}: keypoints{self.keypoints}")
            # self.curr_ee_pose = self.env.get_ee_pose()  # TODO check, may be constant? 
            # self.curr_joint_pos = self.env.get_arm_joint_positions() 
            # self.curr_ee_pose = self._kinova_get_ee_pos()
            self.curr_ee_pose = self._kinova_get_ee_pos_world()
            self.curr_joint_pos = self._kinova_get_joint_pos()
            # self.sdf_voxels = self.env.get_sdf_voxels(self.config['sdf_voxel_size']) # TODO ???
            self.collision_points = self.env.get_collision_points()
            
            # stage = input(f"Enter stage number (1-{self.program_info['num_stages']}): ")
            stage = int(stage)
            if stage > self.program_info['num_stages']:
                print(f"{bcolors.FAIL}Stage {stage} is out of bounds, skipping\n{bcolors.ENDC}")
                return
            self._update_stage(stage)
            # if self.stage > 1:
            #     path_constraints = self.constraint_fns[self.stage]['path']
            #     for constraints in path_constraints:
            #         violation = constraints(self.keypoints[0], self.keypoints[1:])
            #         if violation > self.config['constraint_tolerance']:
            #             # backtrack = True
            #             print(f"\n\n\nConstraint violation: {violation}\n\n\n")
            #             break
       
            # Generate actions for this stage
            next_subgoal = self._get_next_subgoal(from_scratch=self.first_iter)
            print(f'next subgoal(world/camera): {next_subgoal}') # 假设subgoal是基于相机坐标系的
            print(f'next subgoal(robot): {self.pose7d_world2robot(next_subgoal)}')

            # 维度对上了
            next_path = self._get_next_path(next_subgoal, from_scratch=self.first_iter)
            self.first_iter = False

            # 将subgoal和path输出到文件中
            currrent_time = datetime.datetime.now()
            timestamp = '{}-{}-{}-{}-{}-{}'.format(
                currrent_time.year,
                currrent_time.month,
                currrent_time.day,
                currrent_time.hour,
                currrent_time.minute,
                currrent_time.second,
            )
            output_dir = './outputs/'
            subgoal_file_path = f'{output_dir}next_subgoals_{timestamp}'
            next_path_file_path = f'{output_dir}next_paths_{timestamp}'
            with open(subgoal_file_path,'a') as f:
                f.write(f'{next_subgoal}\n')
                print(f'subgoal written to {subgoal_file_path}')
            with open(next_path_file_path,'a') as f:
                f.write(f'{next_path}\n')
                print(f'next path written to {next_path_file_path}')

            # pdb.set_trace()
            # Add gripper actions based on stage type
            # True or False from metadata.json
            # if self.is_grasp_stage: 
            #     next_path[-1, 7] = self.env.get_gripper_close_action() # Todo aliagn shape?
                
            # elif self.is_release_stage:
            #     next_path[-1, 7] = self.env.get_gripper_open_action() 
                
            for i in range(next_path.shape[0]):
                if self.is_grasp_stage:
                    print('is grasp stage')
                    next_path[i, 7] = self.env.get_gripper_close_action()
                elif self.is_release_stage:
                    print('is release stage')
                    next_path[i, 7] = self.env.get_gripper_open_action()

            for i in range(next_path.shape[0]):
                target_pos_world = next_path[i, :7]
                # 将世界坐标系的位置转换为机器人坐标系的位置
                mat_target_pos_world = np.eye(4)
                mat_target_pos_world[:3,:3] = R.from_quat(target_pos_world[3:]).as_matrix()
                mat_target_pos_world[:3,3] = target_pos_world[:3]
                mat_target_pos_robot = self.mat_world2robot@mat_target_pos_world
                target_pos_robot = np.zeros(7)
                target_pos_robot[3:] = R.from_matrix(mat_target_pos_robot[:3,:3]).as_quat()
                target_pos_robot[:3] = mat_target_pos_robot[:3,3]

                # 

                self._kinova_move_to_ee_pos(target_pos_robot) # 前往机器人base坐标系下的位置
                
            self.all_actions.append(next_path)
            stage += 1
            self._update_stage(stage)

        # Combine all action sequences
        combined_actions = np.concatenate(self.all_actions, axis=0)

        if self.stage <= self.program_info['num_stages']: 
            # self.env.sleep(2.0)
            save_path = os.path.join('./outputs', 'action.json') # TODO: save by stage?
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump({"ee_action_seq": combined_actions.tolist(), "stage": stage}, f, indent=4)

            with open(os.path.join(rekep_program_dir, f'stage{stage}_actions.json'), 'w') as f:
                json.dump({"ee_action_seq": combined_actions.tolist(), "stage": stage}, f, indent=4)
            print(f"{bcolors.OKGREEN}Actions saved to {save_path}\n and added to {rekep_program_dir}\n{bcolors.ENDC}")
            return
        else:
            print(f"{bcolors.OKGREEN}All stages completed\n\n{bcolors.ENDC}")
            # TODO: return to reset pose?
            # self.env.reset()
            return  
            
            return

    def _kinova_get_joint_pos(self):
        # joint_pos = self.kinova.get_joint_positions()
        joint_pos = self.robot_env.robot.get_joint_positions()
        return joint_pos
    
    def _kinova_get_ee_pos(self):
        # ee_pos = self.kinova.get_tool_position()
        ee_pos = self.robot_env.robot.get_tcp_pose() # tool central pose 工具中心点
        angles = ee_pos[3:]
        # angles = np.radians(angles)
        rotation = R.from_euler('xyz', angles)
        print('angles:',angles)
        print('rotation:', rotation.as_matrix())
        quat = rotation.as_quat() # 可能是四元旋转矩阵
        quat = quat.reshape(1, -1)
        quat = quat[0]
        print(quat)
        return np.concatenate([ee_pos[:3], quat])

    def _kinova_get_ee_pos_world(self):
        """
        获取ur5机器人的ee的7D坐标, [x,y,z,quat], 同时是在世界坐标系下(相机坐标系下)
        """
        # 获取机器人ee在基座base坐标系下的原始坐标
        ee_pos = self.robot_env.robot.get_tcp_pose() # tool central pose 工具中心点
        angles = ee_pos[3:] # 机器人ee的方向(欧拉角,rad)
        rotation = R.from_euler('xyz', angles) # 将欧拉角转换为scipy标准形式
        print('angles:',angles)
        print('rotation:', rotation.as_matrix())

        # 将机器人ee的base坐标转换为世界坐标系
        # 首先是把ee用4*4的矩阵表示
        ee_matrix_base = np.eye(4)
        ee_matrix_base[:3,:3] = rotation.as_matrix()
        ee_matrix_base[:3,3] = ee_pos[:3]
        # 获取变换矩阵
        mat_robot2world = self.mat_robot2world
        # 将ee矩阵从机器人坐标系变换到世界坐标系(相机坐标系)
        ee_matrix_world = mat_robot2world@ee_matrix_base
        # 将ee矩阵的旋转信息和位置信息解包出来
        ee_pos_world = ee_matrix_world[:3,3] # 机器人ee在世界坐标系下的坐标[x,y,z]
        ee_rotation_world = R.from_matrix(ee_matrix_world[:3,:3]) # 机器人ee在世界坐标系下的方向(转换为scipy标准形式)

        # 将ee方向转换为四元数quat
        print('ee_rotation_world:',ee_rotation_world.as_euler('xyz'))
        print('ee_pos_world:',ee_pos_world)
        quat = ee_rotation_world.as_quat()
        quat = quat.reshape(1, -1)
        quat = quat[0]
        print('quat world:',quat)
        return np.concatenate([ee_pos_world, quat])

    def euler_to_direction(roll, pitch, yaw, local_vector=np.array([1, 0, 0])):
        """
        将欧拉角转换为全局方向向量。
        
        参数：
        - roll: 横滚角（绕 x 轴），单位：弧度
        - pitch: 俯仰角（绕 y 轴），单位：弧度
        - yaw: 偏航角（绕 z 轴），单位：弧度
        - local_vector: 在夹爪局部坐标系中所选的参考向量，默认为 [1, 0, 0]（夹爪前向）
        
        返回：
        - global_vector: 全局坐标系下的方向向量（单位向量）
        """
        # 构造各轴旋转矩阵（按照 Z-Y-X 顺序：yaw、pitch、roll）
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,           0,            1]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0,             1, 0],
            [-np.sin(pitch),0, np.cos(pitch)]
        ])
        
        Rx = np.array([
            [1, 0,             0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])
        
        # 总旋转矩阵
        R = Rz.dot(Ry).dot(Rx)
        
        # 将局部参考向量转换到全局坐标系
        global_vector = R.dot(local_vector)
        
        # 归一化，确保为单位向量
        norm = np.linalg.norm(global_vector)
        if norm != 0:
            global_vector = global_vector / norm
        
        return global_vector
    
    def toolpos2gripperpos(self, tool_pos_mat):
        pass

    
    def _kinova_move_to_tool_ee_pos(self, target_pos):
        """
        将工具末端移到base坐标系下的target_pos:[7D]
        """
        # 工具在base坐标系下的7D坐标
        target_pos_tool = target_pos.copy()

        target_quat = target_pos_tool[3:]
        rotation = R.from_quat(target_quat)
        angles_rad = rotation.as_euler('xyz')
        # angles_deg = np.degrees(angles_rad)
        target_pos_tool_6D = np.concatenate([target_pos[:3], angles_rad])
        # 将target_pos转换为矩阵格式
        target_pos_tool_mat = np.eye(4)
        target_pos_tool_mat[:3,3] = target_pos_tool_6D[:3]
        target_pos_tool_mat[:3,:3] = rotation.as_matrix()

        # 将目标从工具转换到ee
        target_pos_ee_mat = self.toolpos2gripperpos(target_pos_tool_mat)

        # 将ee目标矩阵转换为6D向量
        target_pos = np.zeros(6)
        target_pos[:3] = target_pos_ee_mat[:3,3]
        target_pos[3:] = R.from_matrix(target_pos_ee_mat[:3,:3]).as_euler('xyz')

        # name each element in target_pos
        target_pos = {
            "x": target_pos[0],
            "y": target_pos[1],
            "z": target_pos[2],
            "theta_x": target_pos[3],
            "theta_y": target_pos[4],
            "theta_z": target_pos[5]
        }
        # print(f'kinova move to {target_pos}')
        # self.kinova.move_to_tool_position(target_pos)
        # self.robot_env.step(target_pos,3)
        self.robot_env.move_to_ee_pos(target_pos)

    def _kinova_move_to_ee_pos(self, target_pos):
        """
        前往机器人base坐标系下的位置, target_pos:[7D]
        """
        target_quat = target_pos[3:]
        rotation = R.from_quat(target_quat)
        angles_rad = rotation.as_euler('xyz')
        angles_deg = np.degrees(angles_rad)
        target_pos = np.concatenate([target_pos[:3], angles_deg])
        # name each element in target_pos
        target_pos = {
            "x": target_pos[0],
            "y": target_pos[1],
            "z": target_pos[2],
            "theta_x": target_pos[3],
            "theta_y": target_pos[4],
            "theta_z": target_pos[5]
        }
        # print(f'kinova move to {target_pos}')
        # self.kinova.move_to_tool_position(target_pos)
        # self.robot_env.step(target_pos,3)
        self.robot_env.move_to_ee_pos(target_pos)

    def _load_constraints(self, rekep_program_dir):
        """Helper to load all stage constraints"""
        constraint_fns = dict()
        for stage in range(1, self.program_info['num_stages'] + 1):
            stage_dict = dict()
            for constraint_type in ['subgoal', 'path']:
                load_path = os.path.join(rekep_program_dir, f'stage{stage}_{constraint_type}_constraints.txt')
                get_grasping_cost_fn = get_callable_grasping_cost_fn(self.env)
                stage_dict[constraint_type] = load_functions_from_txt(load_path, get_grasping_cost_fn) if os.path.exists(load_path) else []
            constraint_fns[stage] = stage_dict
        return constraint_fns

    @timer_decorator
    def _get_next_subgoal(self, from_scratch):
        # pdb.set_trace()
        subgoal_constraints = self.constraint_fns[self.stage]['subgoal']
        path_constraints = self.constraint_fns[self.stage]['path']
        param_names = "self.curr_ee_pose,self.keypoints,self.keypoint_movable_mask,subgoal_constraints,path_constraints,self.collision_points,self.is_grasp_stage,self.curr_joint_pos,from_scratch"
        param_name_list = param_names.split(',')
        params = [self.curr_ee_pose,self.keypoints,self.keypoint_movable_mask,subgoal_constraints,path_constraints,self.collision_points,self.is_grasp_stage,self.curr_joint_pos,from_scratch]
        print('subgoal params:')
        for param_name, param in zip(param_name_list,params):
            print('\t',param_name,":",param)
            print()
        
        subgoal_pose, debug_dict = self.subgoal_solver.solve(self.curr_ee_pose,
                                                            self.keypoints,
                                                            self.keypoint_movable_mask,
                                                            subgoal_constraints,
                                                            path_constraints,
                                                            # self.sdf_voxels,
                                                            # self.collision_points,
                                                            self.is_grasp_stage,
                                                            self.curr_joint_pos,
                                                            from_scratch=from_scratch)
        subgoal_pose_homo = T.convert_pose_quat2mat(subgoal_pose)
        # if grasp stage, back up a bit to leave room for grasping
        # if self.is_grasp_stage:
        #     subgoal_pose[:3] += subgoal_pose_homo[:3, :3] @ np.array([-self.config['grasp_depth'] / 2.0, 0, 0])
        debug_dict['stage'] = self.stage
        print_opt_debug_dict(debug_dict)
        if self.visualize:
            self.visualizer.visualize_subgoal(subgoal_pose, self.data_path)
        print('subgoal calc finished')
        return subgoal_pose

    @timer_decorator
    def _get_next_path(self, next_subgoal, from_scratch):
        # pdb.set_trace()
        # print(f"Start solving path from {self.curr_ee_pose} to {next_subgoal}")
        # print(f'curr joint pos:{self.curr_joint_pos}')
        path_constraints = self.constraint_fns[self.stage]['path']
        path, debug_dict = self.path_solver.solve(self.curr_ee_pose,
                                                    next_subgoal,
                                                    self.keypoints,
                                                    self.keypoint_movable_mask,
                                                    path_constraints,
                                                    # self.sdf_voxels,
                                                    self.collision_points,
                                                    self.curr_joint_pos,
                                                    from_scratch=from_scratch)
        print_opt_debug_dict(debug_dict)
        processed_path = self._process_path(path)
        
        if self.visualize:
            self.visualizer.visualize_path(processed_path, self.data_path)
            
        return processed_path
    
    # TODO: check action sequence
    @timer_decorator
    def _process_path(self, path):
        # pdb.set_trace()
        # spline interpolate the path from the current ee pose
        full_control_points = np.concatenate([
            self.curr_ee_pose.reshape(1, -1),
            path,
        ], axis=0)
        num_steps = get_linear_interpolation_steps(full_control_points[0], full_control_points[-1],
                                                    self.config['interpolate_pos_step_size'],
                                                    self.config['interpolate_rot_step_size'])
        dense_path = spline_interpolate_poses(full_control_points, num_steps)
        # add gripper action
        ee_action_seq = np.zeros((dense_path.shape[0], 8))
        ee_action_seq[:, :7] = dense_path
        ee_action_seq[:, 7] = self.env.get_gripper_null_action()
        return ee_action_seq

    def _update_stage(self, stage):
        # update stage
        self.stage = stage
        self.is_grasp_stage = self.program_info['grasp_keypoints'][self.stage - 1] != -1
        self.is_release_stage = self.program_info['release_keypoints'][self.stage - 1] != -1
        # can only be grasp stage or release stage or none
        assert self.is_grasp_stage + self.is_release_stage <= 1, "Cannot be both grasp and release stage"
        if self.is_grasp_stage:  # ensure gripper is open for grasping stage
            self.env.open_gripper()
        # clear action queue
        self.action_queue = []
        # update keypoint movable mask
        self._update_keypoint_movable_mask()
        self.first_iter = True

    def _update_keypoint_movable_mask(self):
        for i in range(1, len(self.keypoint_movable_mask)):  # first keypoint is ee so always movable
            keypoint_object = self.env.get_object_by_keypoint(i - 1)
            self.keypoint_movable_mask[i] = self.env.is_grasping(keypoint_object)

    def _execute_grasp_action(self):
        # pdb.set_trace()
        print("Grasp action")
    
    def _execute_release_action(self):
        print("Release action")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruction', type=str, required=False, help='Instruction for the task')
    parser.add_argument('--rekep_program_dir', type=str, required=False, help='keypoint constrain proposed folder')
    parser.add_argument('--visualize', action='store_true', help='visualize each solution before executing (NOTE: this is blocking and needs to press "ESC" to continue)')
    args = parser.parse_args()

    # args.instruction = "Put the green package in the drawer, the robot is already grasping the package and the package is already aligned with the drawer opening."
    # args.obj_list = ['cloth']

    # vlm_query_dir = "/home/franka/R2D2_3dhat/ReKep/vlm_query/"

    # vlm_dirs = [os.path.join(vlm_query_dir, d) for d in os.listdir(vlm_query_dir) 
    #             if os.path.isdir(os.path.join(vlm_query_dir, d))]
    
    # if vlm_dirs:
    #     newest_rekep_dir = max(vlm_dirs, key=os.path.getmtime)
    #     print(f"\033[92mUsing most recent directory: {newest_rekep_dir}\033[0m")
    # else:
    #     print("No directories found under vlm_query")
    #     sys.exit(1)

    # newest_rekep_dir = "//home//kinova//Rekep4Real//vlm_query//2025-01-21_22-27-34_help_me_take_that_bottle_of_water"
    # newest_rekep_dir = '/home/ur5/rekep/Rekep4Real/vlm_query/2025-01-21_22-27-34_help_me_take_that_bottle_of_water'
    # newest_rekep_dir = '/home/ur5/rekep/ReKepUR5_from_kinova/vlm_query/2025-02-25_15-47-47_help_me_take_the_cube'
    # newest_rekep_dir = '/home/ur5/rekep/ReKepUR5_from_kinova/vlm_query/2025-02-26_10-35-40_help_me_take_the_cube'
    # newest_rekep_dir = '/home/ur5/rekep/ReKepUR5_from_kinova/vlm_query/2025-03-04_19-23-52_help_me_take_the_block'
    newest_rekep_dir = '/home/ur5/rekep/ReKepUR5_from_kinova/vlm_query/2025-03-06_13-41-37_help_me_take_the_block'

    def euler_to_direction(roll, pitch, yaw, local_vector=np.array([0, 1, 0])):
        """
        将欧拉角转换为全局方向向量。
        
        参数：
        - roll: 横滚角（绕 x 轴），单位：弧度
        - pitch: 俯仰角（绕 y 轴），单位：弧度
        - yaw: 偏航角（绕 z 轴），单位：弧度
        - local_vector: 在夹爪局部坐标系中所选的参考向量，默认为 [1, 0, 0]（夹爪前向）
        
        返回：
        - global_vector: 全局坐标系下的方向向量（单位向量）
        """
        # 构造各轴旋转矩阵（按照 Z-Y-X 顺序：yaw、pitch、roll）
        rotation_mat = R.from_euler('xyz',np.array([roll,pitch,yaw])).as_matrix()
        transformation_mat = np.eye(4)
        transformation_mat[:3,:3] = rotation_mat

        local_vector_col = np.array([[local_vector[0]],[local_vector[1]],[local_vector[2]],[1]])
        return (transformation_mat@local_vector_col)[:3]
    print(euler_to_direction(1.188,-2.789,-0.002))
    # main = MainR2D2(visualize=args.visualize)
    # main._kinova_get_ee_pos()
    # main._kinova_get_ee_pos_world()
    # main.perform_task(instruction=args.instruction, rekep_program_dir=newest_rekep_dir)
