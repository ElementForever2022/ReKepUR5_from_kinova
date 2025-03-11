import time
import numpy as np
import os
import datetime
import rekep.transform_utils as T
# import trimesh
# import open3d as o3d
import imageio
import json
from pathlib import Path
# OG related
# import omnigibson as og
# from omnigibson.macros import gm
# from omnigibson.utils.usd_utils import PoseAPI, mesh_prim_mesh_to_trimesh_mesh, mesh_prim_shape_to_trimesh_mesh
# from omnigibson.robots.fetch import Fetch
# from omnigibson.controllers import IsGraspingState


# from og_utils import OGCamera
# from omnigibson.robots.manipulation_robot import ManipulationRobot
# from omnigibson.controllers.controller_base import ControlType, BaseController


from .utils import (
    bcolors,
    get_clock_time,
    angle_between_rotmat,
    angle_between_quats,
    get_linear_interpolation_steps,
    linear_interpolate_poses,
)

# some customization to the OG functions
# def custom_clip_control(self, control):
#     """
#     Clips the inputted @control signal based on @control_limits.

#     Args:
#         control (Array[float]): control signal to clip

#     Returns:
#         Array[float]: Clipped control signal
#     """
#     clipped_control = control.clip(
#         self._control_limits[self.control_type][0][self.dof_idx],
#         self._control_limits[self.control_type][1][self.dof_idx],
#     )
#     idx = (
#         self._dof_has_limits[self.dof_idx]
#         if self.control_type == ControlType.POSITION
#         else [True] * self.control_dim
#     )
#     if len(control) > 1:
#         control[idx] = clipped_control[idx]
#     return control

# Fetch._initialize = ManipulationRobot._initialize
# BaseController.clip_control = custom_clip_control



class RobotController:
    def __init__(self):
        self.joint_limits = {
            'position': {
                'upper': [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],  # Franka真实限制
                'lower': [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
            },
            'velocity': {
                'upper': [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100],
                'lower': [-2.1750, -2.1750, -2.1750, -2.1750, -2.6100, -2.6100, -2.6100]
            }
        }
        
        # self.current_joint_angles = np.zeros(7)
        self.current_ee_pose = np.array([0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
        self.current_eef_position = np.array([0.5, 0.0, 0.5]) 
        self.world2robot_homo = np.eye(4)  
        self.gripper_state = 0.0
        self.robot_state_path =  Path('./robot_state.json')

    def get_relative_eef_position(self):
        """
        Mock version of get_relative_eef_position
        Returns:
            np.ndarray: [3,] array representing the end effector position in robot frame
        """
        return self.current_eef_position

    def get_relative_eef_orientation(self):
        """
        Mock version of get_relative_eef_orientation
        Returns:
            np.ndarray: [4,] array representing the end effector orientation as quaternion [w,x,y,z]
        """
        return np.array([1.0, 0.0, 0.0, 0.0])  # 默认朝向
    
    def clip_control(self, control, control_type='position'):
        return control 
        # if control_type not in self.joint_limits:
        #     print(f"Warning: Unknown control type {control_type}")
        #     return control
        # upper = self.joint_limits[control_type]['upper']
        # lower = self.joint_limits[control_type]['lower']
        
        # n_joints = min(len(control), len(upper))
        # clipped = np.clip(control[:n_joints], lower[:n_joints], upper[:n_joints])
        # print(f"Clipping {control_type} control from {control} to {clipped}")
        # return clipped

    def send_command(self, command, control_type='position'):
        """
        Send a safe command to the robot
        
        Args:
            command: Control command
            control_type: Type of control
        """
        safe_command = self.clip_control(command, control_type)
        print(f"Sending {control_type} command: {safe_command}")
        # Here you would add your actual robot control code
        return safe_command


class R2D2Env:
    def __init__(self, config=None, verbose=False):
        self.video_cache = []
        self.config = config
        self.verbose = verbose 
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.interpolate_pos_step_size = self.config['interpolate_pos_step_size']
        self.interpolate_rot_step_size = self.config['interpolate_rot_step_size']
        self.robot_state_path =  Path('./robot_state.json')
        
        # self.robot = RobotController() # TODO 不知道有什么用处
        self.reset_joint_pos = np.array([0.5, 0, 0.5, 1, 0, 0])
        # self.reset_joint_pos = np.array([0.5, 0, 0.5, 1, 0, 0, 0])  # Default home position
        self.gripper_state = 1.0  # 1.0 is open, 0.0 is closed
        self.world2robot_homo = np.eye(4)
        print("Robot interface initialized")
          # Initialize robot state
        self.update_robot_state()
        
    def update_robot_state(self):
        """Read and update robot state from json file"""
        with open(self.robot_state_path, 'r') as f:
            robot_state = json.load(f)
            
        # self.current_joint_angles = np.array(robot_state['joint_info']['joint_positions'])
        
        # self.ee_position = np.array(robot_state['ee_info']['position'])
        # self.ee_orientation = np.array(robot_state['ee_info']['orientation'])
        # self.ee_pose = np.concatenate([self.ee_position, self.ee_orientation])
        
        self.gripper_state = robot_state['gripper_info']['state']
        
        self.collision_status = robot_state['safety_info']['collision_status']
        self.safety_status = robot_state['safety_info']['safety_status']
        
        # Update misc information
        self.world2robot_homo = np.array(robot_state['misc']['world2robot_homo'])
        
        return True
    
    def get_ee_pose(self,from_robot=False):
        """Get end-effector pose"""
        print('getting ee pose from R2D2Env in environment.py')
        if from_robot:
            print(f"Getting EE pose from robot: {self.ee_pose}")
            return self.ee_pose # TODO 
        else:
            print(f"Getting EE pose: {self.ee_pose}")
            return self.ee_pose
    
    def get_ee_pos(self):
        """Get end-effector position"""
        return self.ee_pose[:3]
    
    def get_ee_quat(self):
        """Get end-effector orientation"""
        return self.ee_pose[3:]
    
    def compute_target_delta_ee(self, target_pose):
        target_pos, target_xyzw = target_pose[:3], target_pose[3:]
        ee_pose = self.get_ee_pose() 
        ee_pos, ee_xyzw = ee_pose[:3], ee_pose[3:]
        pos_diff = np.linalg.norm(ee_pos - target_pos)
        rot_diff = angle_between_quats(ee_xyzw, target_xyzw)
        return pos_diff, rot_diff
    
    # def get_arm_joint_positions(self):
    #     """
    #     Mock version of get_arm_joint_positions that returns simulated joint positions
        
    #     Returns:
    #         np.ndarray: Array of 7 joint positions for a 7-DOF arm
    #         [torso_lift, shoulder_pan, shoulder_lift, upperarm_roll, elbow_flex, forearm_roll, wrist_flex]
    #     """
    #     return self.current_joint_angles

        # # Return mock joint positions within reasonable ranges
        # mock_joint_positions = np.array([
        #     0.0,    # torso_lift   (-0.1, 0.4)
        #     0.0,    # shoulder_pan (-1.6, 1.6)
        #     0.0,    # shoulder_lift (-1.25, 1.25)
        #     0.0,    # upperarm_roll (-2.0, 2.0)
        #     0.0,    # elbow_flex   (-2.0, 2.0)
        #     0.0,    # forearm_roll (-2.0, 2.0)
        #     0.0     # wrist_flex   (-1.8, 1.8)
        # ])
        
            
        # return mock_joint_positions
    # def execute_action(self, action, precise=True):
    #     """
    #     Execute robot action
    #     action: [x,y,z, qx,qy,qz,qw, gripper_action]
    #     """
    #     target_pose = action[:7]
    #     gripper_action = action[7]
        
    #     print(f"Moving to pose: {target_pose}")
    #     self.ee_pose = target_pose  # Update internal state
        
    #     if gripper_action == self.get_gripper_close_action():
    #         self.close_gripper()
    #     elif gripper_action == self.get_gripper_open_action():
    #         self.open_gripper()
            
    #     # Return mock position and rotation errors
    #     return 0.01, 0.01

    # TODO: future work, not used yet
    def execute_action(
            self,
            action,
            precise=True,
        ):
            """
            Moves the robot gripper to a target pose by specifying the absolute pose in the world frame and executes gripper action.

            Args:
                action (x, y, z, qx, qy, qz, qw, gripper_action): absolute target pose in the world frame + gripper action.
                precise (bool): whether to use small position and rotation thresholds for precise movement (robot would move slower).
            Returns:
                tuple: A tuple containing the position and rotation errors after reaching the target pose.
            """
            if precise:
                pos_threshold = 0.03
                rot_threshold = 3.0
            else:
                pos_threshold = 0.10
                rot_threshold = 5.0
            action = np.array(action).copy()
            assert action.shape == (8,)
            target_pose = action[:7]
            gripper_action = action[7]

            # ======================================
            # = status and safety check
            # ======================================
            if np.any(target_pose[:3] < self.bounds_min) \
                 or np.any(target_pose[:3] > self.bounds_max):
                print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Target position is out of bounds, clipping to workspace bounds{bcolors.ENDC}')
                target_pose[:3] = np.clip(target_pose[:3], self.bounds_min, self.bounds_max)

            # ======================================
            # = interpolation
            # ======================================
            current_pose = self.get_ee_pose()
            pos_diff = np.linalg.norm(current_pose[:3] - target_pose[:3])
            rot_diff = angle_between_quats(current_pose[3:7], target_pose[3:7])
            pos_is_close = pos_diff < self.interpolate_pos_step_size
            rot_is_close = rot_diff < self.interpolate_rot_step_size
            if pos_is_close and rot_is_close:
                self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Skipping interpolation{bcolors.ENDC}')
                pose_seq = np.array([target_pose])
            else:
                num_steps = get_linear_interpolation_steps(current_pose, target_pose, self.interpolate_pos_step_size, self.interpolate_rot_step_size)
                pose_seq = linear_interpolate_poses(current_pose, target_pose, num_steps)
                self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Interpolating for {num_steps} steps{bcolors.ENDC}')

            # ======================================
            # = move to target pose
            # ======================================
            # move faster for intermediate poses
            intermediate_pos_threshold = 0.10
            intermediate_rot_threshold = 5.0
            for pose in pose_seq[:-1]:
                self._move_to_waypoint(pose, intermediate_pos_threshold, intermediate_rot_threshold)
            # move to the final pose with required precision
            pose = pose_seq[-1]
            self._move_to_waypoint(pose, pos_threshold, rot_threshold, max_steps=20 if not precise else 40) 
            # compute error
            pos_error, rot_error = self.compute_target_delta_ee(target_pose)
            self.verbose and print(f'\n{bcolors.BOLD}[environment.py | {get_clock_time()}] Move to pose completed (pos_error: {pos_error}, rot_error: {np.rad2deg(rot_error)}){bcolors.ENDC}\n')

            # ======================================
            # = apply gripper action
            # ======================================
            if gripper_action == self.get_gripper_open_action():
                self.open_gripper()
            elif gripper_action == self.get_gripper_close_action():
                self.close_gripper()
            elif gripper_action == self.get_gripper_null_action():
                pass
            else:
                raise ValueError(f"Invalid gripper action: {gripper_action}")
            
            return pos_error, rot_error
    def close_gripper(self):
        """Close gripper"""
        print("Closing gripper")
        self.gripper_state = 1.0 # transfer to AnyGrasp
        
    def open_gripper(self):
        """Open gripper"""
        print("Opening gripper")
        self.gripper_state = 0.0    

    def get_gripper_open_action(self):
        return -1.0
    
    def get_gripper_close_action(self):
        return 1.0
    
    def get_gripper_null_action(self):
        return 0.0

    def is_grasping(self, candidate_obj=None):
        """Check if gripper is grasping"""
        # Could be enhanced with force sensor readings
        # TODO, how to modify this
        print("Yes it is grasping")
        return self.gripper_state == 1.0

    def get_collision_points(self, noise=True):
        """Get collision points of gripper"""
        # Return mock collision points
        return np.array([
            [0.5, 0.5, 0.5],
            [0.6, 0.6, 0.6],
            [0.4, 0.4, 0.4]
        ])

    def get_keypoint_positions(self):
        return self.keypoints
    
    def get_object_by_keypoint(self, keypoint_idx):
        # TODO: asscociate keypoints with closest object (mask?)
        return None
    
    def register_keypoints(self, keypoints):
        """Register keypoints to track"""
        print(f"Registering keypoints number: {len(keypoints)}")
        self.keypoints = keypoints

    def get_sdf_voxels(self, resolution, exclude_robot=True, exclude_obj_in_hand=True):
        """Get signed distance field"""
        print("Getting SDF voxels (mock data)")
        # Return mock SDF grid
        return np.zeros((10, 10, 10))

    def get_cam_obs(self):
        """Get camera observations"""
        print("Getting camera observations")
        # Return mock RGB-D data
        return {
            "rgb": np.zeros((480, 640, 3)),
            "depth": np.zeros((480, 640))
        }

    def reset(self):
        """Reset robot to home position"""
        print("Resetting robot to home position")
        self.ee_pose = [0.5, 0, 0.5, 1, 0, 0, 0]
        self.open_gripper()

    def sleep(self, seconds):
        """Wait for specified duration"""
        print(f"Sleeping for {seconds} seconds")
        time.sleep(seconds)

    def save_video(self, save_path=None):
        save_dir = os.path.join(os.path.dirname(__file__), 'videos')
        os.makedirs(save_dir, exist_ok=True)
        if save_path is None:
            save_path = os.path.join(save_dir, f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.mp4')
        video_writer = imageio.get_writer(save_path, fps=30)
        for rgb in self.video_cache:
            video_writer.append_data(rgb)
        video_writer.close()
        return save_path


    # ======================================
    # = internal functions
    # ======================================
    def _check_reached_ee(self, target_pos, target_xyzw, pos_threshold, rot_threshold):
        """
        this is supposed to be for true ee pose (franka hand) in robot frame
        """ # TODO transform to robot frame
        current_pos = self.get_ee_pose()[:3]
        current_xyzw = self.get_ee_pose()[3:7]
        current_rotmat = T.quat2mat(current_xyzw)
        target_rotmat = T.quat2mat(target_xyzw)
        # calculate position delta
        pos_diff = (target_pos - current_pos).flatten()
        pos_error = np.linalg.norm(pos_diff)
        # calculate rotation delta
        rot_error = angle_between_rotmat(current_rotmat, target_rotmat)
        # print status
        self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}]  Curr pose: {current_pos}, {current_xyzw} (pos_error: {pos_error.round(4)}, rot_error: {np.rad2deg(rot_error).round(4)}){bcolors.ENDC}')
        self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}]  Goal pose: {target_pos}, {target_xyzw} (pos_thres: {pos_threshold}, rot_thres: {rot_threshold}){bcolors.ENDC}')
        if pos_error < pos_threshold and rot_error < np.deg2rad(rot_threshold):
            self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] OSC pose reached (pos_error: {pos_error.round(4)}, rot_error: {np.rad2deg(rot_error).round(4)}){bcolors.ENDC}')
            return True, pos_error, rot_error
        return False, pos_error, rot_error

    def _move_to_waypoint(self, target_pose_world, pos_threshold=0.02, rot_threshold=3.0, max_steps=10):
        pos_errors = []
        rot_errors = []
        count = 0
        while count < max_steps:
            reached, pos_error, rot_error = self._check_reached_ee(
                target_pose_world[:3], 
                target_pose_world[3:7], 
                pos_threshold, 
                rot_threshold
            )
            pos_errors.append(pos_error)
            rot_errors.append(rot_error)
            if reached:
                break
            # convert world pose to robot pose
            target_pose_robot = np.dot(self.world2robot_homo, T.convert_pose_quat2mat(target_pose_world))
            
            # Franka的action空间: [7个关节角度 + 1个夹爪值]
            action = np.zeros(8)  
            # 前7维是关节角度 - 这里需要通过逆运动学(IK)计算
            joint_angles = self.compute_ik(target_pose_robot)  # 需要实现IK
            action[:7] = joint_angles
            # 最后1维是夹爪
            action[7] = self.gripper_state
            
            # 执行动作
            self._step(action=action)
            count += 1

        if count == max_steps:
            print(f'Pose not reached after {max_steps} steps (pos_error: {pos_errors[-1]:.4f}, rot_error: {np.rad2deg(rot_errors[-1]):.4f})')

    def compute_ik(self, target_pose):
        """Mock IK solver - 在实际应用中需要替换为真实的IK"""
        # 这里返回一个合理范围内的关节角度
        return np.zeros(7)  # 7个关节的角度

    def _step(self, action):
        """执行一步动作"""
        # 模拟执行动作，更新机器人状态
        joint_angles = action[:7]
        gripper_action = action[7]
        
        # 更新状态
        # self.current_joint_angles = joint_angles
        self.gripper_state = gripper_action
        
        # 通过正运动学更新末端执行器位置（在实际应用中需要从真实机器人读取）
        # self.current_eef_position = self.compute_fk(joint_angles)  # 需要实现FK

# class ReKepOGEnv:
#     def __init__(self, config, scene_file, verbose=False):
#         self.video_cache = []
#         self.config = config
#         self.verbose = verbose
#         self.config['scene']['scene_file'] = scene_file
#         self.bounds_min = np.array(self.config['bounds_min'])
#         self.bounds_max = np.array(self.config['bounds_max'])
#         self.interpolate_pos_step_size = self.config['interpolate_pos_step_size']
#         self.interpolate_rot_step_size = self.config['interpolate_rot_step_size']
#         # create omnigibson environment
#         self.step_counter = 0
#         self.og_env = og.Environment(dict(scene=self.config['scene'], robots=[self.config['robot']['robot_config']], env=self.config['og_sim']))
#         self.og_env.scene.update_initial_state()
#         for _ in range(10): og.sim.step()
#         # robot vars
#         self.robot = self.og_env.robots[0]
#         dof_idx = np.concatenate([self.robot.trunk_control_idx,
#                                   self.robot.arm_control_idx[self.robot.default_arm]])
#         self.reset_joint_pos = self.robot.reset_joint_pos[dof_idx]
#         self.world2robot_homo = T.pose_inv(T.pose2mat(self.robot.get_position_orientation()))
#         # initialize cameras
#         self._initialize_cameras(self.config['camera'])
#         self.last_og_gripper_action = 1.0

#     # ======================================
#     # = exposed functions
#     # ======================================
#     def get_sdf_voxels(self, resolution, exclude_robot=True, exclude_obj_in_hand=True):
#         """
#         open3d-based SDF computation
#         1. recursively get all usd prim and get their vertices and faces
#         2. compute SDF using open3d
#         """
#         start = time.time()
#         exclude_names = ['wall', 'floor', 'ceiling']
#         if exclude_robot:
#             exclude_names += ['fetch', 'robot']
#         if exclude_obj_in_hand:
#             assert self.config['robot']['robot_config']['grasping_mode'] in ['assisted', 'sticky'], "Currently only supported for assisted or sticky grasping"
#             in_hand_obj = self.robot._ag_obj_in_hand[self.robot.default_arm]
#             if in_hand_obj is not None:
#                 exclude_names.append(in_hand_obj.name.lower())
#         trimesh_objects = []
#         for obj in self.og_env.scene.objects:
#             if any([name in obj.name.lower() for name in exclude_names]):
#                 continue
#             for link in obj.links.values():
#                 for mesh in link.collision_meshes.values():
#                     mesh_type = mesh.prim.GetPrimTypeInfo().GetTypeName()
#                     if mesh_type == 'Mesh':
#                         trimesh_object = mesh_prim_mesh_to_trimesh_mesh(mesh.prim)
#                     else:
#                         trimesh_object = mesh_prim_shape_to_trimesh_mesh(mesh.prim)
#                     world_pose_w_scale = PoseAPI.get_world_pose_with_scale(mesh.prim_path)
#                     trimesh_object.apply_transform(world_pose_w_scale)
#                     trimesh_objects.append(trimesh_object)
#         # chain trimesh objects
#         scene_mesh = trimesh.util.concatenate(trimesh_objects)
#         # Create a scene and add the triangle mesh
#         scene = o3d.t.geometry.RaycastingScene()
#         vertex_positions = scene_mesh.vertices
#         triangle_indices = scene_mesh.faces
#         vertex_positions = o3d.core.Tensor(vertex_positions, dtype=o3d.core.Dtype.Float32)
#         triangle_indices = o3d.core.Tensor(triangle_indices, dtype=o3d.core.Dtype.UInt32)
#         _ = scene.add_triangles(vertex_positions, triangle_indices)  # we do not need the geometry ID for mesh
#         # create a grid
#         shape = np.ceil((self.bounds_max - self.bounds_min) / resolution).astype(int)
#         steps = (self.bounds_max - self.bounds_min) / shape
#         grid = np.mgrid[self.bounds_min[0]:self.bounds_max[0]:steps[0],
#                         self.bounds_min[1]:self.bounds_max[1]:steps[1],
#                         self.bounds_min[2]:self.bounds_max[2]:steps[2]]
#         grid = grid.reshape(3, -1).T
#         # compute SDF
#         sdf_voxels = scene.compute_signed_distance(grid.astype(np.float32))
#         # convert back to np array
#         sdf_voxels = sdf_voxels.cpu().numpy()
#         # open3d has flipped sign from our convention
#         sdf_voxels = -sdf_voxels
#         sdf_voxels = sdf_voxels.reshape(shape)
#         self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] SDF voxels computed in {time.time() - start:.4f} seconds{bcolors.ENDC}')
#         return sdf_voxels

#     def get_cam_obs(self):
#         self.last_cam_obs = dict()
#         for cam_id in self.cams:
#             self.last_cam_obs[cam_id] = self.cams[cam_id].get_obs()  # each containing rgb, depth, points, seg
#         return self.last_cam_obs
    
#     def register_keypoints(self, keypoints):
#         """
#         Args:
#             keypoints (np.ndarray): keypoints in the world frame of shape (N, 3)
#         Returns:
#             None
#         Given a set of keypoints in the world frame, this function registers them so that their newest positions can be accessed later.
#         """
#         if not isinstance(keypoints, np.ndarray):
#             keypoints = np.array(keypoints)
#         self.keypoints = keypoints
#         self._keypoint_registry = dict()
#         self._keypoint2object = dict()
#         exclude_names = ['wall', 'floor', 'ceiling', 'table', 'fetch', 'robot']
#         for idx, keypoint in enumerate(keypoints):
#             closest_distance = np.inf
#             for obj in self.og_env.scene.objects:
#                 if any([name in obj.name.lower() for name in exclude_names]):
#                     continue
#                 for link in obj.links.values():
#                     for mesh in link.visual_meshes.values():
#                         mesh_prim_path = mesh.prim_path
#                         mesh_type = mesh.prim.GetPrimTypeInfo().GetTypeName()
#                         if mesh_type == 'Mesh':
#                             trimesh_object = mesh_prim_mesh_to_trimesh_mesh(mesh.prim)
#                         else:
#                             trimesh_object = mesh_prim_shape_to_trimesh_mesh(mesh.prim)
#                         world_pose_w_scale = PoseAPI.get_world_pose_with_scale(mesh.prim_path)
#                         trimesh_object.apply_transform(world_pose_w_scale)
#                         points_transformed = trimesh_object.sample(1000)
                        
#                         # find closest point
#                         dists = np.linalg.norm(points_transformed - keypoint, axis=1)
#                         point = points_transformed[np.argmin(dists)]
#                         distance = np.linalg.norm(point - keypoint)
#                         if distance < closest_distance:
#                             closest_distance = distance
#                             closest_prim_path = mesh_prim_path
#                             closest_point = point
#                             closest_obj = obj
#             self._keypoint_registry[idx] = (closest_prim_path, PoseAPI.get_world_pose(closest_prim_path))
#             self._keypoint2object[idx] = closest_obj
#             # overwrite the keypoint with the closest point
#             self.keypoints[idx] = closest_point

#     def get_keypoint_positions(self):
#         """
#         Args:
#             None
#         Returns:
#             np.ndarray: keypoints in the world frame of shape (N, 3)
#         Given the registered keypoints, this function returns their current positions in the world frame.
#         """
#         assert hasattr(self, '_keypoint_registry') and self._keypoint_registry is not None, "Keypoints have not been registered yet."
#         keypoint_positions = []
#         for idx, (prim_path, init_pose) in self._keypoint_registry.items():
#             init_pose = T.pose2mat(init_pose)
#             centering_transform = T.pose_inv(init_pose)
#             keypoint_centered = np.dot(centering_transform, np.append(self.keypoints[idx], 1))[:3]
#             curr_pose = T.pose2mat(PoseAPI.get_world_pose(prim_path))
#             keypoint = np.dot(curr_pose, np.append(keypoint_centered, 1))[:3]
#             keypoint_positions.append(keypoint)
#         return np.array(keypoint_positions)

#     def get_object_by_keypoint(self, keypoint_idx):
#         """
#         Args:
#             keypoint_idx (int): the index of the keypoint
#         Returns:
#             pointer: the object that the keypoint is associated with
#         Given the keypoint index, this function returns the name of the object that the keypoint is associated with.
#         """
#         assert hasattr(self, '_keypoint2object') and self._keypoint2object is not None, "Keypoints have not been registered yet."
#         return self._keypoint2object[keypoint_idx]

#     def get_collision_points(self, noise=True):
#         """
#         Get the points of the gripper and any object in hand.
#         """
#         # add gripper collision points
#         collision_points = []
#         for obj in self.og_env.scene.objects:
#             if 'fetch' in obj.name.lower():
#                 for name, link in obj.links.items():
#                     if 'gripper' in name.lower() or 'wrist' in name.lower():  # wrist_roll and wrist_flex
#                         for collision_mesh in link.collision_meshes.values():
#                             mesh_prim_path = collision_mesh.prim_path
#                             mesh_type = collision_mesh.prim.GetPrimTypeInfo().GetTypeName()
#                             if mesh_type == 'Mesh':
#                                 trimesh_object = mesh_prim_mesh_to_trimesh_mesh(collision_mesh.prim)
#                             else:
#                                 trimesh_object = mesh_prim_shape_to_trimesh_mesh(collision_mesh.prim)
#                             world_pose_w_scale = PoseAPI.get_world_pose_with_scale(mesh_prim_path)
#                             trimesh_object.apply_transform(world_pose_w_scale)
#                             points_transformed = trimesh_object.sample(1000)
#                             # add to collision points
#                             collision_points.append(points_transformed)
#         # add object in hand collision points
#         in_hand_obj = self.robot._ag_obj_in_hand[self.robot.default_arm]
#         if in_hand_obj is not None:
#             for link in in_hand_obj.links.values():
#                 for collision_mesh in link.collision_meshes.values():
#                     mesh_type = collision_mesh.prim.GetPrimTypeInfo().GetTypeName()
#                     if mesh_type == 'Mesh':
#                         trimesh_object = mesh_prim_mesh_to_trimesh_mesh(collision_mesh.prim)
#                     else:
#                         trimesh_object = mesh_prim_shape_to_trimesh_mesh(collision_mesh.prim)
#                     world_pose_w_scale = PoseAPI.get_world_pose_with_scale(collision_mesh.prim_path)
#                     trimesh_object.apply_transform(world_pose_w_scale)
#                     points_transformed = trimesh_object.sample(1000)
#                     # add to collision points
#                     collision_points.append(points_transformed)
#         collision_points = np.concatenate(collision_points, axis=0)
#         return collision_points

#     def reset(self):
#         self.og_env.reset()
#         self.robot.reset()
#         for _ in range(5): self._step()
#         self.open_gripper()
#         # moving arm to the side to unblock view 
#         ee_pose = self.get_ee_pose()
#         ee_pose[:3] += np.array([0.0, -0.2, -0.1])
#         action = np.concatenate([ee_pose, [self.get_gripper_null_action()]])
#         self.execute_action(action, precise=True)
#         self.video_cache = []
#         print(f'{bcolors.HEADER}Reset done.{bcolors.ENDC}')

#     def is_grasping(self, candidate_obj=None):
#         return self.robot.is_grasping(candidate_obj=candidate_obj) == IsGraspingState.TRUE

#     def get_ee_pose(self):
#         ee_pos, ee_xyzw = (self.robot.get_eef_position(), self.robot.get_eef_orientation())
#         ee_pose = np.concatenate([ee_pos, ee_xyzw])  # [7]
#         return ee_pose

#     def get_ee_pos(self):
#         return self.get_ee_pose()[:3]

#     def get_ee_quat(self):
#         return self.get_ee_pose()[3:]
    
    # def get_arm_joint_postions(self):
    #     assert isinstance(self.robot, Fetch), "The IK solver assumes the robot is a Fetch robot"
    #     arm = self.robot.default_arm
    #     dof_idx = np.concatenate([self.robot.trunk_control_idx, self.robot.arm_control_idx[arm]])
    #     arm_joint_pos = self.robot.get_joint_positions()[dof_idx]
    #     return arm_joint_pos

#     def close_gripper(self):
#         """
#         Exposed interface: 1.0 for closed, -1.0 for open, 0.0 for no change
#         Internal OG interface: 1.0 for open, 0.0 for closed
#         """
#         if self.last_og_gripper_action == 0.0:
#             return
#         action = np.zeros(12)
#         action[10:] = [0, 0]  # gripper: float. 0. for closed, 1. for open.
#         for _ in range(30):
#             self._step(action)
#         self.last_og_gripper_action = 0.0

#     def open_gripper(self):
#         if self.last_og_gripper_action == 1.0:
#             return
#         action = np.zeros(12)
#         action[10:] = [1, 1]  # gripper: float. 0. for closed, 1. for open.
#         for _ in range(30):
#             self._step(action)
#         self.last_og_gripper_action = 1.0

#     def get_last_og_gripper_action(self):
#         return self.last_og_gripper_action
    
#     def get_gripper_open_action(self):
#         return -1.0
    
#     def get_gripper_close_action(self):
#         return 1.0
    
#     def get_gripper_null_action(self):
#         return 0.0
    
#     def compute_target_delta_ee(self, target_pose):
#         target_pos, target_xyzw = target_pose[:3], target_pose[3:]
#         ee_pose = self.get_ee_pose()
#         ee_pos, ee_xyzw = ee_pose[:3], ee_pose[3:]
#         pos_diff = np.linalg.norm(ee_pos - target_pos)
#         rot_diff = angle_between_quats(ee_xyzw, target_xyzw)
#         return pos_diff, rot_diff

#     def execute_action(
#             self,
#             action,
#             precise=True,
#         ):
#             """
#             Moves the robot gripper to a target pose by specifying the absolute pose in the world frame and executes gripper action.

#             Args:
#                 action (x, y, z, qx, qy, qz, qw, gripper_action): absolute target pose in the world frame + gripper action.
#                 precise (bool): whether to use small position and rotation thresholds for precise movement (robot would move slower).
#             Returns:
#                 tuple: A tuple containing the position and rotation errors after reaching the target pose.
#             """
#             if precise:
#                 pos_threshold = 0.03
#                 rot_threshold = 3.0
#             else:
#                 pos_threshold = 0.10
#                 rot_threshold = 5.0
#             action = np.array(action).copy()
#             assert action.shape == (8,)
#             target_pose = action[:7]
#             gripper_action = action[7]

#             # ======================================
#             # = status and safety check
#             # ======================================
#             if np.any(target_pose[:3] < self.bounds_min) \
#                  or np.any(target_pose[:3] > self.bounds_max):
#                 print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Target position is out of bounds, clipping to workspace bounds{bcolors.ENDC}')
#                 target_pose[:3] = np.clip(target_pose[:3], self.bounds_min, self.bounds_max)

#             # ======================================
#             # = interpolation
#             # ======================================
#             current_pose = self.get_ee_pose()
#             pos_diff = np.linalg.norm(current_pose[:3] - target_pose[:3])
#             rot_diff = angle_between_quats(current_pose[3:7], target_pose[3:7])
#             pos_is_close = pos_diff < self.interpolate_pos_step_size
#             rot_is_close = rot_diff < self.interpolate_rot_step_size
#             if pos_is_close and rot_is_close:
#                 self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Skipping interpolation{bcolors.ENDC}')
#                 pose_seq = np.array([target_pose])
#             else:
#                 num_steps = get_linear_interpolation_steps(current_pose, target_pose, self.interpolate_pos_step_size, self.interpolate_rot_step_size)
#                 pose_seq = linear_interpolate_poses(current_pose, target_pose, num_steps)
#                 self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Interpolating for {num_steps} steps{bcolors.ENDC}')

#             # ======================================
#             # = move to target pose
#             # ======================================
#             # move faster for intermediate poses
#             intermediate_pos_threshold = 0.10
#             intermediate_rot_threshold = 5.0
#             for pose in pose_seq[:-1]:
#                 self._move_to_waypoint(pose, intermediate_pos_threshold, intermediate_rot_threshold)
#             # move to the final pose with required precision
#             pose = pose_seq[-1]
#             self._move_to_waypoint(pose, pos_threshold, rot_threshold, max_steps=20 if not precise else 40) 
#             # compute error
#             pos_error, rot_error = self.compute_target_delta_ee(target_pose)
#             self.verbose and print(f'\n{bcolors.BOLD}[environment.py | {get_clock_time()}] Move to pose completed (pos_error: {pos_error}, rot_error: {np.rad2deg(rot_error)}){bcolors.ENDC}\n')

#             # ======================================
#             # = apply gripper action
#             # ======================================
#             if gripper_action == self.get_gripper_open_action():
#                 self.open_gripper()
#             elif gripper_action == self.get_gripper_close_action():
#                 self.close_gripper()
#             elif gripper_action == self.get_gripper_null_action():
#                 pass
#             else:
#                 raise ValueError(f"Invalid gripper action: {gripper_action}")
            
#             return pos_error, rot_error
    
#     def sleep(self, seconds):
#         start = time.time()
#         while time.time() - start < seconds:
#             self._step()
    
#     def save_video(self, save_path=None):
#         save_dir = os.path.join(os.path.dirname(__file__), 'videos')
#         os.makedirs(save_dir, exist_ok=True)
#         if save_path is None:
#             save_path = os.path.join(save_dir, f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.mp4')
#         video_writer = imageio.get_writer(save_path, fps=30)
#         for rgb in self.video_cache:
#             video_writer.append_data(rgb)
#         video_writer.close()
#         return save_path

#     # ======================================
#     # = internal functions
#     # ======================================
#     def _check_reached_ee(self, target_pos, target_xyzw, pos_threshold, rot_threshold):
#         """
#         this is supposed to be for true ee pose (franka hand) in robot frame
#         """
#         current_pos = self.robot.get_eef_position()
#         current_xyzw = self.robot.get_eef_orientation()
#         current_rotmat = T.quat2mat(current_xyzw)
#         target_rotmat = T.quat2mat(target_xyzw)
#         # calculate position delta
#         pos_diff = (target_pos - current_pos).flatten()
#         pos_error = np.linalg.norm(pos_diff)
#         # calculate rotation delta
#         rot_error = angle_between_rotmat(current_rotmat, target_rotmat)
#         # print status
#         self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}]  Curr pose: {current_pos}, {current_xyzw} (pos_error: {pos_error.round(4)}, rot_error: {np.rad2deg(rot_error).round(4)}){bcolors.ENDC}')
#         self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}]  Goal pose: {target_pos}, {target_xyzw} (pos_thres: {pos_threshold}, rot_thres: {rot_threshold}){bcolors.ENDC}')
#         if pos_error < pos_threshold and rot_error < np.deg2rad(rot_threshold):
#             self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] OSC pose reached (pos_error: {pos_error.round(4)}, rot_error: {np.rad2deg(rot_error).round(4)}){bcolors.ENDC}')
#             return True, pos_error, rot_error
#         return False, pos_error, rot_error

#     def _move_to_waypoint(self, target_pose_world, pos_threshold=0.02, rot_threshold=3.0, max_steps=10):
#         pos_errors = []
#         rot_errors = []
#         count = 0
#         while count < max_steps:
#             reached, pos_error, rot_error = self._check_reached_ee(target_pose_world[:3], target_pose_world[3:7], pos_threshold, rot_threshold)
#             pos_errors.append(pos_error)
#             rot_errors.append(rot_error)
#             if reached:
#                 break
#             # convert world pose to robot pose
#             target_pose_robot = np.dot(self.world2robot_homo, T.convert_pose_quat2mat(target_pose_world))
#             # convert to relative pose to be used with the underlying controller
#             relative_position = target_pose_robot[:3, 3] - self.robot.get_relative_eef_position()
#             relative_quat = T.quat_distance(T.mat2quat(target_pose_robot[:3, :3]), self.robot.get_relative_eef_orientation())
#             assert isinstance(self.robot, Fetch), "this action space is only for fetch"
#             action = np.zeros(12)  # first 3 are base, which we don't use
#             action[4:7] = relative_position
#             action[7:10] = T.quat2axisangle(relative_quat)
#             action[10:] = [self.last_og_gripper_action, self.last_og_gripper_action]
#             # step the action
#             _ = self._step(action=action)
#             count += 1
#         if count == max_steps:
#             print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] OSC pose not reached after {max_steps} steps (pos_error: {pos_errors[-1].round(4)}, rot_error: {np.rad2deg(rot_errors[-1]).round(4)}){bcolors.ENDC}')

#     def _step(self, action=None):
#         if hasattr(self, 'disturbance_seq') and self.disturbance_seq is not None:
#             next(self.disturbance_seq)
#         if action is not None:
#             self.og_env.step(action)
#         else:
#             og.sim.step()
#         cam_obs = self.get_cam_obs()
#         rgb = cam_obs[1]['rgb']
#         if len(self.video_cache) < self.config['video_cache_size']:
#             self.video_cache.append(rgb)
#         else:
#             self.video_cache.pop(0)
#             self.video_cache.append(rgb)
#         self.step_counter += 1

#     def _initialize_cameras(self, cam_config):
#         """
#         ::param poses: list of tuples of (position, orientation) of the cameras
#         """
#         self.cams = dict()
#         for cam_id in cam_config:
#             cam_id = int(cam_id)
#             self.cams[cam_id] = OGCamera(self.og_env, cam_config[cam_id])
#         for _ in range(10): og.sim.render()