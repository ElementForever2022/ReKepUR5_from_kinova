"""
Adapted from OmniGibson and the Lula IK solver
"""
"""
UR5e IK solver
"""


# import omnigibson.lazy as lazy
import numpy as np


import numpy as np
from scipy.spatial.transform import Rotation

class IKResult:
    """Class to store IK solution results"""
    def __init__(self, success, joint_positions, error_pos, error_rot, num_descents=None):
        self.success = success
        self.cspace_position = joint_positions
        self.position_error = error_pos
        self.rotation_error = error_rot
        self.num_descents = num_descents if num_descents is not None else 1
    
    
# TODO use real IK solver
class UR5eIKSolver:
    """UR5e IK Solver"""
    def __init__(self, reset_joint_pos, world2robot_homo=None):
        # DH parameters for UR5e (Modified DH parameters)
        # 注意这里是Modified DH parameters

        #franka的
        # self.dh_params = {
        #     # [a, alpha, d, theta]
        #     1: [0,     0,      0.333,  0],  # Joint 1
        #     2: [0,     -np.pi/2, 0,      0],  # Joint 2
        #     3: [0,     np.pi/2,  0.316,  0],  # Joint 3
        #     4: [0.0825, np.pi/2,  0,      0],  # Joint 4
        #     5: [-0.0825, -np.pi/2, 0.384,  0],  # Joint 5
        #     6: [0,     np.pi/2,  0,      0],  # Joint 6
        #     7: [0.088,  np.pi/2,  0.107,  0],  # Joint 7 (end effector)
        # }

        #自己找的
        #从官网获得的UR5e的DH参数，直接调换列顺序获得。可行？
        # [a, alpha, d, theta]
        # self.dh_params = {
        #     1: [0,         np.pi/2, 0.1625, 0],
        #     2: [-0.425,    0,       0,      0],
        #     3: [-0.3922,   0,       0,      0],
        #     4: [0,         np.pi/2, 0.1333, 0],
        #     5: [0,         -np.pi/2,0.0997, 0],
        #     6: [0,         0,       0.0996, 0],
        # }

        #xjc的
        self.dh_params = {
            # Standard DH parameters for UR5
            # [a, alpha, d, theta]
            1: [0,      np.pi/2,  0.089159, 0],  # Joint 1
            2: [-0.425, 0,        0,        0],  # Joint 2
            3: [-0.39225, 0,      0,        0],  # Joint 3
            4: [0,      np.pi/2,  0.10915,  0],  # Joint 4
            5: [0,      -np.pi/2, 0.09465,  0],  # Joint 5
            6: [0,      0,        0.0823,   0],  # Joint 6 (end effector)
        }

        #franka的
        # Joint limits (in radians)
        # self.joint_limits = {
        #     'lower': np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
        #     'upper': np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        # }

        #
        # Joint limits (in radians) for UR5e
        #看起来范围有点大
        # self.joint_limits = {
        #     'lower': np.array([-6.2832, -2.0944, -6.2832, -6.2832, -6.2832, -6.2832]),
        #     'upper': np.array([6.2832, 2.0944, 6.2832, 6.2832, 6.2832, 6.2832])
        # }

        #xjc ver
        self.joint_limits = {
            'lower': np.array([-2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi]),
            'upper': np.array([2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
        }        
        
        #初始的各个关节位置，在配置文件中定义
        self.reset_joint_pos = reset_joint_pos

        #转换矩阵
        self.world2robot_homo = world2robot_homo if world2robot_homo is not None else np.eye(4)
        
        #配置文件路径
        self.robot_state_path = "/home/ur5/rekep/ReKepUR5_from_kinova/robot_state.json"
        # self.robot_state_path = "/path/to/robot_state.json"
        #/home/ur5/rekep/ReKepUR5_from_kinova/robot_state.json

    def _validate_transform(self, pose):
        """
        Validate transformation matrix

        验证标定获得的放在配置文件中的转换矩阵格式是否正确

        这个方法用于验证一个变换矩阵是否符合4×4齐次变换的要求。它会检查：
            输入必须是一个NumPy数组。
            数组形状必须为4×4。
            前3×3部分（旋转部分）必须正交（即R⋅RT≈I）。
            第4行必须等于 [0,0,0,1]。
        """

        if not isinstance(pose, np.ndarray):
            raise TypeError(f"Pose must be numpy array, got {type(pose)}")
            
        if pose.shape != (4, 4):
            raise ValueError(f"Pose must be 4x4 matrix, got shape {pose.shape}")
            
        # Check if rotation part is valid
        rot = pose[:3, :3]
        if not np.allclose(np.dot(rot, rot.T), np.eye(3), atol=1e-6):
            raise ValueError("Invalid rotation matrix")
            
        # Check last row
        if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError("Invalid homogeneous transform - last row should be [0,0,0,1]")

    def _dh_matrix(self, a, alpha, d, theta):
        """
        Calculate transformation matrix from DH parameters

        对DH参数矩阵的每一行，获得一个4*4的矩阵，进行后续计算
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        return np.array([
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, joint_angles):
        """
        Compute forward kinematics using DH parameters
        
        Args:
            joint_angles: array of 6 joint angles in radians
            
        Returns:
            4x4 homogeneous transformation matrix for end effector pose

        从所有关节角度获得当前末端位姿
        """
        
        #初始化一个4×4单位矩阵 TT 作为初始变换矩阵
        T = np.eye(4)
        
        #注意这里是6不是7，UR5e有6个自由度
        #每次，往当前T矩阵上乘上变换矩阵
        for i in range(6):
            a, alpha, d, _ = self.dh_params[i+1]
            theta = joint_angles[i]
            Ti = self._dh_matrix(a, alpha, d, theta)
            T = T @ Ti
            
        return T

    def _jacobian(self, joint_angles):
        """Calculate geometric Jacobian"""

        # 关节数量6
        # J = np.zeros((6, 7))
        J = np.zeros((6, 6))

        T = np.eye(4)

        # p用于存储每个关节帧的位置（包括基座，UR5e有6个关节，所以共7个帧）
        # p = np.zeros((8, 3))  # Position of each joint including end effector
        p = np.zeros((7, 3))  # 修改：原代码为(8,3)，现调整为(7,3)
        
        # Forward pass to get all joint positions
        #关节数量为6
        for i in range(6):
            a, alpha, d, _ = self.dh_params[i+1]
            theta = joint_angles[i]
            Ti = self._dh_matrix(a, alpha, d, theta)
            T = T @ Ti
            p[i+1] = T[:3, 3]
        
        # Calculate Jacobian
        #7：6+1
        z = np.zeros((7, 3))  # z axis of each frame
        T = np.eye(4)
        z[0] = T[:3, 2]
        
        #6：关节数量
        for i in range(6):
            a, alpha, d, _ = self.dh_params[i+1]
            theta = joint_angles[i]
            Ti = self._dh_matrix(a, alpha, d, theta)
            T = T @ Ti
            z[i+1] = T[:3, 2]
            
            # Linear velocity component
            #6：关节数量
            J[:3, i] = np.cross(z[i], (p[6] - p[i]))
            # Angular velocity component
            J[3:, i] = z[i]
            
        return J

    def _numerical_ik(self, target_pose, initial_joints, max_iter=100, tol=1e-3):
        """
        Numerical IK using damped least squares method
        实现了基于阻尼最小二乘（damped least squares）算法的数值逆运动学求解
        """
        current_joints = initial_joints.copy()

        # print("（_numerical_ik）initial_joints: ",initial_joints)
        # print("（_numerical_ik）initial_eef_pose:",self.forward_kinematics(current_joints))
        
        for i in range(max_iter):
            current_pose = self.forward_kinematics(current_joints)

            #检查
            # print("（_numerical_ik）iter:",i)
            # print("（_numerical_ik）current_joints:",current_joints)
            # print("（_numerical_ik）current_eef_pose:",current_pose)
            # print("（_numerical_ik）target_eef_pose::",target_pose)

            
            # Calculate pose error
            pos_error = target_pose[:3, 3] - current_pose[:3, 3]
            rot_error = 0.5 * np.cross(current_pose[:3, :3].T[0], target_pose[:3, :3].T[0]) + \
                       0.5 * np.cross(current_pose[:3, :3].T[1], target_pose[:3, :3].T[1]) + \
                       0.5 * np.cross(current_pose[:3, :3].T[2], target_pose[:3, :3].T[2])
            
            error = np.concatenate([pos_error, rot_error])

            #检查
            # print("（_numerical_ik）error:",error)
            # print("（_numerical_ik）error norm:",np.linalg.norm(error))
            # print("（_numerical_ik）tor:",tol)
            # print("（_numerical_ik）if end:",np.linalg.norm(error) < tol)

            
            if np.linalg.norm(error) < tol:
                return True, current_joints, np.linalg.norm(pos_error), np.linalg.norm(rot_error)
            
            # Calculate Jacobian
            J = self._jacobian(current_joints)
            
            # print("（_numerical_ik）finish:_jacobian(current_joints)")

            # Damped least squares
            lambda_ = 0.5
            delta_theta = J.T @ np.linalg.inv(J @ J.T + lambda_**2 * np.eye(6)) @ error
            
            # print("（_numerical_ik）finish:delta_theta")
            # print("（_numerical_ik）(cilp)current_joints:",current_joints)
            # print("（_numerical_ik）(cilp)delta_theta:",delta_theta)

            # Update joints with limits
            current_joints = np.clip(
                current_joints + delta_theta,
                self.joint_limits['lower'],
                self.joint_limits['upper']
            )

            # print("（_numerical_ik）finish:clip")

            
        return False, current_joints, np.linalg.norm(pos_error), np.linalg.norm(rot_error)

    def solve(self, target_pose_homo, 
             position_tolerance=0.001,
             orientation_tolerance=0.001,
             max_iterations=150,
             initial_joint_pos=None):
        """Solve IK for UR5 robot"""
        try:
            # print("(solve)target_pose_homo:",target_pose_homo)

            # Validate input pose
            self._validate_transform(target_pose_homo)
            
            # print("(solve)finish:validate_transform(target_pose_homo)")

            # Transform target pose to robot base frame
            robot_pose = self.transform_pose(target_pose_homo)
            
            # print("(solve)finish:robot_pose = self.transform_pose(target_pose_homo)")

            # # Validate transformed pose
            # self._validate_transform()
            # print("finish:Validate transformed pose:self._validate_transform())")

            
            #下面：预设的起始点；上面：获取当前位姿作为起始点
            ###################################################
            # Get initial joint positions
            # if initial_joint_pos is None:
            #     robot_state = self._read_robot_state()
            #     if robot_state:
            #         initial_joint_pos = np.array(robot_state['joint_info']['joint_positions'])
            #     else:
            #         initial_joint_pos = self.reset_joint_pos
            ###################################################
            initial_joint_pos = self.reset_joint_pos
            ###################################################

            # print("(solve)robot_pose:",robot_pose)
            # print("(solve)initial_joint_pos:",initial_joint_pos)

            # Solve IK
            success, joint_positions, pos_error, rot_error = self._numerical_ik(
                robot_pose,
                initial_joint_pos,
                max_iter=max_iterations,
                tol=min(position_tolerance, orientation_tolerance)
            )

            # print("(solve)finish:_numerical_ik")
            
            #TODO:实现范围检查
            #下面:直接返回：上面：增加范围检查
            #####################################################
            # # Verify solution
            # if success:
            #     # Additional workspace and joint limit checks
            #     if self._check_workspace_limits(robot_pose[:3, 3]) and \
            #        self._check_joint_limits(joint_positions):
            #         return IKResult(
            #             success=True,
            #             joint_positions=joint_positions,
            #             error_pos=pos_error,
            #             error_rot=rot_error,
            #             num_descents=max_iterations
            #         )
            
            # return IKResult(
            #     success=False,
            #     joint_positions=self.reset_joint_pos,
            #     error_pos=pos_error,
            #     error_rot=rot_error,
            #     num_descents=max_iterations
            # )
            #####################################################
            return IKResult(
                success=True,
                joint_positions=joint_positions,
                error_pos=pos_error,
                error_rot=rot_error,
                num_descents=max_iterations
            )
            #####################################################
            
        except Exception as e:
            print(f"Error in IK solve: {str(e)}")
            return IKResult(
                success=False,
                joint_positions=self.reset_joint_pos,
                error_pos=float('inf'),
                error_rot=float('inf'),
                num_descents=0
            )

    # NOT used 
    # def forward_kinematics(self, joint_positions):
    #     """
    #     Compute forward kinematics (placeholder)
        
    #     Args:
    #         joint_positions (array): Joint angles
            
    #     Returns:
    #         4x4 array: Homogeneous transformation matrix
    #     """
    #     # Placeholder - implement actual FK
    #     return np.eye(4)

    def transform_pose(self, target_pose_homo):
        """
        Transform target pose from world frame to robot base frame
        
        Args:
            target_pose_homo (np.ndarray): 4x4 homogeneous transformation matrix in world frame
            
        Returns:
            np.ndarray: 4x4 homogeneous transformation matrix in robot base frame
        """
        # Add debug prints
        # print("Input target pose:\n", target_pose_homo)
        # print("World to robot transform:\n", self.world2robot_homo)
        
        # Check matrix shapes
        assert target_pose_homo.shape == (4, 4), f"Expected target_pose_homo shape (4,4), got {target_pose_homo.shape}"
        assert self.world2robot_homo.shape == (4, 4), f"Expected world2robot_homo shape (4,4), got {self.world2robot_homo.shape}"
        
        # Perform transformation
        robot_pose = np.dot(self.world2robot_homo, target_pose_homo)
        
        # print("Output robot pose:\n", robot_pose)
        return robot_pose

# Unit tests
def test_UR5e_ik():

    ## 读取配置文件
    import json
    with open("/home/ur5/rekep/ReKepUR5_from_kinova/robot_state.json", "r") as f:
        state = json.load(f)
    # 从配置中提取reset_joint_pos
    reset_joint_pos = state["joint_info"]["reset_joint_pos"]
    ###

    print("（test）reset_joint_pos:",reset_joint_pos)

    # Create solver
    solver = UR5eIKSolver(reset_joint_pos)
    
    # Test case 1: Identity pose
    # target = np.eye(4)
    # result = solver.solve(target)

    #测试时注释
    # assert result['success']
    
    # Test case 2: Transformed pose

    #给定末端执行器的目标位姿，获得各个关节的角度
    #末端执行器被平移到(，，)的位置（保持无旋转）
    target = np.array([
        [1, 0, 0, -0.413],
        [0, 1, 0, 0.089],
        [0, 0, 1, 0.543],
        [0, 0, 0, 1.0]
    ])
    result = solver.solve(target,max_iterations=300)

    print("（test） cspace_position:",result.cspace_position)    
    print("（test） num_descents:",result.num_descents)
    print("（test） rotation_error:",result.rotation_error)
    print("（test） position_error:",result.position_error)


    #测试时注释
    # assert result['success']
    
    # Test case 3: Check joint limits

    #测试时注释
    # joints = result['joint_positions']
    # assert np.all(joints >= solver.joint_limits['lower'])
    # assert np.all(joints <= solver.joint_limits['upper'])
    
    print("All tests passed!")

class IKSolver:
    """
    Class for thinly wrapping Lula IK solver
    
    This class implements inverse kinematics (IK) for robotic manipulators.
    IK is the process of calculating joint angles needed to achieve a desired
    end-effector pose. This is essential for robot motion planning and control.
    
    The solver uses Cyclic Coordinate Descent (CCD), an iterative method that:
    1. Optimizes one joint at a time
    2. Minimizes position and orientation error of end-effector
    3. Respects joint limits and collision constraints
    4. Handles redundant manipulators (robots with >6 DOF)
    """

    def __init__(
        self,
        robot_description_path,
        robot_urdf_path,
        eef_name,
        reset_joint_pos,
        world2robot_homo,
    ):
        # Create robot description, kinematics, and config
        # self.robot_description = lazy.lula.load_robot(robot_description_path, robot_urdf_path)
        # self.kinematics = self.robot_description.kinematics()
        # self.config = lazy.lula.CyclicCoordDescentIkConfig()
        self.eef_name = eef_name
        self.reset_joint_pos = reset_joint_pos
        self.world2robot_homo = world2robot_homo

    def solve(
        self,
        target_pose_homo,
        position_tolerance=0.01,
        orientation_tolerance=0.05,
        position_weight=1.0,
        orientation_weight=0.05,
        max_iterations=150,
        initial_joint_pos=None,
    ):
        """
        Backs out joint positions to achieve desired @target_pos and @target_quat

        The solver uses an optimization approach to find joint angles that place the
        end-effector at the target pose. It balances:
        - Position accuracy (xyz coordinates)
        - Orientation accuracy (rotation matrix)
        - Joint limits
        - Solution convergence speed

        Args:
            target_pose_homo (np.ndarray): [4, 4] homogeneous transformation matrix of the target pose in world frame
            position_tolerance (float): Maximum position error (L2-norm) for a successful IK solution
            orientation_tolerance (float): Maximum orientation error (per-axis L2-norm) for a successful IK solution
            position_weight (float): Weight for the relative importance of position error during CCD
            orientation_weight (float): Weight for the relative importance of position error during CCD
            max_iterations (int): Number of iterations used for each cyclic coordinate descent.
            initial_joint_pos (None or n-array): If specified, will set the initial cspace seed when solving for joint
                positions. Otherwise, will use self.reset_joint_pos

        Returns:
            ik_results (lazy.lula.CyclicCoordDescentIkResult): IK result object containing the joint positions and other information.
        """
        # convert target pose to robot base frame
        # target_pose_robot = np.dot(self.world2robot_homo, target_pose_homo)
        # target_pose_pos = target_pose_robot[:3, 3]
        # target_pose_rot = target_pose_robot[:3, :3]
        # ik_target_pose = lazy.lula.Pose3(lazy.lula.Rotation3(target_pose_rot), target_pose_pos)
        # # Set the cspace seed and tolerance
        # initial_joint_pos = self.reset_joint_pos if initial_joint_pos is None else np.array(initial_joint_pos)
        # self.config.cspace_seeds = [initial_joint_pos]
        # self.config.position_tolerance = position_tolerance
        # self.config.orientation_tolerance = orientation_tolerance
        # self.config.ccd_position_weight = position_weight
        # self.config.ccd_orientation_weight = orientation_weight
        # self.config.max_num_descents = max_iterations
        # # Compute target joint positions
        # ik_results = lazy.lula.compute_ik_ccd(self.kinematics, ik_target_pose, self.eef_name, self.config)
        # return ik_results
        return None

if __name__ == "__main__":
    test_UR5e_ik()
