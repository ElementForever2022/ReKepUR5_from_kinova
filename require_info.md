# Robot info needed from Franka arm:

    self.current_joint_angles = np.zeros(7)
    self.current_ee_pose = np.array([0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
    self.current_eef_position = np.array([0.5, 0.0, 0.5]) 
    self.world2robot_homo = np.eye(4)  
    self.gripper_state = 0.0

write into 'robot_state.json'

# Joint information
- joint_positions          # Current joint angles for all 7 joints
- joint_velocities        # Current joint velocities 
- joint_torques          # Joint torque readings
- joint_limits           # Position and velocity limits for each joint

# End effector information  
- ee_position           # Current end effector position in Cartesian space
- ee_orientation        # Current end effector orientation (quaternion)

# Gripper information
- gripper_width         # Current gripper opening width ?
- gripper_state        # float: Current state (moving/grasping/idle)

# Safety information
- collision_status     # Whether robot is in collision
- safety_status       # Overall safety status
- errors             # Any active error states

# Control modes
- control_mode        # Current control mode (position/velocity/torque)
- operation_mode     # Current operation mode (auto/manual)

# Misc
- robot_mode         # Overall robot state (idle/moving/error)
- temperature       # Joint motor temperatures
- tool_frame       # Current tool frame transformation
