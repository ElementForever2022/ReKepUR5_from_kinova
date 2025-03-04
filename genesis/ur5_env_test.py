# import necessary libs
import genesis as gs

import time

# initialize simulation scene
gs.init(backend=gs.cuda)

# create scene
scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -3.5, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    show_viewer = True,
)

# add entitites
# the plane
plane = scene.add_entity(gs.morphs.Plane())
# the robot arm(with gripper)
ur5_robot = scene.add_entity(
    gs.morphs.URDF(
        file='/home/ur5/rekep/ReKepUR5_from_kinova/genesis/ur_description/urdf/ur5e_with_gripper.urdf',
        fixed=True,
        pos   = (1.0, 1.0, 0.0), # initial base position
        euler = (0, 0, 0),
    )
)



# build the scene
scene.build()

# cnofigure joints
joint_names = [
    'shoulder_pan_joint', # base joint
    'shoulder_lift_joint', # shoulder joint
    'elbow_joint', # elbow joint
    'wrist_1_joint', # wrist 1 joint
    'wrist_2_joint', # wrist 2 joint
    'wrist_3_joint', # wrist 3 joint
    # 'arm_gripper_joint' # gripper(85)
]

dofs_idx = [ur5_robot.get_joint(name).dof_idx_local for name in joint_names]
print(dofs_idx)
time.sleep(5)
for i in range(10000):
    scene.step()