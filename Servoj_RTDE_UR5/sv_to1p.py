import sys
import logging
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import time
from min_jerk_planner_translation import PathPlanTranslation

def list_to_setp(setp, list):
    for i in range(6):
        setp.__dict__[f"input_double_register_{i}"] = list[i]
    return setp

ROBOT_HOST = '192.168.0.201'
ROBOT_PORT = 30004
config_filename = 'control_loop_configuration.xml'
FREQUENCY = 500
desired_pose = [-0.503, -0.0088, 0.31397, 1.266, -2.572, -0.049]


logging.getLogger().setLevel(logging.INFO)

conf = rtde_config.ConfigFile(config_filename)
state_names, state_types = conf.get_recipe('state')
setp_names, setp_types = conf.get_recipe('setp')
watchdog_names, watchdog_types = conf.get_recipe('watchdog')

con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
connection_state = con.connect()

while connection_state != 0:
    time.sleep(0.5)
    connection_state = con.connect()
print("---------------Successfully connected to the robot-------------\n")

con.get_controller_version()

con.send_output_setup(state_names, state_types, FREQUENCY)
setp = con.send_input_setup(setp_names, setp_types)
watchdog = con.send_input_setup(watchdog_names, watchdog_types)

setp.input_double_register_0 = 0
setp.input_double_register_1 = 0
setp.input_double_register_2 = 0
setp.input_double_register_3 = 0
setp.input_double_register_4 = 0
setp.input_double_register_5 = 0

setp.input_bit_registers0_to_31 = 0
# watchdog.input_int_register_0 = 0

if not con.send_start():
    sys.exit()

state = con.receive()
tcp = state.actual_TCP_pose
print("Current TCP pose:", tcp)

print("-------Executing servoJ to point 1 -----------\n")
watchdog.input_int_register_0 = 2
con.send(watchdog)

trajectory_time = 8
planner = PathPlanTranslation(tcp, desired_pose, trajectory_time)
orientation_const = tcp[3:]

t_start = time.time()
while time.time() - t_start < trajectory_time:
    state = con.receive()
    t_current = time.time() - t_start

    if state.runtime_state > 1 and t_current <= trajectory_time:
        position_ref, lin_vel_ref, acceleration_ref = planner.trajectory_planning(t_current)
        pose = position_ref.tolist() + orientation_const
        list_to_setp(setp, pose)
        con.send(setp)

print(f"It took {time.time()-t_start}s to execute the servoJ to point 1")
print('Final TCP pose:', con.receive().actual_TCP_pose)

# watchdog.input_int_register_0 = 3
con.send(watchdog)

con.send_pause()
con.disconnect()



