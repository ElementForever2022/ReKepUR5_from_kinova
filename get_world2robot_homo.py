import pyrealsense2 as rs
import numpy as np

def get_pitch_roll_yaw():
    # Create a context object. This object owns the handles to all connected realsense devices.
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.pose)

    # Start streaming with the configuration
    pipeline.start(config)

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            pose_frame = frames.get_pose_frame()

            if pose_frame:
                pose_data = pose_frame.get_pose_data()

                # Extract the rotation quaternion
                w = pose_data.rotation.w
                x = pose_data.rotation.x
                y = pose_data.rotation.y
                z = pose_data.rotation.z

                # Convert quaternion to Euler angles
                pitch, roll, yaw = quaternion_to_euler_angle(w, x, y, z)

                print(f"Pitch: {pitch}, Roll: {roll}, Yaw: {yaw}")
                return pitch, roll, yaw

    finally:
        # Stop streaming
        pipeline.stop()

def quaternion_to_euler_angle(w, x, y, z):
    # Convert quaternion to Euler angles (pitch, roll, yaw)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return pitch, roll, yaw

if __name__ == "__main__":
    get_pitch_roll_yaw()