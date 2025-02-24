# -*- coding: utf-8 -*-
import pyrealsense2 as rs
import numpy as np
import os
import argparse
import time

# create a class to take color and depth frames and save them as npy
class PRS:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = 0.001
        self.intrinsics = self.get_intrinsics()

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
    
    def get_frames(self, retries=5):
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
                return color_image, depth_image
            except RuntimeError as e:
                print(f"Error: {e}. Retrying...")
                time.sleep(1)
        raise RuntimeError("Failed to acquire frames after several retries.")
    
    def save_frames(self, color_image, depth_image, data_path, frame_number):
        color_path = os.path.join(data_path, 'color_{:06d}.npy'.format(frame_number))
        depth_path = os.path.join(data_path, 'depth_{:06d}.npy'.format(frame_number))
        np.save(color_path, color_image)
        np.save(depth_path, depth_image)
        return color_path, depth_path
    
    def close(self):
        self.pipeline.stop()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #图片保存的标识和路径
    #在这里设default=N，则获得图像序号为N
    parser.add_argument('--frame_number', type=int, default=5)
    parser.add_argument('--data_path', type=str, default='data')

    args = parser.parse_args()

    #路径
    data_path = "/home/ur5/rekep/Rekep4Real/data"

    prs = PRS()
    try:
        color_image, depth_image = prs.get_frames()

        #保存颜色图和深度图两张
        prs.save_frames(color_image, depth_image, args.data_path, args.frame_number)
    except RuntimeError as e:
        print(f"Failed to get frames: {e}")
    finally:
        prs.close()
    print('Done')