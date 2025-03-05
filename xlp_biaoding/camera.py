#首先运行这个，获取多张图像

import sys
import numpy as np

import pyrealsense2 as rs

import cv2
import time, datetime
import os
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image

class Camera:
    def __init__(self, width=640, height=480, fps=30):  
        # 图片格式根据项目要求的输入图片分辨率进行更改

        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, fps)
        
        # 红外图 不需要
        # self.config.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, fps)
        # self.config.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, fps)
        
        self.pipeline.start(self.config)  # 获取图像视频流

        print("start camera...")
        time.sleep(2)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()  # 获得frame (包括彩色，深度图)
        colorizer = rs.colorizer()  # 创建伪彩色图对象

        # 滤波之类的提高棋盘格识别精度的图像处理方法
        # decimation = rs.decimation_filter()
        # spatial = rs.spatial_filter()
        # temporal = rs.temporal_filter()
        # hole_filling = rs.hole_filling_filter()
        # depth_to_disparity = rs.disparity_transform(True)
        # disparity_to_depth = rs.disparity_transform(False)

        # 创建对齐对象
        align_to = rs.stream.color  # rs.align允许我们执行深度帧与其他帧的对齐
        align = rs.align(align_to)  # “align_to”是我们计划对齐深度帧的流类型。
        aligned_frames = align.process(frames)
        # 获取对齐的帧
        color_frame = aligned_frames.get_color_frame()
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame是对齐的深度图

        color_image = np.asanyarray(color_frame.get_data())
        depthx_image = np.asanyarray(aligned_depth_frame.get_data())  # 对其的原始深度图
        colorizer_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

        #color_image：纯彩色图
        #depthx_image：对应原始深度数据（还没伪彩）
        #colorizer_depth：伪彩色深度图，可用于可视化
        #aligned_depth_frame：RealSense 提供的深度帧对象，可用于后续获取内参、单位等
        return color_image, depthx_image, colorizer_depth, aligned_depth_frame

    def release(self):
        self.pipeline.stop()


def main():
    # rospy.init_node('camera')

    #取决于项目需要的输入图像的分辨率
    w, h, fps = 640, 480, 30
    # w, h, fps = 1280, 720, 15

    cam = Camera(w, h, fps)
    # bridge = CvBridge()
    # image_pub = rospy.Publisher('camera/color/image_raw', Image, queue_size=10)
    # rate = rospy.Rate(10)
    print('退出请按q')
    
    frame_index = 0
    while True:
        
        color_image, depthxy_image, colorizer_depth, _ = cam.get_frame()   # 读取图像帧，包括RGB图和深度图

        # ros相关，不需要
        # ros_image = bridge.cv2_to_imgmsg(color_image, encoding="bgr8")  # 转换成ros的msg格式
        # image_pub.publish(ros_image)

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)

        #显示一些额外的图像窗口，比如伪彩色深度图 (colorizer_depth)
        # cv2.imshow('Filtered Depth Image', colorizer_depth)
        # images = np.hstack((color_image, depthxy_image))
        # cv2.imshow('RealSense', images)

        key = cv2.waitKey(1)

        if key & 0xFF == ord('s'):
        # print('...录制视频中...')
        # 保存图像帧
        # wr.write(color_image)  # 保存RGB图像帧
        # wr_colordepth.write(colorizer_depth)  # 保存相机自身着色深度图
        # res, depth16_image = cv2.imencode('.png', depthxy_image)  # 深度图解码方式一：点云小，但是出错
        # depth16_image = cv2.imencode('.png', depthxy_image)[1]  # 深度图解码方式二：文件较大，测试稳定
        # depth_map_name = str(id).zfill(5) + '_depth.png'
        # wr_depth[str(idx).zfill(5)] = depth16_image          #  储存方法：1 前3帧和没11帧出现高质量点云，其他错误
        # wr_depth[depth_map_name] = depth16_image  # 储存方法：2 所有点云准确，但是点云质量不高
            image_file_name = f'/home/ur5/rekep/ReKepUR5_from_kinova/xlp_biaoding/img/rgb_frame_{frame_index}_{datetime.datetime.now()}.png'
            # image_file_name = f'/home/user/113/rgb_frame_{frame_index}_{datetime.datetime.now()}.png'
            cv2.imwrite(image_file_name, color_image)
            print(f'Image(frame no. {frame_index}) saverd to {image_file_name}')
        # cv2.imwrite('depth.png', depthxy_image)
        # print(depthxy_image[400][640]/1000)
        # idx += 1
        # id += 1
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            print('...录制结束/直接退出...')
            break

        #ros相关，不需要
        # rate.sleep()
    
    # wr.release()  # 录制完毕，释放对象
    # wr_colordepth.release()
    # wr_depth.close()
        frame_index += 1
    cam.release()
    # print(f'若保存视频，则视频保存在：{video_path}')

if __name__ == "__main__":
    main()

