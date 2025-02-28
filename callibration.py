# 手眼标定

import cv2
import numpy as np

import pyrealsense2 as rs

import time, os, datetime

import robotEnv



# class PRS:
#     def __init__(self):
#         self.pipeline = rs.pipeline()
#         self.config = rs.config()
#         self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#         self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#         self.pipeline.start(self.config)
#         self.align = rs.align(rs.stream.color)
#         self.depth_scale = 0.001
#         self.intrinsics = self.get_intrinsics()

#         time.sleep(2)
#         print('camera is ready')

#     def get_intrinsics(self):
#         # D435i 的默认内参（你可以根据实际情况修改这些值）
#         class RS_Intrinsics:
#             def __init__(self):
#                 self.fx = 386.738  # focal length x
#                 self.fy = 386.738
#                 self.ppx = 319.5
#                 self.ppy = 239.5
#         intrinsics = RS_Intrinsics()
#         return intrinsics
    
#     def realtime_shoot(self, shoot_button='s', quit_button='q', save_dir='./chessboards/', retries=5):
#         """
#         实时获取摄像机图像，并且按下特定的按键可以拍照或退出
#         """
#         threshold = 100 # 图片二值化阈值

#         try:
#             while True:
#                 for _ in range(retries):
#                     try:
#                         frames = self.pipeline.wait_for_frames()
#                         aligned_frames = self.align.process(frames)
#                         depth_frame = aligned_frames.get_depth_frame()
#                         color_frame = aligned_frames.get_color_frame()
#                         if not depth_frame or not color_frame:
#                             raise RuntimeError("Could not acquire depth or color frame.")
#                         depth_image = np.asanyarray(depth_frame.get_data())
#                         color_image = np.asanyarray(color_frame.get_data())
#                         break
                        
#                         # print(f'color image shape: {color_image.shape}')
#                         # print(f'depth image shape: {depth_image.shape}')
#                         # return color_image, depth_image
#                     except RuntimeError as e:
#                         print(f"Error: {e}. Retrying...")
#                         time.sleep(1)
#                 else:
#                     raise RuntimeError(f"Failed to acquire frames after {retries} retries.")
#                 # 投射到cv2窗口上面的画面(非保存的画面)
#                 screen = np.zeros((color_image.shape[0], color_image.shape[1]*2, 3),dtype=np.uint8)
#                 # 左边部分是原图
#                 screen[:,:color_image.shape[1],:] = color_image.copy()
#                 # 实时找到画面的棋盘格
#                 gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
#                 # 对图片进行中值滤波
#                 gray = cv2.medianBlur(gray, 5)
#                 # 对图片进行二值化
#                 gray[gray<threshold] = 0
#                 gray[gray>=threshold] = 255
#                 # 将灰度图转换回彩色图(看起来还是黑白的)
#                 gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
#                 # 在屏幕右边绘制用于处理的图片
#                 screen[:,color_image.shape[1]:,:] = gray_3channel
#                 pattern_size = (8,8) # 棋盘格大小是8*8
#                 ret, corners = cv2.findChessboardCorners(gray_3channel, pattern_size, None) # 实时找到画面8*8的棋盘格
#                 if ret:
#                     cv2.drawChessboardCorners(screen, pattern_size, corners, ret)
#                 # 对画面进行一些修饰
#                 position_line1 = (30, 20)  # (x, y)坐标
#                 position_line2 = (30, 50)  # (x, y)坐标
#                 position_line3 = (30, 80)  # (x, y)坐标
#                 position_line4 = (30, 110)  # (x, y)坐标
#                 position_line5 = (30, 140)
#                 # 定义字体、大小、颜色和粗细
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 font_scale = 0.5
#                 color = (255, 255, 255)  # 白色 (B, G, R)
#                 if ret:
#                     color = (255,0,0) # 如果找到了棋盘，则字体为红色
#                 thickness = 1
#                 cv2.rectangle(screen, (position_line1[0],10), (screen.shape[1]//2,position_line5[-1]), (0,0,0), thickness=-1)
#                 cv2.putText(screen,f'press {shoot_button} to save image', position_line1, font, font_scale, color, thickness)
#                 cv2.putText(screen,f'press {quit_button} to quit', position_line2, font, font_scale, color, thickness)
#                 cv2.putText(screen,f'Bi-value threshold: {threshold}, press p to increase and press m to decrease', position_line3, font, font_scale, color, thickness)
#                 cv2.putText(screen,f'RGB resolution: {color_image.shape}', position_line4, font, font_scale, color, thickness)
#                 cv2.putText(screen,f'depth resolution: {depth_image.shape}', position_line5, font, font_scale, color, thickness)
                
#                 # 实时展示画面
#                 cv2.imshow('realtime camera', screen)
#                 key_presseed = cv2.waitKey(1)
#                 if key_presseed & 0xff==ord('q'):
#                     # 按下q就退出
#                     break
#                 elif key_presseed & 0xff==ord('s') and ret:
#                     # 按下s就保存
#                     current_time = datetime.datetime.now()
#                     timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
#                     cv2.imwrite(f'{save_dir}rgb_{timestamp}.png', gray_3channel) # 要保存的是可以识别出棋盘的画面
#                     np.save(f'{save_dir}depth_{timestamp}.npy', depth_image)
#                     print('Image saved!')
#                 elif key_presseed & 0xff==ord('p'):
#                     threshold += 1
#                 elif key_presseed & 0xff==ord('m'):
#                     threshold -= 1
#                 else:
#                     # 其他按键保留
#                     continue
#         finally:
#             cv2.destroyAllWindows()
#     def get_frames(self, retries=5):
#         for _ in range(retries):
#             try:
#                 frames = self.pipeline.wait_for_frames()
#                 aligned_frames = self.align.process(frames)
#                 depth_frame = aligned_frames.get_depth_frame()
#                 color_frame = aligned_frames.get_color_frame()
#                 if not depth_frame or not color_frame:
#                     raise RuntimeError("Could not acquire depth or color frame.")
#                 depth_image = np.asanyarray(depth_frame.get_data())
#                 color_image = np.asanyarray(color_frame.get_data())
                
#                 print(f'color image shape: {color_image.shape}')
#                 print(f'depth image shape: {depth_image.shape}')
#                 return color_image, depth_image
#             except RuntimeError as e:
#                 print(f"Error: {e}. Retrying...")
#                 time.sleep(1)
#         raise RuntimeError("Failed to acquire frames after several retries.")
    
#     def save_frames(self, color_image, depth_image, data_path, frame_number):
#         color_path = os.path.join(data_path, 'color_{:06d}.npy'.format(frame_number))
#         depth_path = os.path.join(data_path, 'depth_{:06d}.npy'.format(frame_number))
#         np.save(color_path, color_image)
#         np.save(depth_path, depth_image)
        
#         # 新增：保存为 PNG 格式
#         png_path = os.path.join(data_path, 'color_{:06d}.png'.format(frame_number))
#         cv2.imwrite(png_path, color_image)
        
#         return color_path, depth_path, png_path
    
#     def close(self):
#         self.pipeline.stop()
class Callibrator(object):
    def __init__(self):
        self.side_length = 98.9/6/1000 # 边长，单位为m

        # 内参矩阵
        class RS_Intrinsics:
            def __init__(self):
                self.fx = 386.738/1000  # focal length x:m
                self.fy = 386.738/1000
                self.ppx = 319.5 # 光心坐标
                self.ppy = 239.5
        intrinsics = RS_Intrinsics()
        self.intrinsics = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0,0,1]
        ])

    def calc_Mcc(self, timestamp):
        """
        计算数据文件夹内的某一个位姿对应的转移矩阵Mcc
        return Mcc
        """
        # 读取rgb图片和深度图
        color_image = cv2.imread(f'./chessboards/rgb_{timestamp}.png')
        depth_image = np.load(f'./chessboards/depth_{timestamp}.npy')
        
        # 得到rgb图片中的那些角点
        ret, corners = cv2.findChessboardCornersSB(color_image, (7,7), None)
        if ret:
            # 将这些角点转换为相机坐标系下的坐标
            corner_xy = np.round(np.array(corners)[:,0,:]).astype(np.int32)
            corner_zs = []
            valid_coordinates_cam = []
            valid_coordinates_cal = []
            for i,xy in enumerate(corner_xy):
                if xy[0]>=480 or xy[1]>=640:
                    continue
                z = depth_image[xy[0],xy[1]]
                if z != 0:
                    corner_zs.append(z)

                    u,v = xy[0],xy[1] # 像素坐标
                    Z = z # 像素深度
                    K = self.intrinsics # 内参矩阵

                    x_cam, y_cam, z_cam = self.pixel2cam(u,v,Z,K) # 计算像素坐标对应的相机坐标

                    valid_coordinates_cam.append([x_cam, y_cam, z_cam ,1])

                    cal_x = i%7*self.side_length
                    cal_y = i//7*self.side_length
                    cal_z = 0
                    valid_coordinates_cal.append([cal_x, cal_y, cal_z,1])

            # 讲这些坐标转换为np array格式
            valid_coordinates_cal = np.array(valid_coordinates_cal)
            valid_coordinates_cam = np.array(valid_coordinates_cam)
            
            # 计算线性变换矩阵
            Mcc = self.calc_transformation_matrix(valid_coordinates_cam, valid_coordinates_cal)
            return Mcc # 返回Mcc
            # print(Mcc@valid_coordinates_cam.T-valid_coordinates_cal.T)
        else:
            raise Exception('这个图片没有找到棋盘!!!!!')
    def calc_Mbt(self,timestamp):
        """
        由某一个时间戳读出此处的tcp, 进而计算出Mbt (单位:m)
        """
        tcp = np.load(f'./chessboards/tcp_{timestamp}.npy')

        xyz, rxryrz = tcp[:3],tcp[3:]
        rotation_matrix, _ = cv2.Rodrigues(rxryrz)

        Mbt = np.vstack([
            np.hstack([rotation_matrix, xyz.reshape(-1,1)]),
            [0,0,0,1]
        ])
        print(Mbt)
        return Mbt

    @staticmethod
    def pixel2cam(u,v,Z,K):
        """
        将像素坐标转换为相机坐标
        u, v: pixel coordinate
        Z: depth
        K: intri

        return: (x,y,z)
        """
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        return X, Y, Z

    @staticmethod
    def calc_transformation_matrix(source, target):
        """
        根据两个点集计算出线性变换矩阵
        source: np.array N*4, 补1
        target: np.array N*4, 补1

        return: 4*4线性变换矩阵, source@M^T=target
        """
        assert source.shape == target.shape

        source_ = source[:,:-1]
        target_ = target[:,:-1]

        # 计算质心
        centroid_source = np.mean(source_, axis=0)
        centroid_target = np.mean(target_, axis=0)

        # 去中心化
        source_centered = source_ - centroid_source
        target_centered = target_ - centroid_target

        # 计算协方差矩阵
        H = np.dot(source_centered.T, target_centered)

        # SVD分解
        U, S, Vt = np.linalg.svd(H)

        # 计算旋转矩阵
        R = np.dot(Vt.T, U.T)

        # 处理旋转矩阵的反射情况
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)

        # 计算平移向量
        t = centroid_target - np.dot(R, centroid_source)

        # 构造4x4变换矩阵
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = R
        transformation_matrix[:3, 3] = t

        return transformation_matrix
    def eye_to_hand_calibration(self, M_bt_list, M_cc_list):
        """
        计算眼在手外标定的基座到相机的变换矩阵。

        参数:
        - M_bt_list: 基座到工具的变换矩阵列表。
        - M_cc_list: 相机到棋盘的变换矩阵列表。

        返回:
        - M_bc: 基座到相机的变换矩阵。
        """
        assert len(M_bt_list) == len(M_cc_list)

        # 初始化累积矩阵
        A = np.zeros((4, 4))
        B = np.zeros((4, 4))

        for M_bt, M_cc in zip(M_bt_list, M_cc_list):
            # 计算工具到棋盘的变换矩阵
            M_tc = np.linalg.inv(M_bt) @ M_cc

            # 累积计算
            A += M_bt
            B += M_tc @ M_cc

        # 计算基座到相机的变换矩阵
        M_bc = np.linalg.inv(A) @ B

        return M_bc


    # 相机标定
    # @staticmethod
    # def calibrate_camera(images, pattern_size, square_size):
    #     obj_points = []  # 3D点在世界坐标系中的位置
    #     img_points = []  # 2D点在图像平面的位置

    #     # 准备棋盘格的3D点
    #     objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    #     objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    #     objp *= square_size

    #     for image in images:
    #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    #         if ret:
    #             img_points.append(corners)
    #             obj_points.append(objp)

    #     # 相机标定
    #     ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    #     return camera_matrix, dist_coeffs

    # 标定总流程
    def callbrate(self):
        """
        直接从文件夹中读取图片和必要矩阵信息, 计算变换矩阵, 进而得到总体外参矩阵R和t
        """
        # 获取时间戳
        with open(f'./chessboards/timestamps','r') as f:
            timestamps = f.read().split('\n')[:-1]

        # 用于储存每次计算得到的Mcc和Mbt
        Mccs, Mbts = [], []

        for timestamp in timestamps:
            # 计算该时间点下的Mcc和Mbt
            Mcc = self.calc_Mcc(timestamp)
            Mbt = self.calc_Mbt(timestamp)

            # 将其储存在列表中
            Mccs.append(Mcc)
            Mbts.append(Mbt)
        
        # 将其代入最终cv2的结果
        R, t = self.hand_eye_calibration(Mccs, Mbts)
        return R,t
        # Mt = self.eye_to_hand_calibration(Mbts, Mccs)
        # return Mt

    # 手眼标定
    @staticmethod
    def hand_eye_calibration(Mccs, Mbts):
        """
        由Mcc和Mbt得到标定外参矩阵
        """
        R_bts,t_bts, R_ccs, t_ccs = [],[],[],[]

        for Mcc, Mbt in zip(Mccs,Mbts):
            R_bts.append(Mbt[:3,:3])
            
            t_bts.append(Mbt[:3,3])
            R_ccs.append(Mcc[:3,:3])
            t_ccs.append(Mcc[:3,3])

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_bts, t_bts, R_ccs, t_ccs, method=cv2.CALIB_HAND_EYE_TSAI
        )

        return R_cam2gripper, t_cam2gripper

def main():
    robot_env = robotEnv.RobotEnv()
    robot_env.realtime_shoot()

    callibrator = Callibrator()
    Mt = callibrator.callbrate()
    print(Mt)

if __name__ == '__main__':
    main()
    