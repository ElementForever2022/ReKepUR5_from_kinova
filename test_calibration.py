import cv2
import numpy as np
import pyrealsense2 as rs
from camera import Camera
import os
import rotation_matrix

def convert_to_transformation_matrix(rvecs, tvecs):
    """
    rvecs: 相对于机器人坐标系的旋转角度,例如相对于z轴旋转-90度,rvecs=[0, 0, -90]
    tvecs: 相对于机器人坐标系的平移,例如在xyz分别平移1m,2m,3m,tvecs=[1, 2, 3]
    """
    rotate_matrix = rotation_matrix.rotation_matrix_z(rvecs[2])
    translation_vector = tvecs
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = np.linalg.inv(rotate_matrix)  # 填入旋转矩阵
    transformation_matrix[:3, 3] = translation_vector.ravel()  # 填入平移向量

    print("transformation_matrix", np.linalg.inv(transformation_matrix))
    return np.linalg.inv(transformation_matrix)

def get_matrix(color_image, aligned_depth_frame):
    # 获取图片
    # cam = Camera(1280, 720, 15)
    # color_image, depthm_image, _, aligned_depth_frame = cam.get_frame() 
    cv2.imwrite("/home/user/112/0.png", color_image)
    directory = "/home/user/111/"
    image_names = os.listdir(directory)
    image_paths = ["/home/user/112/0.png"]+[os.path.join(directory, name) for name in image_names]  # 标定图片路径
    # image_paths =[os.path.join(directory, name) for name in image_names]  # 标定图片路径

    # 配置参数
    pattern_size = (11, 8)  # 棋盘格内部角点数量，行数为9，列数为12
    square_size = 0.01  # 棋盘格每格的边长，单位为米（10mm -> 0.01m）
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 计算棋盘格中心点
    # 棋盘格在机器人坐标系中的固定位置（需要事先测量得到）
    # 矩阵的形式为 4x4 齐次变换矩阵
    # robot_to_board = np.array([
    #     [0, 1, 0, 0.7087157],  # X方向平移0.7087157米
    #     [-1, 0, 0, -0.0475],  # Y方向平移-0.0475米
    #     [0, 0, 1, 0.2623343],  # Z方向平移0.2623343米
    #     [0, 0, 0, 1]
    # ])
    # robot_to_board = np.array([
    #     [0, -1, 0, 0.415],  # X方向平移0.7087157米
    #     [1, 0, 0, -0.05],  # Y方向平移-0.0475米
    #     [0, 0, 1, 0.30],  # Z方向平移0.2623343米
    #     [0, 0, 0, 1]
    # ])
    # robot_to_board = np.array([
    #     [0, 1, 0, 0.415],  # X方向平移0.7087157米
    #     [-1, 0, 0, -0.05],  # Y方向平移-0.0475米
    #     [0, 0, 1, 0.30],  # Z方向平移0.2623343米
    #     [0, 0, 0, 1]
    # ])
    r2b_rvecs = np.array([0, 0, -90])
    r2b_tvecs = np.array([ 0.5637157, -0.0474726, 0.3023343])
    robot_to_board = convert_to_transformation_matrix(r2b_rvecs, r2b_tvecs)
    # 1. 生成棋盘格的 3D 物理坐标（棋盘格坐标系）
    # 这些点是棋盘格的角点在世界坐标系中的实际位置
    # 每个点的 Z 坐标为 0，因为棋盘格是平面的
    obj_points = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    # mgrid生成一个网格坐标点，按棋盘格的格子大小进行缩放
    obj_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size  # 88*3  8*11


    # 用于存储所有图片的 3D 和 2D 点
    obj_points_list = []  # 保存每张图片中棋盘格的3D物理坐标点
    img_points_list = []  # 保存每张图片中棋盘格的2D像素坐标点
    for image_path in image_paths:
        # 2. 检测所有标定图片中的棋盘格角点
        image = cv2.imread(image_path)
        # image = color_image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图，便于处理
        # 检测棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:  # 如果成功检测到角点
            obj_points_list.append(obj_points)  # 记录棋盘格的3D物理坐标点
            # corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria).squeeze(axis=1)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            if [corners2]:
                img_points_list.append(corners2)  # 记录角点的2D像素坐标点
            else:
                img_points_list.append(corners)

            # # 可视化检测结果（画出角点）
            # cv2.drawChessboardCorners(image, pattern_size, corners, ret)
            # cv2.imshow("Chessboard", image)  # 显示检测结果
            # cv2.waitKey(5000)  # 暂停以观察结果
    cv2.destroyAllWindows()  # 关闭所有窗口

    # 3. 相机标定，计算相机内参和外参
    # calibrateCamera函数会返回相机的内参矩阵、畸变系数以及每张图片的外参
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points_list, img_points_list, gray.shape[::-1], None, None
    )
    print(f"b2c平移向量:{tvecs}")
    print(f"b2c旋转向量:{rvecs}")
    # 打印相机内参矩阵（焦距和光心） 单位为像素
    print("相机内参矩阵:\n", camera_matrix)
    # 打印相机的畸变系数（畸变矫正用）
    print("畸变系数:\n", dist_coeffs)

    # 相机内参
    intrinsic = aligned_depth_frame.profile.as_video_stream_profile().get_intrinsics()
    camera_matrix = np.array([
        [intrinsic.fx, 0, intrinsic.ppx],
        [0, intrinsic.fy, intrinsic.ppy],
        [0, 0, 1]
    ])
    print("相机内参矩阵:\n", camera_matrix)


    # 4. 提取第一张图片的外参矩阵
    # 外参包括旋转向量和平移向量
    rotation_vector = rvecs[0]  # 第一张图片的旋转向量
    translation_vector = tvecs[0]  # 第一张图片的平移向量
    print("第一张图片的旋转向量", rotation_vector)
    print("第一张图片的平移向量", translation_vector)
    # print("translation_vector的类型", translation_vector.type)
    # 将旋转向量转换为旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # # 构造相机到棋盘格的外参矩阵（齐次变换矩阵形式）  
    # camera_to_board = np.eye(4)
    # camera_to_board[:3, :3] = rotation_matrix  # 填入旋转矩阵
    # camera_to_board[:3, 3] = translation_vector.ravel()  # 填入平移向量
    # print("相机到棋盘格的外参矩阵:\n", camera_to_board)

    # 构造棋盘格到相机的外参矩阵（齐次变换矩阵形式）  
    board_to_camera = np.eye(4)
    board_to_camera[:3, :3] = rotation_matrix  # 填入旋转矩阵
    board_to_camera[:3, 3] = translation_vector.ravel()  # 填入平移向量
    print("相机到棋盘格的外参矩阵:\n", board_to_camera)

    # 5. 计算机器人到相机坐标系的外参矩阵
    # 外参矩阵变换：机器人 -> 棋盘格 -> 相机坐标系
    # robot_to_camera = camera_to_board @ np.linalg.inv(robot_to_board)
    robot_to_camera = board_to_camera @ robot_to_board
    print("机器人到相机坐标系的外参矩阵:\n", robot_to_camera)

    return  camera_matrix, robot_to_camera

def main():
    cam = Camera(1280, 720, 15)
    color_image, depth_image, _, aligned_depth_frame = cam.get_frame() 
    camera_matrix, robot_to_camera = get_matrix(color_image, aligned_depth_frame)

    # 从像素坐标得到机器人坐标
    def pixel_to_robot(u, v, Z, intrinsic_matrix, extrinsic_matrix):
        # 像素坐标到相机坐标
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
        X_c = (u - cx) * Z / fx
        Y_c = (v - cy) * Z / fy
        print(f"深度：{Z}")
    
        camera_coords = np.array([X_c, Y_c, Z, 1]).reshape(4,1) # 齐次坐标
        # 相机到板坐标系
        # board_coords = camera_to_board @ camera_coords
        # 相机坐标到机器人坐标
        robot_coords = np.linalg.inv(extrinsic_matrix) @ camera_coords
        print("camera_coords", camera_coords[:3])
        print("robot_coords", robot_coords[:3])

        return robot_coords[:3]

    # 定义鼠标点击事件回调函数
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            # 获取鼠标点的深度
            # depth = depth_image[y, x] * aligned_depth_frame.get_units()
            depth = depth_image[y, x] /1000
            # 将鼠标点从像素点转化为机器人坐标
            robot_to_camera = np.array( [[ 0.03067451,  0.99949243, -0.00859988,  0.05214091],
                            [-0.93411425,  0.03172738,  0.35556143,  0.31722874],
                            [ 0.35565381, -0.0028734,   0.93461335, -0.06063908],
                            [ 0.        ,  0.        ,  0.        ,  1.        ]])
            robot_coords = pixel_to_robot(x, y, depth, camera_matrix, robot_to_camera)
            # 在图像上显示点击的位置
            cv2.circle(color_image, (x, y), 1, (0, 0, 255), -1)
            # 在图像上显示点击点的机器人坐标
            font = cv2.FONT_HERSHEY_SIMPLEX # 设置字体类型
            font_scale = 1 # 设置字体大小
            color = (0, 255, 0)  # 设置文本颜色（BGR格式） # 绿色
            thickness = 2 # 设置文本粗细
            cv2.putText(color_image, str(robot_coords[0])[1:6], (x, y), font, font_scale, color, thickness)
            cv2.imshow("Image", color_image)

    cv2.imshow("Image", color_image)
    cv2.setMouseCallback("Image", mouse_callback)

    # 等待按键事件直到按下 'Q' 键退出
    while True:
        key = cv2.waitKey(1)  # 等待 1 毫秒检查按键事件
        if key == ord('Q'):  # 如果按下 'Q' 键
            print("Exiting...")
            cv2.destroyAllWindows()  # 关闭所有窗口
            break  # 退出循环

if __name__ == "__main__":
    main()
    # rvecs = np.array([0, 0, -90])
    # tvecs = np.array([ 0.415, -0.05, 0.30])
    # convert_to_transformation_matrix(rvecs, tvecs)
