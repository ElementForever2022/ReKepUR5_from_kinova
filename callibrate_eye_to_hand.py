# 本代码的功能是，根据chessboards中的图片和tcp信息，进行眼在手外(eye to hand)的标定

# 导入必要库
import cv2
import numpy as np

import tqdm

import random
from scipy.spatial.transform import Rotation as R

from AXXB import AXXBsolver

# 预定义参数(可修改)
XX, YY = 7, 7 # 棋盘格尺寸为8*8, 所以XX和YY要定义为7*7
# L = 0.0164 # 棋盘格小格子的边长为0.0164, 单位:m
L = 0.0128
save_dir = './chessboards/'
filter_bound = 1
num_samples = 2000 # 抽样数量

# 相机内参
class RS_Intrinsics:
    def __init__(self):
        self.fx = 386.738  # focal length x:m
        self.fy = 386.738
        self.ppx = 319.5 # 光心坐标
        self.ppy = 239.5
intrinsics = RS_Intrinsics()
intrinsics_matrix = np.array([
    [intrinsics.fx, 0, intrinsics.ppx],
    [0, intrinsics.fy, intrinsics.ppy],
    [0,0,1]
])

# L = 12.8 # mm
# 打印预定义参数
print('#'*20)
print(f'size of chessboard(can be manually modified):\n\t{XX}x{YY}')
print(f'length of unit squares(can be manually modified):\n\t{L}m')
print(f'save directory:\n\t{save_dir}')
print(f'filter bound:\n\t±{filter_bound} sigma')
print(f'number of samples:\n\t{num_samples}')
print(f'camera intrinsics matrix:\n{intrinsics_matrix}')
print('#'*20,'\n')

def get_imgs_and_tcps(save_dir=f'{save_dir}'):
    """
    通过读取save_dir下的timestamps, 读取全部图片和机械臂位姿(tcp), 便于后续利用

    输入:
        save_dir, 为图片和机械臂位姿的保存路径
    输出: 
        imgs:List[img], tcps:List[tcp], 为图片和机械臂位姿的np文件的存储列表, 其中tcp格式为[x,y,z,rx,ry,rz]
    """

    # 获取拍照时间戳
    with open(save_dir+'timestamps','r') as f:
        timestamps = f.read().split('\n')[:-1]
    
    # 存储图片(imgs)和机械臂位姿(tcps)
    imgs = []
    tcps = []
    depths = []

    # 读取那些img和那些tcp，并保存
    for timestamp in timestamps:
        try:
            # 读取每个时间戳下对应的图片
            img = cv2.imread(f'{save_dir}rgb_{timestamp}.png')
            # 读取每个时间戳下对应的tcp
            tcp = np.load(f'{save_dir}tcp_{timestamp}.npy')
        
            depth = np.load(f'{save_dir}depth_{timestamp}.npy')
        except FileNotFoundError:
            continue

        # 保存图片和tcp
        imgs.append(img)
        tcps.append(tcp)
        depths.append(depth)
    
    # 将图片和机械臂位姿返回
    # return imgs,tcps,depths
    return imgs, tcps

def calc_Mcc(imgs):
    """
    通过图片列表和预定义参数, 计算每个图片对应的Mcc

    输入:
        imgs:List[img]
    输出:
        rvecs:Tuple[np.array(rvec矩阵)], tvecs:Tuple[np.array(tvec矩阵形式列向量)]
    """

    # 准备计算图片中角点的坐标
    img_points = [] # 存储角点的像素坐标
    obj_points = [] # 存储角点的棋盘坐标(单位为m)

    # 计算棋盘格角点在cal坐标系下的坐标(每个图片中，角点的坐标都是一样的)
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2) # 这段不用理解，知道下面的结论就行
    objp = L * objp # x和y坐标根据格点给出，z设为0

    # 开始遍历所有图片
    for img in imgs:
        # 图片转换为灰色
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用OpenCV的算法得到棋盘格角点的位置
        ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
        # ret为是否找到（可用if判断），corners为棋盘格的像素坐标
        """
        corners格式为[
            [[x1,y1]],
            [[x2,y2]],
            ...,
            [[xn,yn]]
        ]
        """
        
        # 如果找到了角点，那么添加信息
        if ret:
            # 添加空间点坐标
            obj_points.append(objp)
            # 亚像素角点查找准则
            criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
            # 根据图片寻找格点的亚像素坐标(浮点数)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            # 找到格点的像素坐标后添加进img_points
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)
    
    # 寻找到每个图片的点之后，计算每个图片的Mcc
    size = imgs[0].shape[:2][::-1] # size是图片的大小(W,H)
    # 标定，得到图案在相机坐标系下的位姿（Mcc）
    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None,criteria= 
    #                                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    ret, mtx, dist, rvecs_tuple, tvecs_tuple = cv2.calibrateCamera(obj_points, img_points, size, intrinsics_matrix, None,
                                                       criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    # 这里给这个矩阵求个逆
    rvecs_inv_list = []
    tvecs_inv_list = []
    for rvecs,tvecs in zip(rvecs_tuple,tvecs_tuple):
        rvecs = R.from_euler('xyz',rvecs[:,0]).as_matrix()
        rvecs = np.linalg.inv(rvecs)
        tvecs = -rvecs@tvecs
        # rvecs = R.from_matrix(rvecs).as_euler('xyz').reshape(-1,1) # 转换为列向量

        rvecs_inv_list.append(rvecs)
        tvecs_inv_list.append(tvecs)

    # 其中的rvecs就是每个Mcc的旋转矩阵(euler)，tvecs是每个Mcc的平移向量
    # return rvecs_tuple, tvecs_tuple, mtx, dist
    return rvecs_inv_list, tvecs_inv_list

def calc_Mtb(tcps):
    """
    将机械臂位姿tcps转换为变换矩阵格式Mtb(不知道为什么cv2的输入需要的是Mtb而不是Mbt)

    输入:
        tcps:List[tcp] tcp为机械臂位姿的np.array行向量格式,[x,y,z,rx,ry,rz], x,y,z in metre
    输出:
        R_tools:List[np.array 旋转矩阵], t_tools:List[np.array 平移矩阵]
    """
    # 将[rx, ry, rz]格式转换为旋转矩阵格式
    def euler_angles_to_rotation_matrix(rx, ry, rz):
        """
        将欧拉角旋转格式(rx,ry,rz)转换为旋转矩阵格式

        输入:
            rx,ry,rz 欧拉角旋转
        输出:
            MR 旋转矩阵
        """
        # 将其转换为scipy旋转标准形式
        scipy_rotation = R.from_euler('xyz',np.array([rx,ry,rz]))
        # 将标准形式转换为旋转矩阵形式
        MR = scipy_rotation.as_matrix()
        return MR
        # 计算旋转矩阵
        # Rx = np.array([[1, 0, 0],
        #             [0, np.cos(rx), -np.sin(rx)],
        #             [0, np.sin(rx), np.cos(rx)]])

        # Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
        #             [0, 1, 0],
        #             [-np.sin(ry), 0, np.cos(ry)]])

        # Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
        #             [np.sin(rz), np.cos(rz), 0],
        #             [0, 0, 1]])

        # R = Rz@Ry@Rx  # 先绕 x轴旋转 再绕y轴旋转  最后绕z轴旋转
        # return R
    # 将机械臂位姿(tcp)转换为旋转矩阵
    def pose_to_homogeneous_matrix(tcp):
        """
        将机械臂位姿(tcp)转换为旋转矩阵

        输入:
            tcps:List[tcp] tcp为机械臂位姿的np.array行向量格式,[x,y,z,rx,ry,rz], x,y,z in metre
        输出:
            H:np.array H为某个[x,y,z,rx,ry,rz]对应的线性变换矩阵
        """
        # 将变量解包
        x, y, z, rx, ry, rz = tcp
        # 计算旋转矩阵
        MR = euler_angles_to_rotation_matrix(rx, ry, rz)
        # 计算平移矩阵(列向量)
        t = np.array([x, y, z]).reshape(3, 1)

        # 将旋转矩阵和平移矩阵拼接为线性变换矩阵
        H = np.eye(4)
        H[:3, :3] = MR
        H[:3, 3] = t[:, 0]

        # 返回线性变换矩阵
        return H
    
    def inverse_transformation_matrix(T):
        MR = T[:3, :3]
        t = T[:3, 3]

        # 计算旋转矩阵的逆矩阵
        R_inv = MR.T

        # 计算平移向量的逆矩阵
        t_inv = -np.dot(R_inv, t)

        # 构建逆变换矩阵
        T_inv = np.identity(4)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv

        return T_inv
    
    # 用于存储每个"base"坐标系相对于"tool"坐标系的旋转和平移矩阵
    R_tools = []
    t_tools = []

    # 开始遍历每个机械臂的位姿
    for tcp in tcps:
        # 首先计算工具tool相对基座base的变换矩阵
        Mbt = pose_to_homogeneous_matrix(tcp)
        # 对矩阵求逆即为基座base相对tool的变换矩阵
        # Mtb = inverse_transformation_matrix(Mbt)
        # Mtb = np.linalg.inv(Mbt)
        Mtb = Mbt.copy()

        # 对矩阵解包即可得到旋转矩阵R_tool和平移矩阵t_tool
        R_tool = Mtb[:3,:3] # 旋转矩阵
        t_tool = Mtb[:3,3] # 平移矩阵

        # 将旋转矩阵和平移矩阵保存在列表中
        R_tools.append(R_tool)
        t_tools.append(t_tool)
    
    # 返回旋转矩阵和平移矩阵的列表
    return R_tools, t_tools


def skew(A):
    """Compute the skew symmetric matrix."""
    assert A.shape == (3, 1) or A.shape == (3,)
    B = np.zeros((3, 3), dtype=np.float64)
    B[0, 1] = -A[2]
    B[0, 2] = A[1]
    B[1, 0] = A[2]
    B[1, 2] = -A[0]
    B[2, 0] = -A[1]
    B[2, 1] = A[0]
    return B
def tsai_hand_eye(Hgij, Hcij):
    assert len(Hgij) == len(Hcij)
    nStatus = len(Hgij)

    A = []
    b = []

    for i in range(nStatus):
        Rgij = Hgij[i][:3, :3]
        Rcij = Hcij[i][:3, :3]

        rgij, _ = cv2.Rodrigues(Rgij)
        rcij, _ = cv2.Rodrigues(Rcij)

        theta_gij = np.linalg.norm(rgij)
        theta_cij = np.linalg.norm(rcij)

        rngij = rgij / theta_gij
        rncij = rcij / theta_cij

        Pgij = 2 * np.sin(theta_gij / 2) * rngij
        Pcij = 2 * np.sin(theta_cij / 2) * rncij

        tempA = skew(Pgij + Pcij)
        tempb = Pcij - Pgij

        A.append(tempA)
        b.append(tempb)

    A = np.vstack(A)
    b = np.vstack(b)

    # Compute rotation
    pinA = np.linalg.pinv(A)
    Pcg_prime = pinA @ b
    Pcg = 2 * Pcg_prime / np.sqrt(1 + np.linalg.norm(Pcg_prime) ** 2)
    PcgTrs = Pcg.T
    eyeM = np.eye(3)
    Rcg = (1 - np.linalg.norm(Pcg) ** 2 / 2) * eyeM + 0.5 * (Pcg @ PcgTrs + np.sqrt(4 - np.linalg.norm(Pcg) ** 2) * skew(Pcg))

    AA = []
    bb = []

    for i in range(nStatus):
        Rgij = Hgij[i][:3, :3]
        Tgij = Hgij[i][:3, 3].reshape(3, 1)
        Tcij = Hcij[i][:3, 3].reshape(3, 1)

        tempAA = Rgij - eyeM
        tempbb = Rcg @ Tcij - Tgij

        AA.append(tempAA)
        bb.append(tempbb)

    AA = np.vstack(AA)
    bb = np.vstack(bb)

    pinAA = np.linalg.pinv(AA)
    Tcg = pinAA @ bb

    Hcg = np.eye(4)
    Hcg[:3, :3] = Rcg
    Hcg[:3, 3] = Tcg.flatten()

    return Hcg

def solve_ax_xb(A_list, B_list):
    assert len(A_list) == len(B_list), "A and B lists must have the same length."

    # Solve for rotation using quaternions
    M = np.zeros((4, 4))
    for A, B in zip(A_list, B_list):
        RA = A[:3, :3]
        RB = B[:3, :3]
        qA = R.from_matrix(RA).as_quat()
        qB = R.from_matrix(RB).as_quat()
        qA = np.outer(qA, qA)
        qB = np.outer(qB, qB)
        M += qA + qB

    # Eigen decomposition
    _, eigvecs = np.linalg.eigh(M)
    qX = eigvecs[:, -1]  # The eigenvector corresponding to the largest eigenvalue
    RX = R.from_quat(qX).as_matrix()

    # Solve for translation
    C = []
    d = []
    for A, B in zip(A_list, B_list):
        RA = A[:3, :3]
        tA = A[:3, 3]
        tB = B[:3, 3]
        C.append(RA - np.eye(3))
        d.append(tB - RX @ tA)

    C = np.vstack(C)
    d = np.hstack(d)
    tX = np.linalg.lstsq(C, d, rcond=None)[0]

    # Construct the transformation matrix X
    X = np.eye(4)
    X[:3, :3] = RX
    X[:3, 3] = tX

    return X

def MM2AB(Mccs,Mbts):
    """
    将原始的两组变换矩阵转换为AB项
    """
    assert len(Mccs)==len(Mbts), '矩阵数量不相等'
    
    mat_As = []
    mat_Bs = []

    # 任意随机抽取20对矩阵
    num_mats = len(Mccs)
    multations = []
    for i in range(num_mats-1):
        for j in range(i+1,num_mats):
            multations.append([i,j])
    sampled_indices = random.sample(multations,30)
    for sampled_index in sampled_indices:
        index_1, index_2 = sampled_index
        mat_A = np.linalg.inv(Mbts[index_2])@Mbts[index_1]
        mat_B = np.linalg.inv(Mccs[index_2])@Mccs[index_1]

        # mat_A = np.linalg.inv(mat_A)
        # mat_B = np.linalg.inv(mat_B)

        mat_As.append(mat_A)
        mat_Bs.append(mat_B)
    
    return mat_As, mat_Bs

def mat2Rt(mat):
    R = mat[:3,:3]
    t = np.array(mat[:3,3:])

    return R,t

def average_transforms(transforms):
    """
    对齐次变换矩阵数组进行平均：
      - 旋转部分使用四元数平均
      - 平移部分直接取均值
    """
    quats = []
    trans = []
    for T in transforms:
        # 提取旋转矩阵和平移向量
        R_mat = T[:3, :3]
        t_vec = T[:3, 3]
        # 将旋转矩阵转换为四元数，注意：scipy 的 as_quat 返回的是 [x, y, z, w]
        quat = R.from_matrix(R_mat).as_quat()
        quats.append(quat)
        trans.append(t_vec)
    quats = np.array(quats)
    trans = np.array(trans)
    
    # 采用特征分解方法求四元数平均
    A = np.zeros((4, 4))
    for q in quats:
        A += np.outer(q, q)
    A = A / len(quats)
    eigvals, eigvecs = np.linalg.eig(A)
    avg_quat = eigvecs[:, np.argmax(eigvals)]
    # 保证四元数的标量部分为正
    if avg_quat[3] < 0:
        avg_quat = -avg_quat
    avg_R = R.from_quat(avg_quat).as_matrix()
    avg_t = np.mean(trans, axis=0)
    
    T_avg = np.eye(4)
    T_avg[:3, :3] = avg_R
    T_avg[:3, 3] = avg_t
    return T_avg

def compute_cam2base(cam2cal_list, base2tool_list):
    """
    给定多个测量数据，每组数据包含 cam2cal 和 base2tool 两个 4x4 齐次变换矩阵，
    使用左乘行变换的约定，计算 eye-to-hand 标定下的 cam2base。
    
    理论上有：
      cam2base * cam2cal = base2tool
    因此，每组数据候选解为：
      X_i = base2tool_i * inv(cam2cal_i)
    
    最后对所有候选解进行平均，得到稳健的 cam2base 估计。
    """
    X_candidates = []
    for H_cam2cal, H_base2tool in zip(cam2cal_list, base2tool_list):
        H_cam2cal_inv = np.linalg.inv(H_cam2cal)
        X = H_base2tool @ H_cam2cal_inv  # 左乘约定：p * X
        X_candidates.append(X)
    
    # 对候选的 cam2base 进行平均
    cam2base = average_transforms(X_candidates)
    return cam2base

def gpt_calibrate_eye_to_hand(cam2cal_list, base2tool_list):
    """
    利用 OpenCV 的 calibrateHandEye 实现 eye-to-hand 标定，
    输入：
      - cam2cal_list: 多组相机到标定板的 4x4 齐次变换矩阵（列表或数组）
      - base2tool_list: 多组机器人基座到工具（末端执行器）的 4x4 齐次变换矩阵
    输出：
      - cam2base: 估计的相机到机器人基坐标系的 4x4 齐次变换矩阵
    """
    n = len(cam2cal_list)
    # 为了使用 cv2.calibrateHandEye，先转换为相对运动：
    # 这里我们先将绝对变换取逆，得到：
    #   gripper2base = inv(base2tool)  （注意：这里工具通常称作 gripper）
    #   target2cam  = inv(cam2cal)      （标定板到相机的变换）
    R_gripper2base_abs = []
    t_gripper2base_abs = []
    R_target2cam_abs = []
    t_target2cam_abs = []
    for i in range(n):
        T_base2tool = base2tool_list[i]
        T_cam2cal = cam2cal_list[i]
        T_gripper2base = np.linalg.inv(T_base2tool)
        T_target2cam  = np.linalg.inv(T_cam2cal)
        R_gripper2base_abs.append(T_gripper2base[:3, :3])
        t_gripper2base_abs.append(T_gripper2base[:3, 3])
        R_target2cam_abs.append(T_target2cam[:3, :3])
        t_target2cam_abs.append(T_target2cam[:3, 3])
    
    # 接下来计算相邻两帧之间的相对运动：
    R_gripper2base_rel = []
    t_gripper2base_rel = []
    R_target2cam_rel = []
    t_target2cam_rel = []
    for i in range(n - 1):
        # 对机器人部分：
        T1 = np.eye(4)
        T1[:3, :3] = R_gripper2base_abs[i]
        T1[:3, 3]  = t_gripper2base_abs[i]
        T2 = np.eye(4)
        T2[:3, :3] = R_gripper2base_abs[i+1]
        T2[:3, 3]  = t_gripper2base_abs[i+1]
        # 相对变换：从第 i 帧到 i+1 帧
        T_rel = np.linalg.inv(T1) @ T2
        R_gripper2base_rel.append(T_rel[:3, :3])
        t_gripper2base_rel.append(T_rel[:3, 3])
        
        # 对标定板部分：
        T1 = np.eye(4)
        T1[:3, :3] = R_target2cam_abs[i]
        T1[:3, 3]  = t_target2cam_abs[i]
        T2 = np.eye(4)
        T2[:3, :3] = R_target2cam_abs[i+1]
        T2[:3, 3]  = t_target2cam_abs[i+1]
        T_rel = np.linalg.inv(T1) @ T2
        R_target2cam_rel.append(T_rel[:3, :3])
        t_target2cam_rel.append(T_rel[:3, 3])
    
    # 调用 OpenCV 的 calibrateHandEye 求解 X，使得：
    #     A_i * X = X * B_i
    # 对于 eye-to-hand，此处 X 就等价于 cam2base
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base_rel, t_gripper2base_rel,
        R_target2cam_rel, t_target2cam_rel,
        method=cv2.CALIB_HAND_EYE_TSAI)
    
    # 组合旋转和平移，构造 4x4 的齐次变换矩阵
    cam2base = np.eye(4)
    cam2base[:3, :3] = R_cam2base
    cam2base[:3, 3]  = t_cam2base.flatten()
    return cam2base

def imgs_tcps_to_Rt(imgs,tcps):
    # 第二步: 根据图片计算Mcc
    rvecs, tvecs = calc_Mcc(imgs)
    # print('rvecs:', rvecs)
    # print('tvecs:', tvecs)

    # 第三步: 根据那些机械臂位姿(tcps)计算Mtb(不知道为什么cv2要输入的是Mtb而不是Mbt)
    R_tools, t_tools = calc_Mtb(tcps)
    # print('R_tools',R_tools)
    # print('t_tools',t_tools)

    M_cam2cal = [Rt2Mat(rvec,tvec) for rvec,tvec in zip(rvecs,tvecs)]
    M_base2tool = [Rt2Mat(R_tool, t_tool) for R_tool,t_tool in zip(R_tools, t_tools)]

    As,Bs = MM2AB(M_cam2cal, M_base2tool)
    # X = solve_ax_xb(As,Bs) # gpt
    # solver = AXXBsolver(As,Bs) # 网上下的
    # X = solver.solve()
    # X = tsai_hand_eye(As,Bs) # gpt
    X = compute_cam2base(M_cam2cal,M_base2tool)
    # X = gpt_calibrate_eye_to_hand(M_cam2cal,M_base2tool)
    MR,t = mat2Rt(X)


    # 第四步: 根据上面得到的4组矩阵得到标定矩阵R和t
    # MR, t = cv2.calibrateHandEye(R_tools, t_tools, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)
    # MR, t = cv2.calibrateHandEye(rvecs, tvecs,R_tools, t_tools,  cv2.CALIB_HAND_EYE_TSAI)
    # 打印标定结果
    # print('#'*20)
    # print('callibration result:')
    # print('Rotation Matrix:')
    # print(R)
    # print('Shift Matrix:')
    # print(t)
    # print('Centre distance:')
    # print(np.sqrt(t.T@t))
    # print('#'*20)
    # print('MR',MR)
    # print('t',t,np.sqrt(t.T@t))
    return MR,t

from scipy.spatial.transform import Rotation as Rot
def Rt_to_xyzrxryrz(MR,t):

    # 提取平移矩阵
    x,y,z = t[0,0],t[1,0],t[2,0]

    # 创建旋转对象
    rotation = Rot.from_matrix(MR)

    # 将旋转矩阵转换为欧拉角
    # 'xyz'表示欧拉角的旋转顺序
    euler_angles = rotation.as_euler('xyz', degrees=False)
    rx,ry,rz = euler_angles

    return x,y,z,rx,ry,rz

def xyzrxryrz_to_Rt(x,y,z,rx,ry,rz):
    """
    将三维偏移量和欧拉旋转角转换为变换矩阵
    """
    def euler_angles_to_rotation_matrix(rx, ry, rz):
        """
        将欧拉角旋转格式(rx,ry,rz)转换为旋转矩阵格式

        输入:
            rx,ry,rz 欧拉角旋转
        输出:
            MR 旋转矩阵
        """
        # 将其转换为scipy旋转标准形式
        scipy_rotation = R.from_euler('xyz',np.array([rx,ry,rz]))
        # 将标准形式转换为旋转矩阵形式
        MR = scipy_rotation.as_matrix()
        return MR
        # 计算旋转矩阵
        # Rx = np.array([[1, 0, 0],
        #             [0, np.cos(rx), -np.sin(rx)],
        #             [0, np.sin(rx), np.cos(rx)]])

        # Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
        #             [0, 1, 0],
        #             [-np.sin(ry), 0, np.cos(ry)]])

        # Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
        #             [np.sin(rz), np.cos(rz), 0],
        #             [0, 0, 1]])

        # R = Rz@Ry@Rx  # 先绕 x轴旋转 再绕y轴旋转  最后绕z轴旋转
        # return R

    t = np.array([[x],[y],[z]],dtype=np.float32)
    MR = euler_angles_to_rotation_matrix(rx,ry,rz)

    return MR,t


def append_normt_xyzrxryrz(num_workers,num_samples, num_indices, imgs,tcps, progress_counter,lock_counter, global_norm_ts, global_xyzrxryrzs):
    
    for epoch in range(num_samples//num_workers):
        indices = range(num_indices)
        chosen_indices = random.sample(indices,15) # 从下标中抽35个元素, 不重复

        # 按照下标得到对应的图片和tcp
        chosen_imgs = [imgs[img_index] for img_index in chosen_indices]
        chosen_tcps = [tcps[tcp_index] for tcp_index in chosen_indices]

        # 代入计算得到转换矩阵
        MR,t = imgs_tcps_to_Rt(chosen_imgs, chosen_tcps)

        # 计算偏移量的模并将其保存
        norm_t = np.sqrt(t.T@t)[0,0]

        # 计算各个分量的值
        x,y,z,rx,ry,rz = Rt_to_xyzrxryrz(MR,t)


        
        global_norm_ts.append(norm_t)
        global_xyzrxryrzs.append([x,y,z,rx,ry,rz])

        # 更新进度
        lock_counter.acquire()
        progress_counter.value += 1
        lock_counter.release()

def Rt2Mat(R,t):
    """
    将旋转矩阵(3*3)和平移矩阵(列向量或行向量)转换为齐次变换矩阵(4*4)

    R: np.array 3*3
    t: np.array 1*3或3*1或3

    return: Mat np.array 4*4
    """
    Mat = np.eye(4)
    Mat[:3,:3] = R
    if t.shape not in [(1,3),(3,1),(3,)]:
        raise Exception(f'Not correct shape of t! 1*3 or 3*1 is wanted! Given {t.shape}')
    if t.shape==(1,3) or t.shape==(3,):
        Mat[:3,3] = t
    else:
        # t.shape==(3,1)
        Mat[:3,3] = t.T
    
    return Mat

def systemTransform(x,y,z,Mt):
    """
    将原始坐标系下的坐标(x,y,z), 通过变换矩阵Mt变换到新的坐标系

    x,y,z: 原始坐标系下的坐标
    Mt: 从原始坐标系到新坐标系的4*4变换矩阵

    return new_x, new_y, new_z: 新坐标系下的坐标
    """

    P_orig = np.array([[x],[y],[z],[1]]) # 原始坐标系下的坐标的列向量形式
    M_trans = Mt.copy() # 原始坐标系到新坐标系的变换矩阵

    P_new = M_trans@P_orig # 新坐标系下的坐标的列向量形式

    new_x, new_y, new_z = P_new[0,0], P_new[1,0], P_new[2,0] # 将新坐标系下的坐标的列向量解包

    return new_x, new_y, new_z

def main():
    # 第一步: 读取图片和机械臂位姿. 准备计算
    imgs, tcps = get_imgs_and_tcps()

    # 将图片和tcp随机抽取进行计算
    num_indices = len(imgs)
    filtered = np.array([True]*num_samples,dtype=np.bool_)

    # 多进程计算
    import multiprocessing as mp
    num_workers = 20
    # 使用Manager创建一个共享的进度计数器
    manager = mp.Manager()
    progress_counter = manager.Value('i', 0)

    norm_ts = manager.list() # 抽取到的图片组得到的偏移量之模
    xyzrxryrzs = manager.list() # 抽取到的图片组得到的变换矩阵的各个变换参量的列表
    # 创建一个multiprocessing的lock
    lock_counter = mp.Lock()
    # 创建计算工人
    workers = [
        mp.Process(target=append_normt_xyzrxryrz, args=
                   (num_workers,num_samples, num_indices, imgs,tcps,progress_counter,lock_counter, norm_ts, xyzrxryrzs)
        ) 
        for _ in range(num_workers)
    ]
    
    # 启动所有进程
    for worker in workers:
        worker.start()

    # 使用tqdm显示进度条
    with tqdm.tqdm(total=num_samples) as pbar:
        while progress_counter.value < num_samples:
            pbar.n = progress_counter.value
            pbar.refresh()

    # 等待所有进程完成
    for worker in workers:
        worker.join()
    print('calculation finished')

    # 收集并整理数据
    norm_ts = list(norm_ts) # 将其从共享列表转换为普通列表
    xyzrxryrzs = list(xyzrxryrzs)


    # 将其作图
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.subplot(7,1,1)
    norm_ts = np.array(norm_ts)
    sns.histplot(norm_ts, bins=10, kde=True, color='blue')

    mean = np.mean(norm_ts)
    std_dev = np.std(norm_ts)

    # 计算sigma位置
    sigma_pos = mean + filter_bound * std_dev
    sigma_neg = mean - filter_bound * std_dev

    # 计算各个变量是否在sigma之内
    in_sigma = [sigma_neg<=norm_t<=sigma_pos for norm_t in norm_ts]
    in_sigma = np.array(in_sigma,dtype=np.bool_)
    # 更新过滤器
    filtered = np.logical_and(filtered, in_sigma)

    # 在2-sigma位置绘制红色虚线
    plt.axvline(sigma_pos, color='red', linestyle='--', linewidth=2, label=f'+{filter_bound:.1f}σ')
    plt.axvline(sigma_neg, color='red', linestyle='--', linewidth=2, label=f'-{filter_bound:.1f}σ')

    # 设置图形标题和标签
    plt.title('Distribution of Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # 对其他分量也计算分布
    xyzrxryrzs = np.array(xyzrxryrzs)
    for i in range(2,8):
        attrs = xyzrxryrzs[:,i-2]

        plt.subplot(7,1,i)
        sns.histplot(attrs, bins=10, kde=True, color='blue')

        mean = np.mean(attrs)
        std_dev = np.std(attrs)

        # 计算sigma位置
        sigma_pos = mean + filter_bound * std_dev
        sigma_neg = mean - filter_bound * std_dev

        # 计算各个变量是否在sigma之内
        in_sigma = [sigma_neg<=attr<=sigma_pos for attr in attrs]
        in_sigma = np.array(in_sigma,dtype=np.bool_)
        # 更新过滤器
        filtered = np.logical_and(filtered, in_sigma)


        # 在2-sigma位置绘制红色虚线
        plt.axvline(sigma_pos, color='red', linestyle='--', linewidth=2, label=f'+{filter_bound:.2f}σ')
        plt.axvline(sigma_neg, color='red', linestyle='--', linewidth=2, label=f'-{filter_bound:.2f}σ')

        # 设置图形标题和标签
        plt.xlabel('Value')
        plt.ylabel('Frequency')

    print('num filtered:',filtered.astype(np.int32).sum())

    # 显示图形
    plt.show()

    # 将过滤出的样本再次计算平均值, 最终的变换矩阵取这些平均值
    x_average = np.mean(xyzrxryrzs[:,0][filtered])
    y_average = np.mean(xyzrxryrzs[:,1][filtered])
    z_average = np.mean(xyzrxryrzs[:,2][filtered])
    rx_average = np.mean(xyzrxryrzs[:,3][filtered])
    ry_average = np.mean(xyzrxryrzs[:,4][filtered])
    rz_average = np.mean(xyzrxryrzs[:,5][filtered])
    R,t = xyzrxryrz_to_Rt(x_average,y_average,z_average,rx_average,ry_average,rz_average)

    # 计算其偏移量之模
    norm_t_final = np.sqrt(x_average**2+y_average**2+z_average**2)
    print(f'norm of t:\n{norm_t_final:.4f}m')
    # 将旋转矩阵和偏移矩阵拼接为变换矩阵
    callibration_matrix = np.vstack([np.hstack([R,t]),np.array([0,0,0,1])])
    print('callibration matrix:')
    print(f'{callibration_matrix}')
    # 将变换矩阵改成适合json的格式
    print('json form:')
    for row_index in range(4):
        row = callibration_matrix[row_index,:]
        print(list(row),end='')
        if row_index<3:
            print(',')
        else:
            print()
    # 将变换矩阵保存起来
    callibration_matrix_save_dir = f'{save_dir}callibration_matrix.npy'
    np.save(callibration_matrix_save_dir,callibration_matrix)
    print(f'callibration matrix saved to: {callibration_matrix_save_dir}')


def main2():
    imgs,tcps,depths = get_imgs_and_tcps()
    img = imgs[0]
    rvecs,tvecs,mtx,dist = calc_Mcc([img])
    rvec_euler = rvecs[0]
    tvec = tvecs[0]
    print(rvec_euler,'\n', tvec)
    # 基于callbrateCamera的结果计算变换矩阵
    Rcc,tcc = xyzrxryrz_to_Rt(tvec[0,0], tvec[1,0], tvec[2,0], rvec_euler[0,0], rvec_euler[1,0], rvec_euler[2,0])
    print(Rcc,tcc)
    Mcc = Rt2Mat(Rcc,tcc)
    print(Mcc)

    # newP_cam3D = systemTransform(0,0,0,np.linalg.inv(Mcc))
    newP_cam3D = systemTransform(0,0,0,Mcc)
    print(newP_cam3D, np.sqrt(np.sum(np.array(newP_cam3D)**2)))
    newPs_cam3D = np.array([newP_cam3D])
    # newP_cam2D,_ = cv2.projectPoints(newPs_cam3D, rvec_euler, tvec, mtx, dist)
    newP_cam2D,_ = cv2.projectPoints((0,0,0), rvec_euler, tvec, mtx, dist)
    print(newP_cam2D)
    newP_cam2D,_ = cv2.projectPoints((2*1.28/100,1*1.28/100,0), rvec_euler, tvec, mtx, dist)
    print(newP_cam2D)
    print(depths[0][263,62])
    print(tcps[0])
    # ts = Mcc[0]
    # t = ts[0]
    # print(t, np.sqrt(t.T@t))


    cv2.imshow('img',img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()


    # main2()