# 本代码的功能是，根据chessboards中的图片和tcp信息，进行眼在手外(eye to hand)的标定

# 导入必要库
import cv2
import numpy as np

import tqdm

import random

# 预定义参数(可修改)
XX, YY = 7, 7 # 棋盘格尺寸为8*8, 所以XX和YY要定义为7*7
# L = 0.0164 # 棋盘格小格子的边长为0.0164, 单位:m
L = 0.0128
save_dir = './chessboards/'
filter_bound = 1
num_samples = 2000 # 抽样数量
# L = 12.8 # mm
# 打印预定义参数
print('#'*20)
print(f'size of chessboard(can be manually modified):\n\t{XX}x{YY}')
print(f'length of unit squares(can be manually modified):\n\t{L}m')
print(f'save directory:\n\t{save_dir}')
print(f'filter bound:\n\t±{filter_bound} sigma')
print(f'number of samples:\n\t{num_samples}')
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

    # 读取那些img和那些tcp，并保存
    for timestamp in timestamps:
        try:
            # 读取每个时间戳下对应的图片
            img = cv2.imread(f'{save_dir}rgb_{timestamp}.png')
            # 读取每个时间戳下对应的tcp
            tcp = np.load(f'{save_dir}tcp_{timestamp}.npy')
        except FileNotFoundError:
            continue

        # 保存图片和tcp
        imgs.append(img)
        tcps.append(tcp)
    
    # 将图片和机械臂位姿返回
    return imgs,tcps

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
    size = imgs[0].shape[:2][::-1] # size是图片的大小(H,W)
    # 标定，得到图案在相机坐标系下的位姿（Mcc）
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    # 其中的rvecs就是每个Mcc的旋转矩阵，tvecs是每个Mcc的平移向量
    return rvecs, tvecs

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
            R 旋转矩阵
        """
        # 计算旋转矩阵
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])

        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])

        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])

        R = Rz@Ry@Rx  # 先绕 x轴旋转 再绕y轴旋转  最后绕z轴旋转
        return R
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
        R = euler_angles_to_rotation_matrix(rx, ry, rz)
        # 计算平移矩阵(列向量)
        t = np.array([x, y, z]).reshape(3, 1)

        # 将旋转矩阵和平移矩阵拼接为线性变换矩阵
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = t[:, 0]

        # 返回线性变换矩阵
        return H
    
    def inverse_transformation_matrix(T):
        R = T[:3, :3]
        t = T[:3, 3]

        # 计算旋转矩阵的逆矩阵
        R_inv = R.T

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
        Mtb = inverse_transformation_matrix(Mbt)

        # 对矩阵解包即可得到旋转矩阵R_tool和平移矩阵t_tool
        R_tool = Mtb[:3,:3] # 旋转矩阵
        t_tool = Mtb[:3,3] # 平移矩阵

        # 将旋转矩阵和平移矩阵保存在列表中
        R_tools.append(R_tool)
        t_tools.append(t_tool)
    
    # 返回旋转矩阵和平移矩阵的列表
    return R_tools, t_tools

def imgs_tcps_to_Rt(imgs,tcps):
    # 第二步: 根据图片计算Mcc
    rvecs, tvecs = calc_Mcc(imgs)
    # print('rvecs:', rvecs)
    # print('tvecs:', tvecs)

    # 第三步: 根据那些机械臂位姿(tcps)计算Mtb(不知道为什么cv2要输入的是Mtb而不是Mbt)
    R_tools, t_tools = calc_Mtb(tcps)
    # print('R_tools',R_tools)
    # print('t_tools',t_tools)

    # 第四步: 根据上面得到的4组矩阵得到标定矩阵R和t
    R, t = cv2.calibrateHandEye(R_tools, t_tools, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)

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

    return R,t

from scipy.spatial.transform import Rotation as Rot
def Rt_to_xyzrxryrz(R,t):

    # 提取平移矩阵
    x,y,z = t[0,0],t[1,0],t[2,0]

    # 创建旋转对象
    rotation = Rot.from_matrix(R)

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
            R 旋转矩阵
        """
        # 计算旋转矩阵
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])

        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])

        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])

        R = Rz@Ry@Rx  # 先绕 x轴旋转 再绕y轴旋转  最后绕z轴旋转
        return R

    t = np.array([[x],[y],[z]],dtype=np.float32)
    R = euler_angles_to_rotation_matrix(rx,ry,rz)

    return R,t


def append_normt_xyzrxryrz(num_workers,num_samples, num_indices, imgs,tcps, progress_counter,lock_counter, global_norm_ts, global_xyzrxryrzs):
    
    for epoch in range(num_samples//num_workers):
        indices = range(num_indices)
        chosen_indices = random.sample(indices,35) # 从下标中抽35个元素, 不重复

        # 按照下标得到对应的图片和tcp
        chosen_imgs = [imgs[img_index] for img_index in chosen_indices]
        chosen_tcps = [tcps[tcp_index] for tcp_index in chosen_indices]

        # 代入计算得到转换矩阵
        R,t = imgs_tcps_to_Rt(chosen_imgs, chosen_tcps)

        # 计算偏移量的模并将其保存
        norm_t = np.sqrt(t.T@t)[0,0]

        # 计算各个分量的值
        x,y,z,rx,ry,rz = Rt_to_xyzrxryrz(R,t)


        
        global_norm_ts.append(norm_t)
        global_xyzrxryrzs.append([x,y,z,rx,ry,rz])

        # 更新进度
        lock_counter.acquire()
        progress_counter.value += 1
        lock_counter.release()



def main():
    # 第一步: 读取图片和机械臂位姿. 准备计算
    imgs, tcps = get_imgs_and_tcps()

    # 将图片和tcp随机抽取进行计算
    num_indices = len(imgs)
    filtered = np.array([True]*num_samples,dtype=np.bool_)

    # 多进程计算
    import multiprocessing as mp
    num_workers = 100
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

if __name__ == '__main__':
    main()