import numpy as np
import matplotlib.pyplot as plt
import cv2

# 加载.npy文件
color = np.load('/home/ur5/rekep/ReKepUR5_from_kinova/data/color_000005.npy')
color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

depth = np.load('/home/ur5/rekep/ReKepUR5_from_kinova/data/depth_000005.npy')

# 查看数据形状和数据类型，确认数据格式
print("Color 数据形状:", color_rgb.shape)
print("Color 数据类型:", color_rgb.dtype)
print("Depth 数据形状:", depth.shape)
print("Depth 数据类型:", depth.dtype)

# 创建子图，显示两幅图像
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 显示彩色图像
axes[0].imshow(color_rgb)  # color 通常是 (H, W, 3)
axes[0].set_title('Color Image')
axes[0].axis('off')  # 关闭坐标轴

# 显示深度图
axes[1].imshow(depth, cmap='gray')  # depth 可能是单通道数据
axes[1].set_title('Depth Image')
axes[1].axis('off')  # 关闭坐标轴

# 显示图像
plt.tight_layout()
plt.show()
