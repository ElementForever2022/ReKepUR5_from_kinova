import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def draw_6d_pose_by_angle_deg(
    img: np.ndarray,
    centers: dict,
    axis_angles_deg_per_obj: dict,
    axis_lengths_per_obj: dict,
    axis_colors: dict,
    axis_thickness_per_obj: dict,
    draw_order_per_obj: dict,
    label_to_id: dict,
    circle_radius: int = 20,
    circle_fill_color: tuple = (255, 255, 255),
    circle_border_color: tuple = (0, 0, 0),
    circle_border_thickness: int = 3,
    number_color: tuple = (0, 0, 0),
    number_font_scale: float = 0.8,
    number_thickness: int = 2
) -> np.ndarray:
    """
    在图像上为每个物体绘制 6D 位姿，并在最后加上显眼的编号圆。

    新增参数:
    - label_to_id[label] = 整数编号
    - circle_radius: 编号圆半径（像素）
    - circle_fill_color: 圆心填充色 BGR
    - circle_border_color: 圆边框色 BGR
    - circle_border_thickness: 边框线宽
    - number_color: 数字颜色 BGR
    - number_font_scale: 数字字体缩放
    - number_thickness: 数字线宽
    """
    canvas = img.copy()

    # 1. 绘制所有轴线
    for label, (cx, cy) in centers.items():
        angs_deg   = axis_angles_deg_per_obj[label]
        lengths    = axis_lengths_per_obj[label]
        thickness  = axis_thickness_per_obj[label]
        draw_order = draw_order_per_obj[label]

        for axis in draw_order:
            θ = np.deg2rad(angs_deg[axis])
            dx, dy = np.cos(θ), np.sin(θ)
            L = lengths[axis]
            T = thickness[axis]
            pt2 = (int(cx + dx * L), int(cy + dy * L))
            cv2.line(canvas, (cx, cy), pt2,
                     axis_colors[axis], thickness=T)

    # 2. 绘制编号圆（置顶）
    for label, (cx, cy) in centers.items():
        # 圆形背景填充
        cv2.circle(canvas, (cx, cy),
                   circle_radius,
                   circle_fill_color,
                   thickness=-1)
        # 圆形边框
        cv2.circle(canvas, (cx, cy),
                   circle_radius,
                   circle_border_color,
                   thickness=circle_border_thickness)
        # 数字居中
        num = str(label_to_id.get(label, ''))
        (w_txt, h_txt), _ = cv2.getTextSize(
            num,
            cv2.FONT_HERSHEY_SIMPLEX,
            number_font_scale,
            number_thickness
        )
        txt_x = cx - w_txt // 2
        txt_y = cy + h_txt // 2
        cv2.putText(canvas,
                    num,
                    (txt_x, txt_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    number_font_scale,
                    number_color,
                    number_thickness)

    return canvas

if __name__ == "__main__":
    # ==== 保持原始输入/输出路径 不变 ====
    input_image_path  = "/home/liwenbo/project/yt/ReKepUR5_from_kinova/data/color_000001.png"
    output_image_path = "/home/liwenbo/project/yt/ReKepUR5_from_kinova/draw/6d_pose_img.png"

    if not os.path.isfile(input_image_path):
        raise FileNotFoundError(f"找不到图像：{input_image_path}")
    img = cv2.imread(input_image_path)
    if img is None:
        raise RuntimeError(f"cv2.imread 失败，请确认图像格式：{input_image_path}")

    centers = {
      "bear": (250, 340),
      "cat":  (470, 340)
    }

    axis_angles_deg_per_obj = {
      "bear": {"x": 60,   "y": -20,  "z": -100},
      "cat":  {"x": -91,  "y":   0,  "z":  -89}
    }

    axis_lengths_per_obj = {
      "bear": {"x": 70,   "y": 80,  "z": 80},
      "cat":  {"x": 90,   "y": 95,  "z": 70}
    }

    axis_colors = {
      "x": (0,   0,   255),
      "y": (0,   255, 255),
      "z": (255, 0,   0)
    }

    axis_thickness_per_obj = {
      "bear": {"x": 4, "y": 4, "z": 4},
      "cat":  {"x": 4, "y": 4, "z": 4}
    }

    draw_order_per_obj = {
      "bear": ['z', 'y', 'x'],
      "cat":  ['z', 'y', 'x']
    }

    # 给每个 label 定义一个编号
    label_to_id = {
      "bear": 1,
      "cat":  2
    }

    # 调用并保存
    out = draw_6d_pose_by_angle_deg(
        img,
        centers,
        axis_angles_deg_per_obj,
        axis_lengths_per_obj,
        axis_colors,
        axis_thickness_per_obj,
        draw_order_per_obj,
        label_to_id,
        circle_radius=15,              # 圆半径
        circle_fill_color=(255,255,255),   # 白色填充
        circle_border_color=(0,0,0),       # 黑色边框
        circle_border_thickness=3,
        number_color=(0,0,0),
        number_font_scale=0.5,#字号
        number_thickness=2
    )
    cv2.imwrite(output_image_path, out)
    print("已保存绘制结果到:", output_image_path)

    plt.figure(figsize=(8,6))
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
