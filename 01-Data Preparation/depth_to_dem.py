import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, uniform_filter

# =============== 可配置参数 ===============
depth_path = '/home/hy/project/01/DSC05911_depth-anything-v2.png'   # 深度图路径
output_dir = '/home/hy/project/01/A/group2/DA-V2'                 # 输出目录

# 构建相对地形的虚拟参数 (仅用于初步起伏表达，无绝对物理意义)
RELATIVE_Z_RANGE = 50.0  # 将深度映射到 0~50 的相对高程起伏中 (为了后续可视化和计算梯度，此值可自定)
INVERT_DEPTH = False     # 是否反转深度。DA-V2通常较亮处(值大)代表较近(高程较高)，若发现地形呈反向(坑变成了山)可设为True

# =========================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_and_log(arr, path, desc):
    np.savetxt(path, arr, fmt="%.4f")
    print(f"{desc} 已保存到: {path}")

def save_xyz(arr, path, desc):
    np.savetxt(path, arr, fmt="%.2f %.2f %.4f")
    print(f"{desc} 已保存到: {path}")

if __name__ == "__main__":
    ensure_dir(output_dir)

    # 1. 读取深度图
    print(">>> 正在加载深度图先验...")
    # 使用 'L' 模式读取灰度图
    depth_img = Image.open(depth_path).convert('L')
    depth_arr = np.array(depth_img).astype(np.float32)
    h, w = depth_arr.shape
    print(f"--> 深度图规格: {h} x {w}")
    print(f"--> 原始深度灰度值范围: {depth_arr.min()} ~ {depth_arr.max()}")

    # 2. 映射为相对高程 (Relative Elevation)
    # 首先将其严谨地归一化到 0~1 区间
    depth_norm = (depth_arr - depth_arr.min()) / (depth_arr.max() - depth_arr.min() + 1e-8)

    # 如果神经网络估计的深度逻辑与实际地形相反，进行翻转
    if INVERT_DEPTH:
        depth_norm = 1.0 - depth_norm

    # 映射到指定的相对高程区间，生成初步的相对 DEM
    elevation_matrix = depth_norm * RELATIVE_Z_RANGE
    print(f"--> 映射后相对高程范围: {elevation_matrix.min():.2f} ~ {elevation_matrix.max():.2f}")

    # 3. 形态学后处理 (完全保留原有的核心平滑与去噪逻辑)
    print(">>> 正在进行形态学平滑与去噪...")
    # 中值滤波：去除局部的脉冲噪声（黑白噪点）
    elevation_matrix_filtered = median_filter(elevation_matrix, size=4)
    
    # 截断极值：滤除网络预测时可能产生的极少数离群悬崖点
    z_min, z_max = np.percentile(elevation_matrix_filtered.ravel(), 2), np.percentile(elevation_matrix_filtered.ravel(), 98)
    elevation_matrix_clipped = np.clip(elevation_matrix_filtered, z_min, z_max)
    
    # 均值滤波：保证微地形起伏的平滑连续性
    elevation_matrix_smooth = uniform_filter(elevation_matrix_clipped, size=20)

    # 4. 保存结果用于下一步模块 (GSR / PDEC)
    print(">>> 正在保存相对 DEM 结果...")
    
    # 4.1 保存为 2D 矩阵 txt
    matrix_path = os.path.join(output_dir, "relative_elevation_matrix.txt")
    save_and_log(elevation_matrix_smooth, matrix_path, "相对高程 2D 矩阵")

    # 4.2 保存为 XYZ 格式点云 (使用像素坐标系作为 XY，用于后续与激光点云融合)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    # 注意：此时的 XYZ 是在像素坐标系下的相对形态，并未赋予真实地理坐标
    xyz_relative = np.column_stack((xx.ravel(), yy.ravel(), elevation_matrix_smooth.ravel()))
    xyz_path = os.path.join(output_dir, "relative_elevation_xyz.txt")
    save_xyz(xyz_relative, xyz_path, "相对地形 XYZ 文本文件")

    # 5. 可视化相对 DEM
    print(">>> 正在生成可视化伪彩图...")
    plt.figure(figsize=(10, 8))
    im = plt.imshow(elevation_matrix_smooth, cmap='jet')
    plt.colorbar(im, label='Relative Elevation (pseudo-scale)')
    plt.title('Initial Relative DEM (From Monocular Prior)')
    plt.axis('off')
    vis_path = os.path.join(output_dir, "relative_dem_visualization.png")
    plt.savefig(vis_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"--> 相对 DEM 可视化热力图已保存到: {vis_path}")

    print("✅ 初始相对拓扑 DEM 构建完成！可进入下一阶段 (PC-SDF)。")
