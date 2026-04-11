import numpy as np
from PIL import Image
import laspy
import os
import matplotlib.pyplot as plt

# =============== 可配置参数 ===============
depth_path = '/home/hy/project/01/DSC05911_depth-anything-v2.png'   # 深度图路径
las_path = '/home/hy/project/01/A/group2-gt/group2-gt.las'   # 点云路径
output_dir = '/home/hy/project/01/A/group2/DA-V2'         # 输出目录

# ==================== 自动标定点对齐相关函数 ====================
def get_uniform_cloud_points(points, n_points=15):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_points, random_state=42)
    idx = kmeans.fit_predict(points[:, :2])
    centroids = kmeans.cluster_centers_
    cloud_points = []
    for centroid in centroids:
        dist = np.linalg.norm(points[:, :2] - centroid, axis=1)
        nearest_idx = np.argmin(dist)
        cloud_points.append(points[nearest_idx])
    cloud_points = np.array(cloud_points)
    return cloud_points

def cloud_to_img_coords(x_cloud, y_cloud, min_x, max_x, min_y, max_y, w, h):
    x_img = int((x_cloud - min_x) / (max_x - min_x) * (w - 1))
    y_img = int((y_cloud - min_y) / (max_y - min_y) * (h - 1))
    return x_img, y_img

def extract_calib_pairs(cloud_points, depth, min_x, max_x, min_y, max_y):
    h, w = depth.shape
    img_points = []
    cloud_points_xy = []
    cloud_points_z = []
    calib_depth_values = []
    cloud_points_uv = []
    for pt in cloud_points:
        x_img, y_img = cloud_to_img_coords(pt[0], pt[1], min_x, max_x, min_y, max_y, w, h)
        x_img = np.clip(x_img, 0, w-1)
        y_img = np.clip(y_img, 0, h-1)
        img_points.append([x_img, y_img])
        cloud_points_xy.append([pt[0], pt[1]])
        cloud_points_uv.append([x_img, y_img])  # 归一化后的点云坐标
        cloud_points_z.append(pt[2])
        calib_depth_values.append(depth[y_img, x_img])
    return np.array(img_points), np.array(cloud_points_xy), np.array(cloud_points_z), np.array(calib_depth_values), np.array(cloud_points_uv)
# ==================== 新增部分结束 ====================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_and_log(arr, path, desc):
    np.savetxt(path, arr, fmt="%.4f")
    print(f"{desc} 已保存到: {path}")

def save_xyz(arr, path, desc):
    np.savetxt(path, arr, fmt="%.2f %.2f %.4f")
    print(f"{desc} 已保存到: {path}")

def plot_scatter(pred_z, real_z, out_path, title="拟合点分布"):
    plt.figure(figsize=(6,6))
    plt.scatter(pred_z, real_z, s=1, alpha=0.5)
    plt.xlabel("Pred Z (from depth image)")
    plt.ylabel("Real Z (from LAS)")
    plt.title(title)
    plt.grid()
    plt.savefig(out_path)
    plt.close()
    print(f"{title} 已保存到: {out_path}")

def plot_markers(img_points, cloud_points_uv, w, h, out_path):  
    plt.figure(figsize=(10,10))
    # 绘制红色点（影像像素坐标）
    plt.scatter(img_points[:,0], img_points[:,1], c='r', label='Image Points', s=60, zorder=2)
    # 绘制蓝色点（点云归一化后坐标）
    plt.scatter(cloud_points_uv[:,0], cloud_points_uv[:,1], c='b', label='Cloud XY', s=60, zorder=3)
    # 绘制绿色虚线（连接红蓝点）
    for i in range(len(img_points)):
        plt.plot([img_points[i,0], cloud_points_uv[i,0]], [img_points[i,1], cloud_points_uv[i,1]], 'g--', alpha=0.5, zorder=1)
    plt.xlim([0, w])
    plt.ylim([0, h])
    plt.legend()
    plt.title("Spatial Distribution of Sample Points")
    plt.savefig(out_path)
    plt.close()
    print(f"标定点空间分布图已保存到: {out_path}")

def fit_affine(src_xy, dst_uv):
    A = np.vstack([src_xy.T, np.ones((1, src_xy.shape[0]))]).T
    params_x, _, _, _ = np.linalg.lstsq(A, dst_uv[:,0], rcond=None)
    params_y, _, _, _ = np.linalg.lstsq(A, dst_uv[:,1], rcond=None)
    return params_x, params_y

def apply_affine(xy, params_x, params_y):
    A = np.vstack([xy.T, np.ones((1, xy.shape[0]))]).T
    u = A @ params_x
    v = A @ params_y
    return np.column_stack([u, v])

def fit_elevation_poly_linear(depths, zs):
    coeffs = np.polyfit(depths, zs, deg=1)  # 线性拟合
    return np.poly1d(coeffs)

def clip_z(z, z_min, z_max): 
    return np.clip(z, z_min, z_max)

def report_residual(pred_z, real_z, deg):  
    residuals = pred_z - real_z
    rmse = np.sqrt(np.mean(residuals**2))
    max_res = np.max(np.abs(residuals))
    mean_res = np.mean(np.abs(residuals))
    print(f"阶数deg={deg}, 高程拟合RMSE: {rmse:.2f}, 最大残差: {max_res:.2f}, 平均残差: {mean_res:.2f}")
    return rmse

if __name__ == "__main__":

    ensure_dir(output_dir)

    # 1. 读取深度图
    #depth = np.array(Image.open(depth_path)).astype(np.float32)
    depth = np.array(Image.open(depth_path).convert('L')).astype(np.float32)  # depth-anything-v2使用
    h, w = depth.shape
    print(f"深度图规格: {h} x {w}")
    print("深度图最大值:", depth.max(), "最小值:", depth.min())
    sample_u, sample_v = 400, 250  
    print("河流区域示例像素值:", depth[sample_v, sample_u])

    # 2. 读取全部点云数据
    las = laspy.read(las_path)
    points = las.xyz  # (N, 3)
    print("real_z.min():", points[:,2].min(), "real_z.max():", points[:,2].max())

    # 自动提取标定点，并获得归一化后的 cloud_points_uv
    min_x, max_x = points[:,0].min(), points[:,0].max()
    min_y, max_y = points[:,1].min(), points[:,1].max()
    n_calib = 15
    img_points, cloud_points_xy, cloud_points_z, calib_depth_values, cloud_points_uv = extract_calib_pairs(
        get_uniform_cloud_points(points, n_points=n_calib),
        depth, min_x, max_x, min_y, max_y
    )

    # 3. 可视化标定点对齐效果（显示红点、蓝点和绿色虚线）
    plot_markers(img_points, cloud_points_uv, w, h, os.path.join(output_dir, "calib_points_distribution.png"))

    # 4. 仿射空间变换
    params_x, params_y = fit_affine(cloud_points_xy, img_points)
    uv_points = apply_affine(points[:, :2], params_x, params_y)
    uv_points = np.round(uv_points).astype(int)
    uv_points[:,0] = np.clip(uv_points[:,0], 0, w-1)
    uv_points[:,1] = np.clip(uv_points[:,1], 0, h-1)
    uv_points[:,1] = h - 1 - uv_points[:,1]

    # 5. 线性拟合高程变换
    poly = fit_elevation_poly_linear(calib_depth_values, cloud_points_z)
    fit_z = poly(calib_depth_values)
    rmse = report_residual(fit_z, cloud_points_z, deg=1)
    plt.figure(figsize=(6,6))
    plt.scatter(calib_depth_values, cloud_points_z, c='b', label='Ground Truth Elevation')
    plt.scatter(calib_depth_values, fit_z, c='r', label='Fitted Elevation')
    plt.title(f"线性拟合, RMSE={rmse:.2f}")
    plt.xlabel('Depth Value')
    plt.ylabel('Elevation')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(output_dir, "linear_polyfit.png"))
    plt.close()
    print(f"线性拟合对比图已保存到: {os.path.join(output_dir, 'linear_polyfit.png')}")

    # 6. 影像坐标处获取深度，并用线性拟合公式得高程
    depth_for_points = depth[uv_points[:,1], uv_points[:,0]]
    pred_z = poly(depth_for_points)
    z_min, z_max = np.percentile(points[:,2], 2), np.percentile(points[:,2], 98)
    real_z = clip_z(points[:,2], z_min, z_max)
    
    mask = (real_z > z_min) & (real_z < z_max)
    report_residual(pred_z[mask], real_z[mask], deg=1)
    plot_scatter(pred_z[mask], real_z[mask], os.path.join(output_dir, "scatter_pred_real_linear.png"), "线性拟合点分布")

    # 7. 生成未约束的高程矩阵
    elevation_matrix = poly(depth)

    # 修正高程正负号（取反）
    elevation_matrix = -elevation_matrix

    # 可选：放大高程对比度（归一化并放大）
    elevation_matrix = (elevation_matrix - elevation_matrix.min()) / (elevation_matrix.max() - elevation_matrix.min())
    elevation_matrix = elevation_matrix * 500 + 100000  # 放大对比度，500可调

    # ===================== 后处理 =====================
    from scipy.ndimage import median_filter, uniform_filter
    elevation_matrix_filtered = median_filter(elevation_matrix, size=4)
    z_min, z_max = np.percentile(elevation_matrix_filtered.ravel(), 10), np.percentile(elevation_matrix_filtered.ravel(), 95)
    elevation_matrix_clipped = np.clip(elevation_matrix_filtered, z_min, z_max)
    elevation_matrix_smooth = uniform_filter(elevation_matrix_clipped, size=20)

    save_and_log(
        elevation_matrix_smooth,
        os.path.join(output_dir, "elevation_matrix_final.txt"),
        "最终高程矩阵（线性拟合+归一化放大+滤波+clip）"
    )

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    xyz_fitted = np.column_stack((xx.ravel(), yy.ravel(), elevation_matrix_smooth.ravel()))
    save_xyz(
        xyz_fitted,
        os.path.join(output_dir, "elevation_matrix_final_xyz.txt"),
        "最终高程矩阵XYZ"
    )

    print("✅ 所有结果已保存。")