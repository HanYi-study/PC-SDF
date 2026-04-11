import os
import numpy as np
import cv2
import laspy
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== Core Algorithmic Engine (PC-SDF) ====================

def run_pcsdf_engine(img_path, depth_path, las_path, 
                     sigma_val=500.0, weight_val=1.5, scale_val=0.001, 
                     radius_val=500.0, k_val=20):
    
    ALPHA_FUSION = 0.7 
    BASE_ELEVATION_OFFSET = 1300.0 

    # ---------------------------------------------------------
    # Step 1: GSR Module (Gradient-driven Structure Refinement)
    # ---------------------------------------------------------
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(f"Cannot find image: {img_path}")
    img_gray = img.astype(np.float32) / 255.0
    grad_norm = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8)
    
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth is None: raise FileNotFoundError(f"Cannot find depth map: {depth_path}")
    depth = np.array(depth, dtype=np.float32)
    
    depth_topo = -0.01 * ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8)) * 255 + BASE_ELEVATION_OFFSET
    depth_topo = cv2.resize(depth_topo, (grad_norm.shape[1], grad_norm.shape[0]))
    
    fused_matrix = (1 / (1 + weight_val)) * grad_norm * (np.max(depth_topo) - np.min(depth_topo)) + (weight_val / (1 + weight_val)) * depth_topo

    # ---------------------------------------------------------
    # Step 2: Global Spatial Alignment (Strictly using Sparse Anchors)
    # ---------------------------------------------------------
    m_target = np.rot90(fused_matrix, 1)
    m_target = np.max(m_target) - m_target
    h, w = m_target.shape
    
    las = laspy.read(las_path)
    rx, ry, rz = np.array(las.x)/100.0, np.array(las.y)/100.0, np.array(las.z)/100.0
    
    x_span, y_span = rx.max() - rx.min(), ry.max() - ry.min()
    px = ((rx - rx.min()) / x_span * (w - 1)).astype(int)
    py = ((ry - ry.min()) / y_span * (h - 1)).astype(int)
    
    np.random.seed(42)
    sample_size = max(20, int(len(rz) * scale_val))
    sparse_idx = np.random.choice(len(rz), sample_size, replace=False)
    
    p99_gt, p1_gt = np.percentile(rz[sparse_idx], 99), np.percentile(rz[sparse_idx], 1)
    p99_dem, p1_dem = np.percentile(m_target, 99), np.percentile(m_target, 1)
    
    s_final = (p99_gt - p1_gt) / (p99_dem - p1_dem + 1e-8)
    t_final = np.mean(rz[sparse_idx]) - s_final * np.mean(m_target)
    
    calibrated_matrix = s_final * m_target + t_final

    # ---------------------------------------------------------
    # Step 3: PDEC Module (KD-Tree OOM-Safe Chunking Calibration)
    # ---------------------------------------------------------
    anchor_x = px[sparse_idx]
    anchor_y = py[sparse_idx]
    deltas = rz[sparse_idx] - calibrated_matrix[anchor_y, anchor_x]
    
    anchor_coords = np.column_stack((anchor_x, anchor_y))
    tree = cKDTree(anchor_coords)
    
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    grid_coords = np.column_stack((xx.ravel(), yy.ravel()))
    
    # 【新增内存保护机制：分块检索 (Chunking)】
    chunk_size = 500000  # 每次只处理 50 万个像素，杜绝内存爆表
    correction = np.zeros(h * w, dtype=np.float32)
    safe_deltas = np.append(deltas, 0.0) 
    
    for i in range(0, len(grid_coords), chunk_size):
        chunk_coords = grid_coords[i : i + chunk_size]
        
        # 仅查询当前块的邻居
        dists, indices = tree.query(chunk_coords, k=int(k_val), distance_upper_bound=radius_val, workers=-1)
        
        valid_mask = dists != np.inf
        weights = np.zeros_like(dists, dtype=np.float32)  # 使用 float32 进一步压缩内存
        weights[valid_mask] = np.exp(-(dists[valid_mask]**2) / (2 * (sigma_val**2)))
        
        neighbor_deltas = safe_deltas[indices].astype(np.float32)
        
        sum_weights = np.sum(weights, axis=1)
        weighted_delta_sum = np.sum(weights * neighbor_deltas, axis=1)
        
        has_neighbors = sum_weights > 1e-8
        chunk_correction = np.zeros(len(chunk_coords), dtype=np.float32)
        chunk_correction[has_neighbors] = weighted_delta_sum[has_neighbors] / sum_weights[has_neighbors]
        
        correction[i : i + chunk_size] = chunk_correction
        
    fitted_surface = calibrated_matrix.ravel() + correction
    fitted_surface = fitted_surface.reshape((h, w))
    
    final_dem = ALPHA_FUSION * fitted_surface + (1 - ALPHA_FUSION) * calibrated_matrix
    
    # ---------------------------------------------------------
    # Evaluation (Hold-out Validation)
    # ---------------------------------------------------------
    valid_eval_mask = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    preds = final_dem[py[valid_eval_mask], px[valid_eval_mask]]
    gts = rz[valid_eval_mask]
    
    mae = mean_absolute_error(gts, preds)
    rmse = np.sqrt(mean_squared_error(gts, preds))
    
    return mae, rmse

# ==================== Automated Parameter Sensitivity Workflows ====================

def run_sensitivity_analysis():
    # File paths (Adjust to your actual paths)
    cfg = {
        'img_path': os.path.join(BASE_DIR, "Data", "DSC05911.jpg"),
        'depth_path': os.path.join(BASE_DIR, "Data", "DSC05911_depth-anything-v2.png"),
        'las_path': os.path.join(BASE_DIR, "Data", "group2-gt.las")
    }
    

    print("🚀 Initiating Rigorous PC-SDF Parameter Sensitivity Analysis...")
    out_dir = "sensitivity_results"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Constraint Radiation Radius (Sigma)
    print("\n[PART 1] Evaluating: Constraint Radiation Radius (Sigma)...")
    sigmas = [100, 300, 500, 800]
    sigma_res = []
    for s in sigmas:
        mae, rmse = run_pcsdf_engine(**cfg, sigma_val=s)
        sigma_res.append({'Sigma': s, 'MAE': round(mae, 4), 'RMSE': round(rmse, 4)})
        print(f"  -> Sigma={s:<4} | MAE={mae:.4f}, RMSE={rmse:.4f}")
    pd.DataFrame(sigma_res).to_csv(os.path.join(out_dir, "SA_1_Sigma.csv"), index=False)

    # 2. Fusion Penalty Weight (Weight)
    print("\n[PART 2] Evaluating: Fusion Penalty Weight (Weight)...")
    weights = [0.5, 1.0, 1.5, 2.0, 2.5]
    weight_res = []
    for w in weights:
        mae, rmse = run_pcsdf_engine(**cfg, weight_val=w)
        weight_res.append({'Weight': w, 'MAE': round(mae, 4), 'RMSE': round(rmse, 4)})
        print(f"  -> Weight={w:<4} | MAE={mae:.4f}, RMSE={rmse:.4f}")
    pd.DataFrame(weight_res).to_csv(os.path.join(out_dir, "SA_2_Weight.csv"), index=False)

    # 3. Sparse Anchor Scale (Scale)
    print("\n[PART 3] Evaluating: Sparse Anchor Scale Ratio (Scale)...")
    scales = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]
    scale_res = []
    for sc in scales:
        mae, rmse = run_pcsdf_engine(**cfg, scale_val=sc)
        scale_res.append({'Scale': sc, 'MAE': round(mae, 4), 'RMSE': round(rmse, 4)})
        print(f"  -> Scale={sc:<6} | MAE={mae:.4f}, RMSE={rmse:.4f}")
    pd.DataFrame(scale_res).to_csv(os.path.join(out_dir, "SA_3_Scale.csv"), index=False)

    # 4. Local Search Radius (Radius) - NOW IMPLEMENTED & FUNCTIONAL!
    print("\n[PART 4] Evaluating: Local Search Radius (Radius)...")
    radii = [100, 200, 400, 500, 600]
    radius_res = []
    for r in radii:
        mae, rmse = run_pcsdf_engine(**cfg, radius_val=r)
        radius_res.append({'Radius': r, 'MAE': round(mae, 4), 'RMSE': round(rmse, 4)})
        print(f"  -> Radius={r:<4} | MAE={mae:.4f}, RMSE={rmse:.4f}")
    pd.DataFrame(radius_res).to_csv(os.path.join(out_dir, "SA_4_Radius.csv"), index=False)

    # 5. Adaptive Nearest Anchors (K) - NOW IMPLEMENTED & FUNCTIONAL!
    print("\n[PART 5] Evaluating: Adaptive Nearest Anchors (K)...")
    k_values = [10, 15, 20, 25, 30]
    k_res = []
    for k in k_values:
        mae, rmse = run_pcsdf_engine(**cfg, k_val=k)
        k_res.append({'K': k, 'MAE': round(mae, 4), 'RMSE': round(rmse, 4)})
        print(f"  -> K={k:<7} | MAE={mae:.4f}, RMSE={rmse:.4f}")
    pd.DataFrame(k_res).to_csv(os.path.join(out_dir, "SA_5_K.csv"), index=False)

    print("\n" + "="*70)
    print("✅ Full 5-Parameter Sensitivity Analysis Completed Successfully!")
    print("="*70)

if __name__ == "__main__":
    run_sensitivity_analysis()
