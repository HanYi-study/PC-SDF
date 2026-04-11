import numpy as np
import os
import cv2
import laspy
from scipy import stats
from sklearn.metrics import mean_squared_error

# =======================================================
# 1. Data Loading and Mapping Modules
# =======================================================

def load_dem_matrix(file_path):
    """Load the predicted DEM matrix."""
    if not os.path.exists(file_path):
        print(f"[Error] Prediction file not found: {file_path}")
        return None
    data = np.loadtxt(file_path)
    
    # Assuming XYZ format, convert to 2D grid
    if data.ndim == 2 and data.shape[1] == 3:
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        ux, uy = np.unique(x), np.unique(y)
        dx_step = ux[1]-ux[0] if len(ux)>1 else 1.0
        dy_step = uy[1]-uy[0] if len(uy)>1 else 1.0
        xi, yi = ((x-x.min())/dx_step).astype(int), ((y-y.min())/dy_step).astype(int)
        grid = np.zeros((yi.max()+1, xi.max()+1))
        grid[yi, xi] = z
        return grid
    return data

def map_gt_las_to_grid(las_path, dem_w, dem_h):
    """Map the high-density Ground Truth LAS to the DEM grid size."""
    if not os.path.exists(las_path):
        print(f"[Error] Ground Truth LiDAR file not found: {las_path}")
        return None, None, None
    las = laspy.read(las_path)
    
    # Convert cm to meters if applicable (adjust based on your data)
    rx, ry, rz = np.array(las.x) / 100.0, np.array(las.y) / 100.0, np.array(las.z) / 100.0

    # Filter out extreme noise outliers in GT
    p_low, p_high = np.percentile(rz, [1.0, 99.0])
    valid_mask = (rz >= p_low) & (rz <= p_high)
    rx_c, ry_c, rz_c = rx[valid_mask], ry[valid_mask], rz[valid_mask]
    
    # Map spatial coordinates to grid indices
    x_span, y_span = rx_c.max() - rx_c.min(), ry_c.max() - ry_c.min()
    px = ((rx_c - rx_c.min()) / x_span * (dem_w - 1)).astype(int)
    py = ((ry_c - ry_c.min()) / y_span * (dem_h - 1)).astype(int)
    
    return px, py, rz_c

# =======================================================
# 2. Core Evaluation Module: Sub-regional Analysis
# =======================================================

def evaluate_partitioned_area(pred_path, gt_path, mask_path, area_name):
    print(f"\n========== [Evaluating {area_name}] ==========")
    
    # 1. Load predicted DEM
    pred_dem = load_dem_matrix(pred_path)
    if pred_dem is None: return None
    h, w = pred_dem.shape
    
    # 2. Load and map Full-Density Ground Truth Point Cloud
    px_gt, py_gt, pz_gt = map_gt_las_to_grid(gt_path, w, h)
    if px_gt is None: return None
    
    # Optional: If coordinate systems strictly require alignment, perform local shift (dx, dy)
    # Note: For strict geometric evaluation, spatial alignment should ideally be achieved prior to this step.
    # Here we assume px_gt and py_gt directly correspond to the DEM grid indices (dx=0, dy=0).
    valid_idx = (px_gt >= 0) & (px_gt < w) & (py_gt >= 0) & (py_gt < h)
    
    # Extract overlapping valid points
    v_dem = pred_dem[py_gt[valid_idx], px_gt[valid_idx]]
    v_gt = pz_gt[valid_idx]
    
    # 3. Calculate Global Metrics (Strict evaluation without re-calibration)
    global_rmse = np.sqrt(mean_squared_error(v_gt, v_dem))
    global_r, _ = stats.pearsonr(v_dem, v_gt)
    
    # 4. Sub-regional Semantic Masking (Land vs. Water)
    if not os.path.exists(mask_path):
        print(f"[Warning] Mask file not found: {mask_path}. Skipping sub-regional analysis.")
        return {"Area": area_name, "RMSE(All)": global_rmse, "P(All)": global_r, "P(Land)": 0, "P(Water)": 0}

    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Extract mask values corresponding to valid points
    mask_vals = mask_img[py_gt[valid_idx], px_gt[valid_idx]]
    
    # Define semantic regions based on threshold
    is_land = mask_vals < 128
    is_water = mask_vals >= 128
    
    pred_land, gt_land = v_dem[is_land], v_gt[is_land]
    pred_water, gt_water = v_dem[is_water], v_gt[is_water]
    
    # Calculate partitioned Pearson Correlation
    land_r = stats.pearsonr(pred_land, gt_land)[0] if (len(pred_land) > 10 and np.std(pred_land) > 1e-5) else 0.0
    water_r = stats.pearsonr(pred_water, gt_water)[0] if (len(pred_water) > 10 and np.std(pred_water) > 1e-5) else 0.0

    print(f"[Success] Processed {len(v_gt)} valid validation points.")
    
    return {
        "Area": area_name,
        "RMSE(All)": global_rmse,
        "P(All)": global_r,
        "P(Land)": land_r,
        "P(Water)": water_r
    }

# =======================================================
# 3. Main Execution Pipeline
# =======================================================
if __name__ == "__main__":
    results = []

    # Evaluate Area 1
    res1 = evaluate_partitioned_area(
        pred_path=os.path.join(BASE_DIR, "Data", "PC_SDF_group1_elevation.txt"), 
        gt_path=os.path.join(BASE_DIR, "Data", "group1-gt-new.las"),
        mask_path=os.path.join(BASE_DIR, "Data", "DSC04644_mask.png"),
        area_name="Area 1"
    )
    if res1: results.append(res1)

    # Evaluate Area 2
    res2 = evaluate_partitioned_area(
        pred_path=os.path.join(BASE_DIR, "Data", "PC_SDF_group2_elevation.txt"),
        gt_path=os.path.join(BASE_DIR, "Data", "group2-gt.las"),
        mask_path=os.path.join(BASE_DIR, "Data", "DSC04644_mask.png"),
        area_name="Area 2"
    )
    if res2: results.append(res2)

    # --- Print Final Formatting Table ---
    if results:
        print("\n" + "="*85)
        print("### PC-SDF Sub-regional Quantitative Evaluation Results")
        print("="*85)
        print(f"| {'Region':<8} | {'RMSE(Global)':<12} | {'Pearson(Global)':<15} | {'Pearson(Land)':<15} | {'Pearson(Water)':<15} |")
        print("|" + "-"*10 + "|" + "-"*14 + "|" + "-"*17 + "|" + "-"*17 + "|" + "-"*17 + "|")
        for r in results:
            print(f"| {r['Area']:<8} | {r['RMSE(All)']:12.4f} | {r['P(All)']:15.4f} | {r['P(Land)']:15.4f} | {r['P(Water)']:15.4f} |")
        print("="*85)
