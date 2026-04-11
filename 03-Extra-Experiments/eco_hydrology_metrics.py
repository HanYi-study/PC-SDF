import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata
import richdem as rd
from scipy.ndimage import uniform_filter
import os
import warnings

# Suppress runtime warnings for expected nan-mean calculations
warnings.filterwarnings("ignore", category=RuntimeWarning)

def xyz_to_dem(xyz_path, out_tif_path, resolution=0.5):
    """
    Module: Rasterize unstructured XYZ point clouds into regular DEM grids.
    """
    print(f"[*] Processing point cloud to DEM (Resolution: {resolution}m): {os.path.basename(xyz_path)}")
    
    # 1. Fast reading of TXT/XYZ files
    try:
        df = pd.read_csv(xyz_path, sep=r'\s+', header=None, comment='#')
    except:
        df = pd.read_csv(xyz_path, sep=',', header=None, comment='#')
        
    x, y, z = df.iloc[:, 0].values, df.iloc[:, 1].values, df.iloc[:, 2].values

    # 2. Compute Bounding Box
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    
    # 3. Calculate matrix dimensions based on resolution
    cols = int(np.ceil((xmax - xmin) / resolution))
    rows = int(np.ceil((ymax - ymin) / resolution))
    
    # 4. Generate regular 2D coordinate grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmax, cols),
        np.linspace(ymax, ymin, rows)
    )
    
    # 5. Spatial Interpolation
    print("    -> Executing Delaunay-based linear spatial interpolation...")
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
    
    # Fill boundary NaNs using nearest neighbor to ensure a complete raster
    invalid_mask = np.isnan(grid_z)
    if invalid_mask.any():
        grid_z_nearest = griddata((x, y), z, (grid_x, grid_y), method='nearest')
        grid_z[invalid_mask] = grid_z_nearest[invalid_mask]

    # 6. Export to standard GeoTIFF
    transform = from_origin(xmin, ymax, resolution, resolution)
    with rasterio.open(
        out_tif_path, 'w', driver='GTiff',
        height=rows, width=cols, count=1,
        dtype=grid_z.dtype, crs="EPSG:32650", # Modify CRS according to your specific survey area if necessary
        transform=transform, nodata=-9999.0
    ) as dst:
        dst.write(grid_z, 1)
        
    print(f"    -> Rasterization completed. Saved to: {out_tif_path}")
    return out_tif_path


def calculate_surface_roughness(dem_path, out_roughness_path, window_size=3):
    """
    Calculate Micro-topographic Surface Roughness (Ra) using local standard deviation.
    """
    print(f"[*] Calculating Surface Roughness (Ra): {os.path.basename(dem_path)}")
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        profile = src.profile
        nodata = src.nodata

    dem_data = dem_data.astype(np.float32)
    if nodata is not None:
        dem_data[dem_data == nodata] = np.nan

    # Highly optimized vectorized calculation for local variance
    c1 = uniform_filter(dem_data, window_size, mode='reflect')
    c2 = uniform_filter(dem_data**2, window_size, mode='reflect')
    variance = np.clip(c2 - c1**2, 0, None) 
    roughness = np.sqrt(variance)

    if nodata is None:
        nodata = -9999.0
        profile.update(nodata=nodata)
    roughness[np.isnan(dem_data)] = nodata

    profile.update(dtype=rasterio.float32)
    with rasterio.open(out_roughness_path, 'w', **profile) as dst:
        dst.write(roughness.astype(rasterio.float32), 1)
        
    mean_roughness = np.nanmean(roughness)
    print(f"    -> Mean Surface Roughness (Ra): {mean_roughness:.4f} m")
    return mean_roughness


def calculate_depression_storage(dem_path, out_depression_path):
    """
    Calculate Depression Storage (Vd) based on the Planchon-Darboux algorithm.
    """
    print(f"[*] Calculating Depression Storage (Vd): {os.path.basename(dem_path)}")
    
    with rasterio.open(dem_path) as src:
        pixel_area = src.res[0] * src.res[1]
        nodata = src.nodata
        profile = src.profile

    # Load DEM into richdem and execute sink filling
    dem_rd = rd.LoadGDAL(dem_path)
    filled_dem = rd.FillDepressions(dem_rd, epsilon=False, in_place=False)

    original_dem = np.array(dem_rd)
    filled_array = np.array(filled_dem)
    
    valid_mask = (original_dem != nodata) if nodata is not None else np.ones_like(original_dem, dtype=bool)
    
    # Calculate volumetric depth differences
    depression_depth = np.zeros_like(original_dem, dtype=np.float32)
    depression_depth[valid_mask] = filled_array[valid_mask] - original_dem[valid_mask]
    
    # Filter out microscopic floating-point noise
    depression_depth[depression_depth < 0.001] = 0 

    total_volume = np.sum(depression_depth) * pixel_area

    profile.update(dtype=rasterio.float32, nodata=-9999.0)
    depression_depth[~valid_mask] = -9999.0
    with rasterio.open(out_depression_path, 'w', **profile) as dst:
        dst.write(depression_depth, 1)

    print(f"    -> Maximum detected artificial sink depth: {np.max(depression_depth):.4f} m")
    print(f"    -> Total Depression Storage (Vd): {total_volume:.2f} m³")
    return total_volume


if __name__ == "__main__":
    # ==========================================
    # 1. Configuration & File Paths
    # ==========================================
    # Replace with your actual relative or absolute paths inside the repository
    xyz_baseline = os.path.join(BASE_DIR, "Data", "baseline_reconstruction_output.xyz")
    xyz_pcsdf = os.path.join(BASE_DIR, "Data", "PC_SDF_final_elevation.xyz")

    out_dir = os.path.join(BASE_DIR, "Data", "outputs")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Crucial: Define the spatial resolution matching the actual survey scale (e.g., 0.5 meters)
    RESOLUTION = 0.5 
    
    print("\n" + "="*60)
    print("PHASE 1: Point Cloud to DEM Rasterization")
    print("="*60)
    dem_base = xyz_to_dem(xyz_baseline, "Intermediate_SFS_DEM.tif", RESOLUTION)
    dem_pcsdf = xyz_to_dem(xyz_pcsdf, "Intermediate_PCSDF_DEM.tif", RESOLUTION)
    
    print("\n" + "="*60)
    print("PHASE 2: Baseline (SFS) Eco-hydrological Evaluation")
    print("="*60)
    roughness_base = calculate_surface_roughness(dem_base, "SFS_Roughness_Map.tif")
    volume_base = calculate_depression_storage(dem_base, "SFS_Depression_Map.tif")
    
    print("\n" + "="*60)
    print("PHASE 3: PC-SDF Eco-hydrological Evaluation")
    print("="*60)
    roughness_pcsdf = calculate_surface_roughness(dem_pcsdf, "PCSDF_Roughness_Map.tif")
    volume_pcsdf = calculate_depression_storage(dem_pcsdf, "PCSDF_Depression_Map.tif")
    
    print("\n" + "="*60)
    print("★ FINAL MANUSCRIPT METRICS COMPARISON ★")
    print("="*60)
    roughness_reduction = (roughness_base - roughness_pcsdf) / roughness_base * 100
    volume_reduction = (volume_base - volume_pcsdf) / volume_base * 100
    
    print(f"-> Surface Roughness Reduction (Artifact Suppression): {roughness_reduction:.2f}%")
    print(f"-> Depression Storage Elimination (Connectivity Recovery): {volume_reduction:.2f}%")
    print("="*60 + "\n")
