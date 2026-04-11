# PC-SDF: Physics-Constrained Sparse Data Fusion

This repository contains the evaluation toolkit and sample datasets for the manuscript: 
*"Large-scale Three-dimensional Wetland Reconstruction and Application Based on Physical Constraints Using Sparse Point Cloud and Image Data"*.

## Code Availability Statement
The PC-SDF framework implements the GSR and PDEC modules within a unified pipeline. For the DEM and point cloud registration process (PDEC), the complete and reproducible scripts are currently under code review and cleanup. **The full source code and extended datasets will be made publicly available upon the official acceptance or publication of the related manuscript.**

## Current Repository Contents
To facilitate the peer-review process and ensure metric transparency, we currently provide:
1. **Sample Yancheng Dataset**: A representative sub-region including the highly sparse LiDAR input (<0.1%), the monocular depth prior (DA-V2), the 100% dense ground truth, and our final PC-SDF predicted DEM.
2. **Evaluation Toolkit**: The metric calculation scripts used in our paper (RMSE, MAE, Pearson R, and Gradient Correlation).
3. **Data Initialization Script**: The standard script for extracting initial relative topography from monocular depth maps.

## Quick Start (Metric Validation)
Reviewers and researchers can independently verify the quantitative performance reported in our manuscript using the provided sample data

***python version:***
```>=3.8```  

***Requirements list:***  
```
numpy>=1.20
matplotlib>=3.4
scipy>=1.6
opencv-python>=4.5
Pillow>=8.0
laspy>=2.0
scikit-image>=0.18
cupy>=10
scikit-learn>=0.24
```  


