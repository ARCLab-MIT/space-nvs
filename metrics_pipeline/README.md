# Generalized Pipeline for 3D Model Processing

This repository contains four Python scripts designed for processing, analyzing, and visualizing 3D models. Below is a detailed explanation of each script, its purpose, and how to use it. This is part of the DreamSat project.

---

## 1. `create_view.py`

### Purpose
This script generates rendered views of 3D models from specified angles. These views can be used for tasks such as visual inspection or as input for 3D reconstruction models.

### How to Use
1. Ensure you have a 3D model in `.obj` format.
2. Update the `obj_file` and `output_img` variables in the script with the path to your `.obj` file and the desired output image path.
3. Run the script:
   ```bash
   python create_view.py
   ```
4. The rendered image will be saved to the specified output path.

---

## 2. `distance_metrics.py`

### Purpose
This script calculates distance metrics between original and reconstructed 3D models, including Chamfer Distance and Hausdorff Distance. It aligns point clouds using ICP registration and provides visualization capabilities with colored point clouds based on distance errors.

### How to Use
1. Update the file paths in the script:
   - Path to original `.obj` model
   - Path to reconstructed `.glb` model
2. Run the script:
   ```bash
   python distance_metrics.py
   ```
3. The script will output Chamfer and Hausdorff distances, generate a colored visualization image, and optionally save a colored `.ply` file.

---

## 3. `generalized_3d_metrics_pipeline.py`

### Purpose
This script processes a set of 3D models, calculates metrics (PSNR, SSIM, LPIPS) between original and reconstructed models, and outputs global statistics. It supports reconstructed models in both `.glb` and `.obj` formats.

### How to Use
1. Ensure the following directory structure:
   - `RECONS_DIR`: Directory containing reconstructed models in `.glb` or `.obj` format.
   - `OG_DIR`: Directory containing original models in `.obj` format.
2. Update the `RECONS_DIR` and `OG_DIR` variables in the script with the paths to your directories.
3. Run the script:
   ```bash
   python generalized_3d_metrics_pipeline.py
   ```
4. The script will process all models in the directories, calculate metrics, and print global statistics.

---

## 4. `iou_volumetric_fast.py`

### Purpose
This script calculates the volumetric Intersection over Union (IoU) between two 3D models. It uses PCA alignment to standardize the models and performs Monte Carlo sampling to estimate the volumetric overlap.

### How to Use
1. Update the file paths in the script:
   - `model1_path`: Path to the original `.obj` model
   - `model2_path`: Path to the reconstructed `.glb` model
2. Run the script:
   ```bash
   python iou_volumetric_fast.py
   ```
3. The script will output the volumetric IoU value and generate an overlay visualization of both models.

---

### Requirements

Install them using:
```bash
pip install -r requirements.txt
```

---

### Notes
- The scripts assume a Linux environment with EGL support for headless rendering.
- Modify paths and parameters in the scripts as needed for your specific use case.

---