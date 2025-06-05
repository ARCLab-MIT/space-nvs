# Generalized Pipeline for 3D Model Processing
# pon en ingles que es parte del proyecto DreamSat en la explcacion que hay justo abajo
This repository contains three Python scripts designed for processing, analyzing, and visualizing 3D models. Below is a detailed explanation of each script, its purpose, and how to use it. This is part of the DreamSat project.

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

## 2. `compare_3d_models.py`

### Purpose
This script compares an original 3D model (`.obj`) with a reconstructed 3D model (`.glb`). It calculates vertex distances, applies a heatmap to visualize errors, and generates a comparison image. Optionally, it can export a colored `.ply` file.

### How to Use
1. Update the following variables in the script:
   - `obj_path`: Path to the original `.obj` model.
   - `glb_path`: Path to the reconstructed `.glb` model.
   - `img_output_path`: Path to save the output comparison image.
   - `ply_output_path`: Path to save the colored `.ply` file (optional).
2. Run the script:
   ```bash
   python compare_3d_models.py
   ```
3. The script will output the mean and maximum vertex errors and save the comparison image and/or `.ply` file.

---

## 3. `generalized_3d_metrics_pipeline.py`

### Purpose
This script processes a set of 3D models, calculates metrics (PSNR, SSIM, LPIPS) between original and reconstructed models, and outputs global statistics. It now supports reconstructed models in both `.glb` and `.obj` formats.

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