# Final Framework - 3D Model Analysis Tools

This repository contains a comprehensive set of Python tools for analyzing and comparing 3D models, specifically designed for evaluating reconstruction quality and generating visualizations.

## 🚀 Overview

The framework provides tools for:
- 2D image-based metrics comparison between original and reconstructed 3D models
- 3D geometric metrics and volumetric analysis
- Visual comparison with error heatmaps
- Automated batch processing of model datasets

## 📋 Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

Main dependencies include:
- `numpy` - Numerical computations
- `trimesh` - 3D mesh processing
- `pyrender` - 3D rendering
- `torch` - Deep learning framework
- `lpips` - Perceptual image similarity
- `opencv-python` - Image processing
- `scikit-image` - Image metrics (SSIM)
- `matplotlib` - Plotting and visualization
- `Pillow` - Image handling
- `open3d` - 3D data processing

## 📁 Files Description

### 🖼️ `2D_metrics.py`
Comprehensive tool for calculating 2D image-based metrics between original and reconstructed 3D models.

**Features:**
- Renders 3D models from multiple angles (0° to 360° in 30° increments)
- Calculates PSNR (Peak Signal-to-Noise Ratio)
- Computes SSIM (Structural Similarity Index)
- Measures LPIPS (Learned Perceptual Image Patch Similarity)
- Generates comparison visualizations
- Exports detailed results in JSON and CSV formats

**Usage:**
```python
python 2D_metrics.py
```

**Key Functions:**
- `render_mesh()` - Renders 3D mesh to 2D image
- `calculate_psnr()` - PSNR metric calculation
- `calculate_ssim()` - SSIM metric calculation
- `calculate_lpips()` - LPIPS perceptual similarity
- `main()` - Complete pipeline for batch processing

### 📐 `3D_metrics.py`
Advanced 3D geometric analysis and volumetric metrics calculator.

**Features:**
- PCA-based mesh alignment
- Volumetric IoU (Intersection over Union) calculation
- Point cloud distance metrics
- Hausdorff distance computation
- Mesh surface area and volume analysis
- Visual overlays and distance heatmaps

**Usage:**
```python
python 3D_metrics.py
```

**Key Components:**
- `UnifiedMetrics` class - Main analysis framework
- PCA alignment for consistent orientation
- Voxelization for volumetric analysis
- Distance field calculations
- Comprehensive statistical reporting

### 🔍 `compare_3d_models.py`
Specialized tool for vertex-level distance analysis and error visualization.

**Features:**
- Vertex-to-vertex distance calculation
- Error heatmap generation on 3D models
- Colored PLY export with error mapping
- Batch processing with comprehensive statistics
- Identification of worst-performing models

**Usage:**
```python
python compare_3d_models.py
```

**Key Functions:**
- `normalize_and_center_mesh()` - Mesh preprocessing
- `compute_vertex_distances()` - Distance calculation using KD-trees
- `apply_error_colors()` - Error visualization mapping
- `compare_models_batch()` - Automated batch processing

**Output:**
- Error visualization images
- Colored PLY files with distance mapping
- Statistical summaries
- Text reports with rankings

### 🎨 `create_view.py`
Simple utility for generating standardized 3D model visualizations.

**Features:**
- Standardized mesh rendering
- Consistent lighting and camera setup
- Batch processing for model folders
- Gray-scale visualization generation

**Usage:**
```python
python create_view.py
```

**Key Functions:**
- `render_simple_obj_model()` - Basic mesh rendering
- `process_obj_folder()` - Batch folder processing
- Normalized mesh sizing and centering

## 🛠️ Configuration

Each script includes configurable parameters at the top:

### Directory Paths
- `RECONS_DIR` - Reconstructed models directory
- `OG_DIR` - Original models directory  
- `TEMP_DIR` - Temporary files directory
- `RESULTS_DIR` - Output results directory

### Rendering Parameters
- `ANGLES` - Camera angles for multi-view rendering
- Image resolution and quality settings
- Lighting and material properties


## 🚀 Quick Start

1. **Setup your data directories** with original (.obj) and reconstructed (.glb/.obj) models
2. **Update the paths** in each script to match your directory structure
3. **Run the desired analysis:**

```bash
# For 2D image-based metrics
python 2D_metrics.py

# For 3D geometric metrics  
python 3D_metrics.py

# For detailed distance analysis
python compare_3d_models.py

# For simple visualizations
python create_view.py
```

## 📈 Metrics Explained

### 2D Metrics
- **PSNR**: Higher values indicate better quality (typically 20-40 dB)
- **SSIM**: Range 0-1, where 1 is perfect similarity
- **LPIPS**: Lower values indicate better perceptual similarity

### 3D Metrics
- **IoU**: Volumetric intersection over union (0-1 scale)
- **Hausdorff Distance**: Maximum distance between surfaces
- **Mean Distance**: Average vertex-to-surface distance
- **Surface Area Ratio**: Reconstructed/Original surface area

## 🔧 Troubleshooting

### Common Issues
- **OpenGL/Rendering**: The scripts use headless EGL rendering
- **Memory**: Large models may require significant RAM
- **Dependencies**: Ensure all packages are correctly installed

### Performance Tips
- Use smaller angle increments for faster processing
- Adjust image resolution based on requirements
- Consider mesh simplification for large datasets

## 📝 Notes

- All scripts are designed for headless operation (no GUI required)
- The framework supports both .obj and .glb model formats
- Results include both numerical metrics and visual outputs
- Model 96 is automatically skipped in batch processing (configurable)
