import os
import numpy as np
import trimesh
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import lpips
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
import pyrender
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Force Pyrender to use headless mode (EGL) if available
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Configurable parameters
RECONS_DIR = "/mnt/data/sdiaz/final_framework/aux/asteroids/rec"  # Directory for reconstructed models
OG_DIR = "/mnt/data/sdiaz/final_framework/aux/asteroids/og"          # Directory for original models
TEMP_DIR = "/mnt/data/sdiaz/final_framework/aux/temp_imgs"       # Temporary directory for intermediate images
RESULTS_DIR = "/mnt/data/sdiaz/final_framework/aux/2d_metrics"  # Directory for results
ANGLES = list(range(0, 360, 30))  # Angles for rendering
GRAY_COLOR = [100, 100, 100, 255]  # Default color for meshes

# Initialize LPIPS loss function
lpips_loss_fn = lpips.LPIPS(net='vgg')
lpips_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def setup_directories():
    """Creates necessary directories."""
    import shutil
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

def clear_temp_directory():
    """Clears and recreates the temporary directory."""
    import shutil
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

def normalize_and_center_mesh(mesh, target_size=1.0):
    """Normalizes and centers a mesh to a target size."""
    bbox = mesh.bounding_box.extents
    scale = target_size / np.linalg.norm(bbox)
    mesh.apply_scale(scale)
    mesh.apply_translation(-mesh.bounding_box.centroid)
    return mesh

def render_mesh(mesh, filename, angle_deg=45):
    """Renders a mesh to an image file from a specified angle."""
    angle = np.radians(angle_deg)
    rotation_y = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
    mesh_copy = mesh.copy()
    mesh_copy.apply_transform(rotation_y)

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=[0.2, 0.2, 0.2])
    scene.add(pyrender.Mesh.from_trimesh(mesh_copy, smooth=False))

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0,  0.0, 0.0],
        [0.0, 1.0,  0.0, 0.0],
        [0.0, 0.0,  1.0, 1.0],
        [0.0, 0.0,  0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(640, 480)
    color, _ = r.render(scene)
    Image.fromarray(color).save(filename)

def calculate_psnr(img1, img2):
    """Calculates the PSNR between two images."""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculates the SSIM between two images."""
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return ssim(img1, img2, data_range=img2.max() - img2.min())

def calculate_lpips(img1_path, img2_path):
    """Calculates the LPIPS metric between two images."""
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    img1 = lpips_transform(img1).unsqueeze(0)
    img2 = lpips_transform(img2).unsqueeze(0)
    with torch.no_grad():
        dist = lpips_loss_fn(img1, img2).item()
    return dist

def save_results_to_txt(results_data, timestamp):
    """Saves results to a text file."""
    txt_path = os.path.join(RESULTS_DIR, f"3d_metrics_results_{timestamp}.txt")
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("         3D METRICS RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total models processed: {len(results_data['individual_results'])}\n")
        f.write(f"Rendering angles: {ANGLES}\n\n")
        
        # Global results
        f.write("GLOBAL METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"PSNR  -> Mean: {results_data['global_stats']['psnr']['mean']:.2f} | "
                f"Max: {results_data['global_stats']['psnr']['max']:.2f} | "
                f"Min: {results_data['global_stats']['psnr']['min']:.2f}\n")
        f.write(f"SSIM  -> Mean: {results_data['global_stats']['ssim']['mean']:.4f} | "
                f"Max: {results_data['global_stats']['ssim']['max']:.4f} | "
                f"Min: {results_data['global_stats']['ssim']['min']:.4f}\n")
        f.write(f"LPIPS -> Mean: {results_data['global_stats']['lpips']['mean']:.4f} | "
                f"Max: {results_data['global_stats']['lpips']['max']:.4f} | "
                f"Min: {results_data['global_stats']['lpips']['min']:.4f}\n\n")
        
        # Individual results
        f.write("INDIVIDUAL RESULTS BY MODEL:\n")
        f.write("-" * 40 + "\n")
        for model_name, metrics in results_data['individual_results'].items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  PSNR:  {metrics['psnr']:.2f}\n")
            f.write(f"  SSIM:  {metrics['ssim']:.4f}\n")
            f.write(f"  LPIPS: {metrics['lpips']:.4f}\n")
    
    print(f"✅ Results saved to: {txt_path}")
    return txt_path



def create_results_visualization(results_data, timestamp):
    """Creates enhanced visualization charts for scientific reports."""
    # Set style for publication quality
    plt.style.use('default')  # Use default style for better control
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('3D Reconstruction Quality Metrics Analysis', fontsize=20, fontweight='bold', y=0.95)
    
    # Extract data
    model_names = list(results_data['individual_results'].keys())
    psnr_vals = [results_data['individual_results'][name]['psnr'] for name in model_names]
    ssim_vals = [results_data['individual_results'][name]['ssim'] for name in model_names]
    lpips_vals = [results_data['individual_results'][name]['lpips'] for name in model_names]
    
    # Color scheme for consistency
    colors = {'psnr': '#2E8B57', 'ssim': '#4682B4', 'lpips': '#CD5C5C'}
    
    # 1. PSNR histogram with statistical info
    axes[0, 0].hist(psnr_vals, bins=15, alpha=0.7, color=colors['psnr'], edgecolor='black', linewidth=1)
    axes[0, 0].set_title('PSNR Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('PSNR (dB)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(psnr_vals), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(psnr_vals):.2f} dB')
    axes[0, 0].axvline(np.median(psnr_vals), color='orange', linestyle=':', linewidth=2,
                       label=f'Median: {np.median(psnr_vals):.2f} dB')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. SSIM histogram with statistical info
    axes[0, 1].hist(ssim_vals, bins=15, alpha=0.7, color=colors['ssim'], edgecolor='black', linewidth=1)
    axes[0, 1].set_title('SSIM Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('SSIM')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(ssim_vals), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(ssim_vals):.4f}')
    axes[0, 1].axvline(np.median(ssim_vals), color='orange', linestyle=':', linewidth=2,
                       label=f'Median: {np.median(ssim_vals):.4f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. LPIPS histogram with statistical info
    axes[0, 2].hist(lpips_vals, bins=15, alpha=0.7, color=colors['lpips'], edgecolor='black', linewidth=1)
    axes[0, 2].set_title('LPIPS Distribution', fontweight='bold')
    axes[0, 2].set_xlabel('LPIPS (lower is better)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].axvline(np.mean(lpips_vals), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(lpips_vals):.4f}')
    axes[0, 2].axvline(np.median(lpips_vals), color='orange', linestyle=':', linewidth=2,
                       label=f'Median: {np.median(lpips_vals):.4f}')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Box plots for comparison
    metrics_data = [psnr_vals, ssim_vals, lpips_vals]
    box_colors = [colors['psnr'], colors['ssim'], colors['lpips']]
    bp = axes[1, 0].boxplot(metrics_data, labels=['PSNR (dB)', 'SSIM', 'LPIPS'], 
                           patch_artist=True, notch=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 0].set_title('Metrics Comparison (Box Plot)', fontweight='bold')
    axes[1, 0].set_ylabel('Metric Values')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Correlation plot PSNR vs SSIM
    axes[1, 1].scatter(psnr_vals, ssim_vals, alpha=0.7, color='purple', s=50, edgecolors='black')
    axes[1, 1].set_title('PSNR vs SSIM Correlation', fontweight='bold')
    axes[1, 1].set_xlabel('PSNR (dB)')
    axes[1, 1].set_ylabel('SSIM')
    
    # Add correlation coefficient
    correlation = np.corrcoef(psnr_vals, ssim_vals)[0, 1]
    axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=axes[1, 1].transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Summary statistics table
    axes[1, 2].axis('off')  # Hide axes
    
    # Create statistics summary
    stats_text = f"""
SUMMARY STATISTICS
{'='*25}

PSNR (Peak Signal-to-Noise Ratio):
   Mean: {np.mean(psnr_vals):.2f} ± {np.std(psnr_vals):.2f} dB
   Range: [{np.min(psnr_vals):.2f}, {np.max(psnr_vals):.2f}] dB
   
SSIM (Structural Similarity Index):
   Mean: {np.mean(ssim_vals):.4f} ± {np.std(ssim_vals):.4f}
   Range: [{np.min(ssim_vals):.4f}, {np.max(ssim_vals):.4f}]
   
LPIPS (Learned Perceptual Similarity):
   Mean: {np.mean(lpips_vals):.4f} ± {np.std(lpips_vals):.4f}
   Range: [{np.min(lpips_vals):.4f}, {np.max(lpips_vals):.4f}]

Total Models: {len(model_names)}
Rendering Angles: {len(results_data['processing_info']['angles'])}
"""
    
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for main title
    
    # Save high-quality image for scientific publication
    img_path = os.path.join(RESULTS_DIR, f"3d_metrics_analysis_{timestamp}.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Also save as PDF for publications
    pdf_path = os.path.join(RESULTS_DIR, f"3d_metrics_analysis_{timestamp}.pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.close()
    
    print(f"✅ Scientific visualization saved to: {img_path}")
    print(f"✅ PDF version saved to: {pdf_path}")
    return img_path, pdf_path

def create_summary_image(results_data, timestamp):
    """Creates an enhanced summary image with key statistics for reports."""
    # Create larger image for better readability
    img_width, img_height = 1000, 800
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_subtitle = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_mono = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    except:
        font_title = ImageFont.load_default()
        font_subtitle = ImageFont.load_default()
        font_text = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_mono = ImageFont.load_default()
    
    # Add border
    draw.rectangle([(10, 10), (img_width-10, img_height-10)], outline='black', width=2)
    
    # Title
    title = "3D RECONSTRUCTION METRICS REPORT"
    draw.text((50, 40), title, fill='black', font=font_title)
    
    # Subtitle
    subtitle = "Quality Assessment Summary"
    draw.text((50, 80), subtitle, fill='gray', font=font_subtitle)
    
    # Date and processing info
    date_str = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    draw.text((50, 120), date_str, fill='gray', font=font_small)
    
    processing_info = f"Models: {len(results_data['individual_results'])} | Angles: {len(results_data['processing_info']['angles'])}"
    draw.text((50, 140), processing_info, fill='gray', font=font_small)
    
    # Main statistics section
    y_pos = 180
    stats = results_data['global_stats']
    
    # Section header
    draw.text((50, y_pos), "QUALITY METRICS SUMMARY", fill='black', font=font_subtitle)
    y_pos += 40
    
    # Create a more structured layout
    # PSNR Section
    draw.rectangle([(70, y_pos), (950, y_pos + 80)], outline='blue', width=2, fill=(240, 248, 255))
    draw.text((90, y_pos + 10), "PSNR (Peak Signal-to-Noise Ratio) - Higher is Better", fill='blue', font=font_text)
    psnr_detail = f"Mean: {stats['psnr']['mean']:.2f} dB  |  Max: {stats['psnr']['max']:.2f} dB  |  Min: {stats['psnr']['min']:.2f} dB  |  Std: {stats['psnr']['std']:.2f}"
    draw.text((90, y_pos + 35), psnr_detail, fill='darkblue', font=font_mono)
    y_pos += 100
    
    # SSIM Section
    draw.rectangle([(70, y_pos), (950, y_pos + 80)], outline='green', width=2, fill=(240, 255, 240))
    draw.text((90, y_pos + 10), "SSIM (Structural Similarity Index) - Higher is Better", fill='green', font=font_text)
    ssim_detail = f"Mean: {stats['ssim']['mean']:.4f}  |  Max: {stats['ssim']['max']:.4f}  |  Min: {stats['ssim']['min']:.4f}  |  Std: {stats['ssim']['std']:.4f}"
    draw.text((90, y_pos + 35), ssim_detail, fill='darkgreen', font=font_mono)
    y_pos += 100
    
    # LPIPS Section
    draw.rectangle([(70, y_pos), (950, y_pos + 80)], outline='red', width=2, fill=(255, 240, 240))
    draw.text((90, y_pos + 10), "LPIPS (Learned Perceptual Similarity) - Lower is Better", fill='red', font=font_text)
    lpips_detail = f"Mean: {stats['lpips']['mean']:.4f}  |  Max: {stats['lpips']['max']:.4f}  |  Min: {stats['lpips']['min']:.4f}  |  Std: {stats['lpips']['std']:.4f}"
    draw.text((90, y_pos + 35), lpips_detail, fill='darkred', font=font_mono)
    y_pos += 120
    
    # Performance interpretation
    draw.text((50, y_pos), "QUALITY INTERPRETATION", fill='black', font=font_subtitle)
    y_pos += 40
    
    # Quality thresholds and interpretation
    interpretation_text = [
        "• PSNR > 30 dB: Excellent quality",
        "• PSNR 25-30 dB: Good quality", 
        "• PSNR 20-25 dB: Fair quality",
        "• PSNR < 20 dB: Poor quality",
        "",
        "• SSIM > 0.9: Excellent structural similarity",
        "• SSIM 0.8-0.9: Good structural similarity",
        "• SSIM < 0.8: Room for improvement",
        "",
        "• LPIPS < 0.1: Excellent perceptual similarity",
        "• LPIPS 0.1-0.3: Good perceptual similarity", 
        "• LPIPS > 0.3: Poor perceptual similarity"
    ]
    
    for line in interpretation_text:
        if line:
            draw.text((70, y_pos), line, fill='black', font=font_small)
        y_pos += 20
    
    # Save image
    summary_img_path = os.path.join(RESULTS_DIR, f"3d_metrics_summary_{timestamp}.png")
    img.save(summary_img_path, quality=95)
    
    # Also save as high-quality PDF
    summary_pdf_path = os.path.join(RESULTS_DIR, f"3d_metrics_summary_{timestamp}.pdf")
    img.save(summary_pdf_path, quality=95)
    
    print(f"✅ Enhanced summary image saved to: {summary_img_path}")
    print(f"✅ Summary PDF saved to: {summary_pdf_path}")
    return summary_img_path, summary_pdf_path

def main():
    """Main pipeline for processing 3D models and calculating metrics."""
    print("🚀 Starting 3D metrics pipeline...")
    
    # Setup
    setup_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get all common file IDs (without extensions)
    file_ids = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(RECONS_DIR)
        if (f.endswith(".glb") or f.endswith(".obj")) and os.path.exists(os.path.join(OG_DIR, f.replace(".glb", ".obj").replace(".obj", ".obj")))
    ])

    # Global accumulators
    global_psnr = []
    global_ssim = []
    global_lpips = []
    individual_results = {}

    for name in tqdm(file_ids, desc="🔄 Processing models"):
        re_path = os.path.join(RECONS_DIR, name + (".glb" if os.path.exists(os.path.join(RECONS_DIR, name + ".glb")) else ".obj"))
        og_path = os.path.join(OG_DIR, name + ".obj")

        # Load meshes
        mesh_re = trimesh.load(re_path, force='mesh')
        mesh_og = trimesh.load(og_path, force='mesh')

        if isinstance(mesh_re, trimesh.Scene):
            mesh_re = trimesh.util.concatenate([g for g in mesh_re.geometry.values()])
        if isinstance(mesh_og, trimesh.Scene):
            mesh_og = trimesh.util.concatenate([g for g in mesh_og.geometry.values()])

        # Apply color
        mesh_re.visual = trimesh.visual.ColorVisuals(mesh_re)
        mesh_re.visual.vertex_colors = np.tile(GRAY_COLOR, (len(mesh_re.vertices), 1))

        mesh_og.visual = trimesh.visual.ColorVisuals(mesh_og)
        mesh_og.visual.vertex_colors = np.tile(GRAY_COLOR, (len(mesh_og.vertices), 1))

        # Normalize
        mesh_re = normalize_and_center_mesh(mesh_re)
        mesh_og = normalize_and_center_mesh(mesh_og)

        # Render OG
        og_img_path = os.path.join(TEMP_DIR, f"{name}_og.png")
        render_mesh(mesh_og, og_img_path, 0)
        og_img = cv2.imread(og_img_path)

        psnr_vals = []
        ssim_vals = []
        lpips_vals = []

        for i, angle in enumerate(ANGLES):
            re_img_path = os.path.join(TEMP_DIR, f"{name}_re_{i}.png")
            render_mesh(mesh_re, re_img_path, angle)
            re_img = cv2.imread(re_img_path)

            # Ensure sizes match
            if re_img.shape != og_img.shape:
                re_img = cv2.resize(re_img, (og_img.shape[1], og_img.shape[0]))
                cv2.imwrite(re_img_path, re_img)  # Save adjusted for LPIPS

            psnr_vals.append(calculate_psnr(og_img, re_img))
            ssim_vals.append(calculate_ssim(og_img, re_img))
            lpips_vals.append(calculate_lpips(og_img_path, re_img_path))

        # Store individual results
        model_psnr = np.mean(psnr_vals)
        model_ssim = np.mean(ssim_vals)
        model_lpips = np.mean(lpips_vals)
        
        individual_results[name] = {
            'psnr': model_psnr,
            'ssim': model_ssim,
            'lpips': model_lpips
        }
        
        global_psnr.append(model_psnr)
        global_ssim.append(model_ssim)
        global_lpips.append(model_lpips)

        clear_temp_directory()

    # Prepare results data
    results_data = {
        'timestamp': timestamp,
        'processing_info': {
            'total_models': len(file_ids),
            'angles': ANGLES,
            'recons_dir': RECONS_DIR,
            'og_dir': OG_DIR
        },
        'global_stats': {
            'psnr': {
                'mean': np.mean(global_psnr),
                'max': np.max(global_psnr),
                'min': np.min(global_psnr),
                'std': np.std(global_psnr)
            },
            'ssim': {
                'mean': np.mean(global_ssim),
                'max': np.max(global_ssim),
                'min': np.min(global_ssim),
                'std': np.std(global_ssim)
            },
            'lpips': {
                'mean': np.mean(global_lpips),
                'max': np.max(global_lpips),
                'min': np.min(global_lpips),
                'std': np.std(global_lpips)
            }
        },
        'individual_results': individual_results
    }

    # Print results to console
    print("\n🎯 GLOBAL METRICS")
    print(f"PSNR  -> Mean: {np.mean(global_psnr):.2f} | Max: {np.max(global_psnr):.2f} | Min: {np.min(global_psnr):.2f}")
    print(f"SSIM  -> Mean: {np.mean(global_ssim):.4f} | Max: {np.max(global_ssim):.4f} | Min: {np.min(global_ssim):.4f}")
    print(f"LPIPS -> Mean: {np.mean(global_lpips):.4f} | Max: {np.max(global_lpips):.4f} | Min: {np.min(global_lpips):.4f}")

    # Save results in text format
    print("\n📄 Saving results...")
    txt_path = save_results_to_txt(results_data, timestamp)
    
    # Create enhanced visualizations for scientific reports
    try:
        print("📊 Creating scientific visualizations...")
        viz_paths = create_results_visualization(results_data, timestamp)
        summary_paths = create_summary_image(results_data, timestamp)
        
        print(f"\n✨ Processing completed!")
        print(f"📁 All files saved to: {RESULTS_DIR}")
        print(f"   - Text results: {os.path.basename(txt_path)}")
        print(f"   - Analysis chart (PNG): {os.path.basename(viz_paths[0])}")
        print(f"   - Analysis chart (PDF): {os.path.basename(viz_paths[1])}")
        print(f"   - Summary image (PNG): {os.path.basename(summary_paths[0])}")
        print(f"   - Summary image (PDF): {os.path.basename(summary_paths[1])}")
        
        print(f"\n📈 Scientific Report Files Ready:")
        print(f"   - Use PDF files for publications and presentations")
        print(f"   - Charts include statistical analysis and correlation data")
        print(f"   - Summary provides quality interpretation guidelines")
        
    except Exception as e:
        print(f"⚠️  Error creating visualizations: {e}")
        print("Text results were saved correctly.")
        import traceback
        print(f"Full error: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
