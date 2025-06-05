import os
import numpy as np
import trimesh
import cv2
import torch
from PIL import Image
from tqdm import tqdm
import lpips
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
import pyrender

# Force Pyrender to use headless mode (EGL) if available
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Configurable parameters
RECONS_DIR = "/mnt/data/sdiaz/tencent_model/data/recons"  # Directory for reconstructed models
OG_DIR = "/mnt/data/sdiaz/tencent_model/data/og"          # Directory for original models
TEMP_DIR = "/mnt/data/sdiaz/tencent_model/generalized_pipeline/temp_imgs"       # Temporary directory for intermediate images, no need to change
ANGLES = list(range(0, 360, 30))  # Angles for rendering
GRAY_COLOR = [100, 100, 100, 255]  # Default color for meshes

# Initialize LPIPS loss function
lpips_loss_fn = lpips.LPIPS(net='vgg')
lpips_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

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

def main():
    """Main pipeline for processing 3D models and calculating metrics."""
    clear_temp_directory()

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

        global_psnr.append(np.mean(psnr_vals))
        global_ssim.append(np.mean(ssim_vals))
        global_lpips.append(np.mean(lpips_vals))

        clear_temp_directory()

    # Global results
    print("\n🎯 GLOBAL METRICS")
    print(f"PSNR  -> Mean: {np.mean(global_psnr):.2f} | Max: {np.max(global_psnr):.2f} | Min: {np.min(global_psnr):.2f}")
    print(f"SSIM  -> Mean: {np.mean(global_ssim):.4f} | Max: {np.max(global_ssim):.4f} | Min: {np.min(global_ssim):.4f}")
    print(f"LPIPS -> Mean: {np.mean(global_lpips):.4f} | Max: {np.max(global_lpips):.4f} | Min: {np.min(global_lpips):.4f}")

if __name__ == "__main__":
    main()
