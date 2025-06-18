import os
import numpy as np
import trimesh
import pyrender
from PIL import Image
from sklearn.decomposition import PCA

os.environ["PYOPENGL_PLATFORM"] = "egl"

def load_and_align_pca(model_path):
    mesh = trimesh.load(model_path, force='mesh')
    
    # Center the mesh
    mesh.apply_translation(-mesh.centroid)

    # Perform PCA to align the model
    pca = PCA(n_components=3)
    pca.fit(mesh.vertices)

    # Get rotation matrix to align with principal axes
    R = pca.components_.T
    
    # Apply inverse rotation to align with global axes
    mesh.vertices = mesh.vertices.dot(R)

    # Scale to standard size
    scale_factor = 1.0 / mesh.scale
    mesh.apply_scale(scale_factor)

    # ALTERNATIVE: Use simplification methods included in trimesh
    if len(mesh.faces) > 10000:
        try:
            # Attempt quadric simplification if available
            mesh = mesh.simplify_quadric_decimation(face_count=10000)
        except (ImportError, AttributeError):
            # If it doesn't work, use convex hull as alternative
            print(f"Quadric simplification not available")
    
    
    return mesh

def calculate_iou(mesh1, mesh2, num_points=100000):
    bbox_min = np.minimum(mesh1.bounds[0], mesh2.bounds[0])
    bbox_max = np.maximum(mesh1.bounds[1], mesh2.bounds[1])
    points = np.random.uniform(bbox_min, bbox_max, (num_points, 3))
    print(f"Generated {num_points} random points in the union space of the models...")

    inside1 = mesh1.contains(points)
    print(f"Point verification inside first model: {np.sum(inside1)} points inside.")
    inside2 = mesh2.contains(points)
    print(f"Point verification inside second model: {np.sum(inside2)} points inside.")

    intersection = np.sum(np.logical_and(inside1, inside2))
    union = np.sum(np.logical_or(inside1, inside2))
    print(f"Intersection points: {intersection}, Union points: {union}")

    return intersection / union if union > 0 else 0

def render_models(mesh1, mesh2, output_path="overlay.png"):
    scene = pyrender.Scene()

    mat1 = pyrender.MetallicRoughnessMaterial(baseColorFactor=[1,0,0,0.5])
    mat2 = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0,0,1,0.5])

    mesh1_pyrender = pyrender.Mesh.from_trimesh(mesh1, material=mat1)
    mesh2_pyrender = pyrender.Mesh.from_trimesh(mesh2, material=mat2)

    scene.add(mesh1_pyrender)
    scene.add(mesh2_pyrender)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(800, 600)
    color, _ = r.render(scene)
    r.delete()

    Image.fromarray(color).save(output_path)

# Paths to the models CHANGE THESE TO YOUR OWN FILES
model1_path = '/mnt/data/sdiaz/tencent_model/data/og/002.obj'
model2_path = '/mnt/data/sdiaz/tencent_model/data/recons/002.glb'

mesh1 = load_and_align_pca(model1_path)
mesh2 = load_and_align_pca(model2_path)

print("PCA alignment completed.")

iou = calculate_iou(mesh1, mesh2, 100000)
print(f"Volumetric IoU after PCA alignment: {iou:.4f}")

# get the name of the model without extension
model1_name = os.path.splitext(os.path.basename(model1_path))[0]
model2_name = os.path.splitext(os.path.basename(model2_path))[0]
render_models(mesh1, mesh2, f"outputs/iou_imgs/{model1_name}_vs_{model2_name}.png")
