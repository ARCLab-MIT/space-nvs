import trimesh
import pyrender
import numpy as np
import imageio
import os

# Load your 3D mesh
mesh = trimesh.load("Eros Gaskell 50k poly.obj")
scene = pyrender.Scene()
scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))

# Get mesh bounding box and compute scale
bbox = mesh.bounds
center = mesh.centroid
mesh_size = np.linalg.norm(bbox[1] - bbox[0])
print(f"Mesh size ≈ {mesh_size:.2f}")

# Camera parameters
radius = mesh_size * 2.5   # orbit radius proportional to mesh size
elevation = np.radians(20)
num_frames = 60

# Set up renderer
renderer = pyrender.OffscreenRenderer(640, 480)
camera = pyrender.PerspectiveCamera(yfov=np.pi / 6.0)  # smaller FOV = zoomed in, try np.pi/6 to np.pi/3

os.makedirs("orbit_frames", exist_ok=True)

for i, theta in enumerate(np.linspace(0, 2 * np.pi, num_frames)):
    # Orbit position
    cam_x = radius * np.cos(theta)
    cam_y = radius * np.sin(theta)
    cam_z = radius * np.sin(elevation)
    cam_pos = np.array([cam_x, cam_y, cam_z]) + center

    # Build camera pose (look at mesh center)
    forward = (center - cam_pos)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0, 0, 1]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, :3] = np.vstack([right, up, -forward]).T
    pose[:3, 3] = cam_pos

    # Add camera and light
    cam_node = scene.add(camera, pose=pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light_node = scene.add(light, pose=pose)

    # Render
    color, _ = renderer.render(scene)
    imageio.imwrite(f"orbit_frames/frame_{i:03d}.png", color)

    # Cleanup
    scene.remove_node(cam_node)
    scene.remove_node(light_node)

    if i == 10:
        break

renderer.delete()
print("✅ Saved frames in ./orbit_frames/")
