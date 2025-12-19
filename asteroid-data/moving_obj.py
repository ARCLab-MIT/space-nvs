import trimesh
import pyrender
import numpy as np
import imageio
import os

# Load your 3D mesh
mesh = trimesh.load("Eros Gaskell 50k poly.obj")
mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=True)

# Create scene and add mesh node
scene = pyrender.Scene()
mesh_node = scene.add(mesh_node)

# Get mesh info
bbox = mesh.bounds
center = mesh.centroid
mesh_size = np.linalg.norm(bbox[1] - bbox[0])
print(f"Mesh size ≈ {mesh_size:.2f}")

# Camera parameters
radius = mesh_size * 2.5
elevation = np.radians(20)
num_frames = 10

# Renderer
renderer = pyrender.OffscreenRenderer(640, 480)
camera = pyrender.PerspectiveCamera(yfov=np.pi / 6.0)

os.makedirs("obj_orbit_frames", exist_ok=True)

# Object motion
translation_speed = np.array([mesh_size * 0.2, mesh_size * 0.1, mesh_size * 0.5])
rotation_speed = np.radians(1)

cam_x = radius
cam_y = 0
cam_z = radius * np.sin(elevation)
cam_pos = np.array([cam_x, cam_y, cam_z]) + center

forward = np.array([-1.0, 0.0, -0.2])
forward /= np.linalg.norm(forward)
right = np.cross(forward, np.array([0, 0, 1]))
right /= np.linalg.norm(right)
up = np.cross(right, forward)

camera_pose = np.eye(4)
camera_pose[:3, :3] = np.vstack([right, up, -forward]).T
camera_pose[:3, 3] = cam_pos

cam_node = scene.add(camera, pose=camera_pose)
light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
light_node = scene.add(light, pose=camera_pose)

for i in range(num_frames):
    # Apply translation + rotation to object
    translation = (i / num_frames) * translation_speed
    angle = i * rotation_speed
    c, s = np.cos(angle), np.sin(angle)
    Rz = np.array([[c, -s, 0],
                   [s,  c, 0],
                   [0,  0, 1]])
    mesh_pose = np.eye(4)
    mesh_pose[:3, :3] = Rz
    mesh_pose[:3, 3] = translation
    scene.set_pose(mesh_node, pose=mesh_pose)

    color, _ = renderer.render(scene)
    imageio.imwrite(f"obj_orbit_frames/frame_{i:03d}.png", color)

renderer.delete()
print("✅ Saved frames with fixed camera and moving object in ./obj_orbit_frames/")
