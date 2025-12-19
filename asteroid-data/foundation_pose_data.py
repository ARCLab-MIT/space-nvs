"""
Generates ground truth data for .obj file in different poses

Input: .obj file, set a camera trajectory
Output: rgb, depth, mask images; meta files
"""

import trimesh
import numpy as np
import os, json, math
import pyrender
from PIL import Image


def convert_to_m():
    """
    Eros Gaskell 50k poly.obj in km I believe
    """
    mesh = trimesh.load("Eros Gaskell 50k poly.obj", force="mesh")
    mesh.apply_scale(1000.0)
    extents = mesh.extents
    mesh.export("Eros_m.obj")


OBJ_PATH = "Eros Gaskell 50k poly.obj"
OUT_DIR = "data/my_object"
N_VIEWS = 24
IMG_WH = (640, 480)
FOV_Y_DEG = 20.0

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "images", "rgb"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "images", "depth"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "images", "mask"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "meta"), exist_ok=True)

print("Loading mesh: ", OBJ_PATH)
mesh = trimesh.load(OBJ_PATH, force="mesh")
if mesh.is_empty:
    raise RuntimeError("Loaded mesh is empty")

print("Original mesh extents: ", mesh.extents)
mesh.apply_translation(-mesh.centroid) # center the mesh

material = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[0.8, 0.8, 0.8, 1.0],
    metallicFactor=0.0,
    roughnessFactor=0.4
)

mesh_tr = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
scene = pyrender.Scene(bg_color=[0.2, 0.2, 0.2, 1.0], ambient_light=[.05, .05, .05])
obj_node = scene.add(mesh_tr, name="object")
r = pyrender.OffscreenRenderer(viewport_width=IMG_WH[0], viewport_height=IMG_WH[1])

fx = fy = (IMG_WH[1] / 2) / math.tan(math.radians(FOV_Y_DEG) / 2)
cx = IMG_WH[0] / 2
cy = IMG_WH[1] / 2
K = [[fx, 0, cx],
     [0, fy, cy],
     [0, 0,  1]]

print(f"Camera intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx}, cy={cy}")

max_dim = float(mesh.extents.max())
radius = max_dim * 2.5

def make_camera():
    znear = max(0.001, max_dim * 0.001)
    zfar = max(10.0, radius * 5.0)
    cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=znear, zfar=zfar)
    return cam

def build_camera_pose(cam_pos, center=np.array([0.0, 0.0, 0.0])):
    forward = (center - cam_pos).astype(np.float64)
    norm = np.linalg.norm(forward)
    if norm == 0:
        raise RuntimeError("camera equals center")
    forward /= norm

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 1.0, 1.0])
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    R = np.column_stack((right, up, -forward))
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = cam_pos
    return pose

def render_view(cam_pos, idx):
    camera = make_camera()
    cam_pose = build_camera_pose(cam_pos)
    cam_node = scene.add(camera, pose=cam_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light_node = scene.add(light, pose=cam_pose)

    color, depth = r.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
    mask = (depth > 0).astype(np.uint16) * 255

    scene.remove_node(cam_node)
    scene.remove_node(light_node)

    rgb_p = os.path.join(OUT_DIR, "images", "rgb", f"img_{idx:06d}.png")
    Image.fromarray(color).save(rgb_p)

    depth_p = os.path.join(OUT_DIR, "images", "depth", f"img_{idx:06d}.png")
    depth_clean = np.nan_to_num(depth, nan=0.0, posinf=0.0)
    # i = np.where(depth_clean > 0)
    # print(depth_clean[i])
    depth_m_int = np.round(depth_clean).astype(np.uint8) # note uint8 may not work for m instead of km
    rgb_image = np.stack([depth_m_int]*3, axis=-1)
    Image.fromarray(rgb_image,  mode='RGB').save(depth_p)


    mask_p = os.path.join(OUT_DIR, "images", "mask", f"img_{idx:06d}.png")
    Image.fromarray(mask).convert("L").save(mask_p)

    cam_to_world = cam_pose
    world_to_cam = np.linalg.inv(cam_to_world)
    pose_gt = world_to_cam.tolist()

    meta = {
        "image_filename": os.path.relpath(rgb_p, OUT_DIR),
        "depth_filename": os.path.relpath(depth_p, OUT_DIR),
        "mask_filename": os.path.relpath(mask_p, OUT_DIR),
        "K": K,
        "pose_gt": pose_gt,
        "camera_pose_world": cam_to_world.tolist()
    }
    meta_p = os.path.join(OUT_DIR, "meta", f"img_{idx:06d}.json")
    with open(meta_p, "w") as f:
        json.dump(meta, f, indent=2)

def generate_data():
    for i in range(N_VIEWS):
        theta = (2 * math.pi * i) / N_VIEWS
        elev_deg = 20 + ((i % 6) * 8)
        phi = math.radians(elev_deg)
        x = radius * math.cos(theta) * math.cos(phi)
        y = radius * math.sin(theta) * math.cos(phi)
        z = radius * math.sin(phi)
        cam_pos = np.array([x, y, z])
        render_view(cam_pos, i)

    first_rgb = os.path.join(OUT_DIR, "images", "rgb", "img_000000.png")
    first_mask = os.path.join(OUT_DIR, "images", "mask", "img_000000.png")
    if os.path.exists(first_rgb) and os.path.exists(first_mask):
        img = np.array(Image.open(first_rgb))
        mask = np.array(Image.open(first_mask).convert("L"))
        print("first RGB mean pixel:", img.mean(), "mask nonzero:", int(np.count_nonzero(mask)))
    else:
        print("first frame not found; check renderer logs.")

    print("Done. Data written to:", OUT_DIR)


def look_numpy(filename):
    data = np.load(filename)
    print(data)

if __name__ == "__main__":
    convert_to_m()
    generate_data()
