import trimesh
import numpy as np
from PIL import Image
import pyrender
import os

# Forzar a Pyrender a usar modo headless (EGL) si está disponible
os.environ["PYOPENGL_PLATFORM"] = "egl"

def normalize_and_center_mesh(mesh, target_size=1.0):
    bbox = mesh.bounding_box.extents
    scale = target_size / np.linalg.norm(bbox)
    mesh.apply_scale(scale)
    mesh.apply_translation(-mesh.bounding_box.centroid)
    return mesh




def render_simple_obj_model(obj_path, output_image_path, angle_deg=45):
    # 📤 Carga la malla
    mesh = trimesh.load(obj_path, force='mesh')

    # ✅ Si es una escena, convertir a malla
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

    # 🎨 Aplicar color gris claro
    gray_color = [100, 100, 100, 255]
    mesh.visual = trimesh.visual.ColorVisuals(mesh)
    mesh.visual.vertex_colors = np.tile(gray_color, (len(mesh.vertices), 1))

    # 🔧 Normalizar tamaño y centrar
    mesh = normalize_and_center_mesh(mesh)

    # 🎥 Render (igual que en tus otras funciones)
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

    renderer = pyrender.OffscreenRenderer(640, 480)
    color, _ = renderer.render(scene)

    # 💾 Guardar la imagen
    Image.fromarray(color).save(output_image_path)
    print(f"✅ Imagen renderizada y guardada en: {output_image_path}")


# Uso del ejemplo:
obj_file = "/mnt/data/sdiaz/tencent_model/data/og/002.obj"

# use the .obj name to create the output image path
img_name = os.path.splitext(os.path.basename(obj_file))[0] + ".png"
output_img = "/mnt/data/sdiaz/tencent_model/generalized_pipeline/outputs/views/" + img_name

render_simple_obj_model(obj_file, output_img)