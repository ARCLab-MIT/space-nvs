import trimesh
import numpy as np
from PIL import Image
import pyrender
import os

# Force Pyrender to use headless mode (EGL) if available
os.environ["PYOPENGL_PLATFORM"] = "egl"

def normalize_and_center_mesh(mesh, target_size=1.0):
    bbox = mesh.bounding_box.extents
    scale = target_size / np.linalg.norm(bbox)
    mesh.apply_scale(scale)
    mesh.apply_translation(-mesh.bounding_box.centroid)
    return mesh




def render_simple_obj_model(obj_path, output_image_path, angle_deg=45):
    # Load the mesh
    mesh = trimesh.load(obj_path, force='mesh')

    # Convert scene to mesh if necessary
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

    # Apply light gray color
    gray_color = [100, 100, 100, 255]
    mesh.visual = trimesh.visual.ColorVisuals(mesh)
    mesh.visual.vertex_colors = np.tile(gray_color, (len(mesh.vertices), 1))

    # Normalize size and center
    mesh = normalize_and_center_mesh(mesh)

    # Render the mesh
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

    # Save the image
    Image.fromarray(color).save(output_image_path)
    print(f"Image rendered and saved to: {output_image_path}")


def process_obj_folder(folder_path, output_folder="outputs/views/", angle_deg=45):
    """
    Process all .obj files in a folder and render them as images.
    
    Args:
        folder_path: Path to the folder containing .obj files
        output_folder: Path where rendered images will be saved
        angle_deg: Rotation angle for rendering
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all .obj files in the folder
    obj_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith('.obj'):
            obj_files.append(os.path.join(folder_path, file))
    
    if not obj_files:
        print(f"No .obj files found in {folder_path}")
        return
    
    print(f"Found {len(obj_files)} .obj files to process")
    
    # Process each .obj file
    for obj_file in obj_files:
        try:
            # Create output image path based on .obj filename
            img_name = os.path.splitext(os.path.basename(obj_file))[0] + ".png"
            output_img = os.path.join(output_folder, img_name)
            
            print(f"Processing: {os.path.basename(obj_file)}")
            render_simple_obj_model(obj_file, output_img, angle_deg)
            
        except Exception as e:
            print(f"Error processing {obj_file}: {str(e)}")
            continue
    
    print(f"Processing complete! Images saved to: {output_folder}")


# Example usage:
if __name__ == "__main__":
    # Specify the folder containing .obj files
    obj_folder = "/mnt/data/sdiaz/tencent_model/data/og/all_data"
    
    # Process all .obj files in the folder
    process_obj_folder(obj_folder, "/mnt/data/sdiaz/tencent_model/data/views")
