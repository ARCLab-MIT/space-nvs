import os
import numpy as np
import trimesh
import pyrender
from matplotlib import cm
from PIL import Image



import os
# Force Pyrender to use headless mode (EGL) if available
os.environ["PYOPENGL_PLATFORM"] = "egl"




def normalize_and_center_mesh(mesh, target_size=1.0):
    """Normalizes and centers a mesh to a target size."""
    bbox = mesh.bounding_box.extents
    scale = target_size / np.linalg.norm(bbox)
    mesh.apply_scale(scale)
    mesh.apply_translation(-mesh.bounding_box.centroid)
    return mesh

def compute_vertex_distances(original_mesh, reconstructed_mesh):
    """
    Calculates the distances between the vertices of the original and reconstructed models.
    """
    kdtree = trimesh.proximity.ProximityQuery(original_mesh)
    distances, _ = kdtree.vertex(reconstructed_mesh.vertices)
    return distances

def apply_error_colors(mesh, distances):
    """
    Applies colors based on distances (heatmap).
    """
    normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())
    colors = cm.jet(normalized_distances)[:, :3] * 255

    colored_mesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_colors=colors.astype(np.uint8)
    )
    return colored_mesh

def compare_models(obj_path, glb_path, output_path, export_colored_ply=False, ply_output_path="colored_mesh.ply"):
    """
    Compares an original .obj model with a reconstructed .glb model.

    Parameters:
        obj_path (str): Path to the original .obj model.
        glb_path (str): Path to the reconstructed .glb model.
        output_path (str): Path to save the output comparison image.
        export_colored_ply (bool): Whether to export the colored .ply model. Default is False.
        ply_output_path (str): Path to save the colored .ply model if export_colored_ply is True. Default is "colored_mesh.ply".
    """
    # Load original model (.obj)
    original_mesh = trimesh.load(obj_path)

    # Load reconstructed model (.glb)
    scene = trimesh.load(glb_path, force='scene')

    # Extract the mesh from the GLB
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform = scene.graph.get(node)[0]
        mesh = scene.geometry[node]
        if isinstance(mesh, trimesh.Trimesh):
            mesh.apply_transform(transform)
            meshes.append(mesh)

    reconstructed_mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]

    # Normalize and center both models
    original_mesh = normalize_and_center_mesh(original_mesh)
    reconstructed_mesh = normalize_and_center_mesh(reconstructed_mesh)

    # Calculate distances
    distances = compute_vertex_distances(original_mesh, reconstructed_mesh)

    # Apply error colors
    colored_mesh = apply_error_colors(reconstructed_mesh, distances)

    # Export the colored mesh if the option is enabled
    if export_colored_ply:
        colored_mesh.export(ply_output_path)

    # Render the colored mesh
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0])

    # Rotate for better visualization
    angle = np.radians(45)
    rotation_y = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
    colored_mesh.apply_transform(rotation_y)

    # Add to the scene
    scene.add(pyrender.Mesh.from_trimesh(colored_mesh, smooth=False))

    # Configure camera and light
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Render
    r = pyrender.OffscreenRenderer(400, 400)
    color, _ = r.render(scene)

    # Save the image
    image = Image.fromarray(color)
    image.save(output_path)

    # Calculate error statistics
    mean_error = np.mean(distances)
    max_error = np.max(distances)

    return mean_error, max_error

if __name__ == "__main__":
    # Example usage
    obj_path = "/mnt/data/sdiaz/tencent_model/data/og/001.obj"  # Replace with your .obj file path
    glb_path = "/mnt/data/sdiaz/tencent_model/data/recons/001.glb"  # Replace with your .glb file path
    img_output_path = "outputs/comparison/comparison.png"
    ply_output_path = "outputs/comparison/colored_mesh_output.ply"

    mean_error, max_error = compare_models(obj_path, glb_path, img_output_path, export_colored_ply=True, ply_output_path=ply_output_path)
    print(f"Mean error: {mean_error:.4f}")
    print(f"Max error: {max_error:.4f}")
