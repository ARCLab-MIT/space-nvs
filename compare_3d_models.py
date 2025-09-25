import os
import numpy as np
import trimesh
import pyrender
from matplotlib import cm
from PIL import Image
import glob
from pathlib import Path

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

def compare_single_model(obj_path, glb_path, output_path, export_colored_ply=False, ply_output_path="colored_mesh.ply"):
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

def compare_models_batch(original_folder, reconstructed_folder, output_folder, export_colored_ply=False):
    """
    Compares all models in two folders: original .obj models with reconstructed .glb models.
    
    Parameters:
        original_folder (str): Path to the folder containing original .obj models.
        reconstructed_folder (str): Path to the folder containing reconstructed .glb models.
        output_folder (str): Path to the folder where comparison images will be saved.
        export_colored_ply (bool): Whether to export colored .ply models. Default is False.
    
    Returns:
        dict: Dictionary with statistics for each model comparison.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create subfolder for PLY files if needed
    if export_colored_ply:
        ply_folder = os.path.join(output_folder, "colored_ply")
        os.makedirs(ply_folder, exist_ok=True)
    
    # Find all .obj files in the original folder
    obj_files = glob.glob(os.path.join(original_folder, "*.obj"))
    
    if not obj_files:
        print(f"No .obj files found in {original_folder}")
        return {}
    
    results = {}
    processed_count = 0
    skipped_count = 0
    
    print(f"Found {len(obj_files)} .obj files to process...")
    
    for obj_path in obj_files:
        # Get the base name without extension
        base_name = Path(obj_path).stem
        
        # Skip model number 96
        if base_name == "96":
            print(f"Skipping model {base_name} as requested...")
            skipped_count += 1
            continue
        
        # Construct the corresponding .glb path
        glb_path = os.path.join(reconstructed_folder, f"{base_name}.glb")
        
        # Check if the corresponding .glb file exists
        if not os.path.exists(glb_path):
            print(f"Warning: No corresponding .glb file found for {base_name}.obj (expected: {glb_path})")
            skipped_count += 1
            continue
        
        # Define output paths
        img_output_path = os.path.join(output_folder, f"{base_name}_comparison.png")
        ply_output_path = os.path.join(ply_folder, f"{base_name}_colored.ply") if export_colored_ply else None
        
        try:
            print(f"Processing {base_name}...")
            
            # Compare the models
            mean_error, max_error = compare_single_model(
                obj_path, 
                glb_path, 
                img_output_path, 
                export_colored_ply=export_colored_ply, 
                ply_output_path=ply_output_path
            )
            
            # Store results
            results[base_name] = {
                'mean_error': mean_error,
                'max_error': max_error,
                'obj_path': obj_path,
                'glb_path': glb_path,
                'img_output_path': img_output_path,
                'ply_output_path': ply_output_path
            }
            
            processed_count += 1
            print(f"  - Mean error: {mean_error:.4f}")
            print(f"  - Max error: {max_error:.4f}")
            
        except Exception as e:
            print(f"Error processing {base_name}: {str(e)}")
            skipped_count += 1
            continue
    
    # Print summary
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} models")
    print(f"Skipped: {skipped_count} models")
    
    # Print overall statistics
    if results:
        all_mean_errors = [result['mean_error'] for result in results.values()]
        all_max_errors = [result['max_error'] for result in results.values()]
        
        print(f"\nOverall Statistics:")
        print(f"Average mean error: {np.mean(all_mean_errors):.4f}")
        print(f"Average max error: {np.mean(all_max_errors):.4f}")
        print(f"Best mean error: {np.min(all_mean_errors):.4f}")
        print(f"Worst mean error: {np.max(all_mean_errors):.4f}")
        
        # Show the 6 worst models by mean error
        print(f"\n{'='*50}")
        print(f"TOP 6 WORST MODELS (by mean error):")
        print(f"{'='*50}")
        
        # Sort results by mean error (worst first)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_error'], reverse=True)
        
        for i, (model_name, stats) in enumerate(sorted_results[:6]):
            print(f"{i+1:2d}. Model: {model_name}")
            print(f"    Mean Error: {stats['mean_error']:.4f}")
            print(f"    Max Error:  {stats['max_error']:.4f}")
            print(f"    Original:   {os.path.basename(stats['obj_path'])}")
            print(f"    Reconstructed: {os.path.basename(stats['glb_path'])}")
            print()
    
    return results

if __name__ == "__main__":
    # Example usage
    original_folder = "/mnt/data/sdiaz/final_framework/aux/asteroids/og" 
    reconstructed_folder = "/mnt/data/sdiaz/final_framework/aux/asteroids/rec" 
    output_folder = "/mnt/data/sdiaz/final_framework/aux/distance_metrics"  # Output folder for results
    
    # Process all models in the folders
    results = compare_models_batch(
        original_folder, 
        reconstructed_folder, 
        output_folder, 
        export_colored_ply=True
    )
    
    # Save results to a text file
    results_file = os.path.join(output_folder, "comparison_results.txt")
    with open(results_file, 'w') as f:
        f.write("Model Comparison Results\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, stats in results.items():
            f.write(f"Model: {model_name}\n")
            f.write(f"  Mean Error: {stats['mean_error']:.4f}\n")
            f.write(f"  Max Error: {stats['max_error']:.4f}\n")
            f.write(f"  Original: {stats['obj_path']}\n")
            f.write(f"  Reconstructed: {stats['glb_path']}\n")
            f.write(f"  Comparison Image: {stats['img_output_path']}\n")
            if stats['ply_output_path']:
                f.write(f"  Colored PLY: {stats['ply_output_path']}\n")
            f.write("\n")
        
        if results:
            all_mean_errors = [result['mean_error'] for result in results.values()]
            all_max_errors = [result['max_error'] for result in results.values()]
            
            f.write("Overall Statistics:\n")
            f.write(f"  Average Mean Error: {np.mean(all_mean_errors):.4f}\n")
            f.write(f"  Average Max Error: {np.mean(all_max_errors):.4f}\n")
            f.write(f"  Best Mean Error: {np.min(all_mean_errors):.4f}\n")
            f.write(f"  Worst Mean Error: {np.max(all_mean_errors):.4f}\n")
    
    print(f"\nResults saved to: {results_file}")
