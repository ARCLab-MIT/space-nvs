import os
import numpy as np
import trimesh
import pyrender
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree, distance
import glob
from pathlib import Path

# Set OpenGL platform for headless rendering
os.environ["PYOPENGL_PLATFORM"] = "egl"

class UnifiedMetrics:
    def __init__(self, original_dir, reconstruction_dir, output_dir="outputs"):
        self.original_dir = original_dir
        self.reconstruction_dir = reconstruction_dir
        self.output_dir = output_dir
        self.ensure_output_dirs()
        
    def ensure_output_dirs(self):
        """Create output directories if they don't exist"""
        dirs = [
            f"{self.output_dir}/iou_overlays",
            f"{self.output_dir}/distance_visualizations",
            f"{self.output_dir}/point_clouds"
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def load_and_align_pca(self, model_path):
        """Load mesh and align using PCA (from iou_volumetric_fast.py)"""
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

        # Simplify mesh if too complex
        if len(mesh.faces) > 10000:
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=10000)
            except (ImportError, AttributeError):
                print(f"Quadric simplification not available for {model_path}")
        
        return mesh

    def calculate_iou(self, mesh1, mesh2, num_points=100000):
        """Calculate volumetric IoU fast"""
        bbox_min = np.minimum(mesh1.bounds[0], mesh2.bounds[0])
        bbox_max = np.maximum(mesh1.bounds[1], mesh2.bounds[1])
        points = np.random.uniform(bbox_min, bbox_max, (num_points, 3))

        inside1 = mesh1.contains(points)
        inside2 = mesh2.contains(points)

        intersection = np.sum(np.logical_and(inside1, inside2))
        union = np.sum(np.logical_or(inside1, inside2))

        return intersection / union if union > 0 else 0

    def render_models_overlay(self, mesh1, mesh2, output_path, title="Model Overlay"):
        """Render two models overlaid fast"""
        scene = pyrender.Scene()

        # Red for original, blue for reconstruction
        mat1 = pyrender.MetallicRoughnessMaterial(baseColorFactor=[1, 0, 0, 0.5])
        mat2 = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0, 0, 1, 0.5])

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
        print(f"Overlay saved to: {output_path}")

    def load_mesh_to_pointcloud(self, file_path, n_points=10000):
        """Load mesh and convert to point cloud (from distance_metrics.py)"""
        mesh = trimesh.load(file_path, force='mesh')
        mesh.apply_scale(1 / mesh.scale)  # normalize scale
        sampled_points, _ = trimesh.sample.sample_surface(mesh, n_points)
        return np.array(sampled_points)

    def align_point_clouds(self, source_points, target_points):
        """Align point clouds using ICP (from distance_metrics.py)"""
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_points)
        target.points = o3d.utility.Vector3dVector(target_points)

        threshold = 0.02  # maximum distance for correspondence
        trans_init = np.eye(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        
        source.transform(reg_p2p.transformation)
        return np.asarray(source.points), np.asarray(target.points)

    def chamfer_distance(self, pcd1, pcd2):
        """Calculate Chamfer distance (from distance_metrics.py)"""
        tree1 = cKDTree(pcd1)
        tree2 = cKDTree(pcd2)
        d1, _ = tree1.query(pcd2)
        d2, _ = tree2.query(pcd1)
        return np.mean(d1**2) + np.mean(d2**2)

    def hausdorff_distance(self, pcd1, pcd2):
        """Calculate Hausdorff distance (from distance_metrics.py)"""
        forward = distance.directed_hausdorff(pcd1, pcd2)[0]
        backward = distance.directed_hausdorff(pcd2, pcd1)[0]
        return max(forward, backward)

    def colorize_by_distance(self, source_points, target_points):
        """Calculate distance from each point to nearest target point"""
        tree = cKDTree(target_points)
        distances, _ = tree.query(source_points)
        return distances

    def save_colored_pointcloud_image(self, points, distances, output_path, title="Distance Visualization"):
        """Save 2D projection of colored point cloud"""
        # Normalize distances
        norm_distances = (distances - np.min(distances)) / (np.ptp(distances) + 1e-8)
        colors = cm.viridis(norm_distances)[:, :3]  # RGB (without alpha)

        # 2D projection (XY axis)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(points[:, 0], points[:, 1], c=colors, s=1)
        ax.set_title(title)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Distance visualization saved to: {output_path}")

    def save_colored_pointcloud_ply(self, points, distances, output_path):
        """Save colored 3D point cloud as PLY file"""
        # Normalize distances
        norm_distances = (distances - np.min(distances)) / (np.ptp(distances) + 1e-8)

        # Convert to RGB colors using 'viridis'
        colormap = cm.get_cmap('viridis')
        colors_rgb = colormap(norm_distances)[:, :3]  # remove alpha channel
        colors_rgb = (colors_rgb * 255).astype(np.uint8)

        # Create PointCloud in Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb / 255.0)

        # Save as .ply file
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Colored 3D model saved to: {output_path}")

    def process_single_pair(self, original_path, reconstruction_path, model_name):
        """Process a single pair of models and calculate all metrics"""
        print(f"\nProcessing {model_name}...")
        
        # Load meshes for IoU calculation
        try:
            mesh1 = self.load_and_align_pca(original_path)
            mesh2 = self.load_and_align_pca(reconstruction_path)
            iou = self.calculate_iou(mesh1, mesh2, 100000)
            print(f"IoU for {model_name}: {iou:.4f}")
        except Exception as e:
            print(f"Error calculating IoU for {model_name}: {e}")
            iou = 0.0
            mesh1, mesh2 = None, None

        # Load point clouds for distance metrics
        try:
            pcd1 = self.load_mesh_to_pointcloud(original_path)
            pcd2 = self.load_mesh_to_pointcloud(reconstruction_path)
            aligned_pcd2, aligned_pcd1 = self.align_point_clouds(pcd2, pcd1)
            
            cd = self.chamfer_distance(aligned_pcd1, aligned_pcd2)
            hd = self.hausdorff_distance(aligned_pcd1, aligned_pcd2)
            
            print(f"Chamfer Distance for {model_name}: {cd:.6f}")
            print(f"Hausdorff Distance for {model_name}: {hd:.6f}")
            
        except Exception as e:
            print(f"Error calculating distance metrics for {model_name}: {e}")
            cd, hd = float('inf'), float('inf')
            aligned_pcd1, aligned_pcd2 = None, None

        return {
            'model_name': model_name,
            'iou': iou,
            'chamfer_distance': cd,
            'hausdorff_distance': hd,
            'original_path': original_path,
            'reconstruction_path': reconstruction_path,
            'meshes': (mesh1, mesh2),
            'point_clouds': (aligned_pcd1, aligned_pcd2)
        }

    def find_model_pairs(self):
        """Find matching pairs of original and reconstruction models"""
        original_files = glob.glob(f"{self.original_dir}/*.obj") + glob.glob(f"{self.original_dir}/*.glb")
        reconstruction_files = glob.glob(f"{self.reconstruction_dir}/*.obj") + glob.glob(f"{self.reconstruction_dir}/*.glb")
        
        pairs = []
        for orig_file in original_files:
            orig_name = Path(orig_file).stem
            for recon_file in reconstruction_files:
                recon_name = Path(recon_file).stem
                if orig_name == recon_name:
                    pairs.append((orig_file, recon_file, orig_name))
                    break
        
        return pairs

    def run_analysis(self):
        """Run complete analysis on all model pairs"""
        pairs = self.find_model_pairs()
        if not pairs:
            print("No matching pairs found!")
            return
        
        print(f"Found {len(pairs)} model pairs to analyze")
        
        counter = 0
        results = []
        for original_path, reconstruction_path, model_name in pairs:
            counter += 1
            print(f"\nProcessing pair {counter}/{len(pairs)}: {model_name}")
            # saltar si es el modelo nº 61, 61.obj o 61.glb
            if model_name == "61.obj" or model_name == "61.glb" or model_name == "61":
                print(f"Skipping model {model_name}")
                continue
            result = self.process_single_pair(original_path, reconstruction_path, model_name)
            results.append(result)
        
        # Find best and worst cases
        valid_results = [r for r in results if r['iou'] > 0 and r['chamfer_distance'] != float('inf')]
        
        if not valid_results:
            print("No valid results found!")
            return
        
        # Best and worst by IoU
        best_iou = max(valid_results, key=lambda x: x['iou'])
        worst_iou = min(valid_results, key=lambda x: x['iou'])
        
        # Best and worst by Chamfer distance (lower is better)
        best_chamfer = min(valid_results, key=lambda x: x['chamfer_distance'])
        worst_chamfer = max(valid_results, key=lambda x: x['chamfer_distance'])
        
        print(f"\n=== RESULTS SUMMARY ===")
        print(f"Best IoU: {best_iou['model_name']} ({best_iou['iou']:.4f})")
        print(f"Worst IoU: {worst_iou['model_name']} ({worst_iou['iou']:.4f})")
        print(f"Best Chamfer: {best_chamfer['model_name']} ({best_chamfer['chamfer_distance']:.6f})")
        print(f"Worst Chamfer: {worst_chamfer['model_name']} ({worst_chamfer['chamfer_distance']:.6f})")
        
        # Generate visualizations for best and worst cases
        self.generate_visualizations(best_iou, worst_iou, best_chamfer, worst_chamfer)
        
        # Save results to file
        self.save_results(results)

    def generate_visualizations(self, best_iou, worst_iou, best_chamfer, worst_chamfer):
        """Generate visualizations for best and worst cases"""
        
        # IoU overlays
        if best_iou['meshes'][0] is not None and best_iou['meshes'][1] is not None:
            self.render_models_overlay(
                best_iou['meshes'][0], 
                best_iou['meshes'][1],
                f"{self.output_dir}/iou_overlays/best_iou_{best_iou['model_name']}.png",
                f"Best IoU: {best_iou['model_name']} ({best_iou['iou']:.4f})"
            )
        
        if worst_iou['meshes'][0] is not None and worst_iou['meshes'][1] is not None:
            self.render_models_overlay(
                worst_iou['meshes'][0], 
                worst_iou['meshes'][1],
                f"{self.output_dir}/iou_overlays/worst_iou_{worst_iou['model_name']}.png",
                f"Worst IoU: {worst_iou['model_name']} ({worst_iou['iou']:.4f})"
            )
        
        # Distance visualizations
        if best_chamfer['point_clouds'][0] is not None and best_chamfer['point_clouds'][1] is not None:
            distances = self.colorize_by_distance(best_chamfer['point_clouds'][1], best_chamfer['point_clouds'][0])
            
            self.save_colored_pointcloud_image(
                best_chamfer['point_clouds'][1], 
                distances,
                f"{self.output_dir}/distance_visualizations/best_chamfer_{best_chamfer['model_name']}.png",
                f"Best Chamfer: {best_chamfer['model_name']} ({best_chamfer['chamfer_distance']:.6f})"
            )
            
            self.save_colored_pointcloud_ply(
                best_chamfer['point_clouds'][1], 
                distances,
                f"{self.output_dir}/point_clouds/best_chamfer_{best_chamfer['model_name']}.ply"
            )
        
        if worst_chamfer['point_clouds'][0] is not None and worst_chamfer['point_clouds'][1] is not None:
            distances = self.colorize_by_distance(worst_chamfer['point_clouds'][1], worst_chamfer['point_clouds'][0])
            
            self.save_colored_pointcloud_image(
                worst_chamfer['point_clouds'][1], 
                distances,
                f"{self.output_dir}/distance_visualizations/worst_chamfer_{worst_chamfer['model_name']}.png",
                f"Worst Chamfer: {worst_chamfer['model_name']} ({worst_chamfer['chamfer_distance']:.6f})"
            )
            
            self.save_colored_pointcloud_ply(
                worst_chamfer['point_clouds'][1], 
                distances,
                f"{self.output_dir}/point_clouds/worst_chamfer_{worst_chamfer['model_name']}.ply"
            )

    def save_results(self, results):
        """Save results to a text file"""
        with open(f"{self.output_dir}/metrics_results.txt", "w") as f:
            f.write("MODEL METRICS ANALYSIS RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            # Calculate statistics for valid results
            valid_results = [r for r in results if r['iou'] > 0 and r['chamfer_distance'] != float('inf')]
            
            if valid_results:
                # Extract metrics
                ious = [r['iou'] for r in valid_results]
                chamfer_dists = [r['chamfer_distance'] for r in valid_results]
                hausdorff_dists = [r['hausdorff_distance'] for r in valid_results if r['hausdorff_distance'] != float('inf')]
                
                # Calculate statistics
                f.write("STATISTICAL SUMMARY\n")
                f.write("=" * 30 + "\n\n")
                
                f.write("IoU Statistics:\n")
                f.write(f"  Mean: {np.mean(ious):.4f}\n")
                f.write(f"  Median: {np.median(ious):.4f}\n")
                f.write(f"  Standard Deviation: {np.std(ious):.4f}\n")
                f.write(f"  Min: {np.min(ious):.4f}\n")
                f.write(f"  Max: {np.max(ious):.4f}\n")
                f.write(f"  Valid samples: {len(ious)}\n\n")
                
                f.write("Chamfer Distance Statistics:\n")
                f.write(f"  Mean: {np.mean(chamfer_dists):.6f}\n")
                f.write(f"  Median: {np.median(chamfer_dists):.6f}\n")
                f.write(f"  Standard Deviation: {np.std(chamfer_dists):.6f}\n")
                f.write(f"  Min: {np.min(chamfer_dists):.6f}\n")
                f.write(f"  Max: {np.max(chamfer_dists):.6f}\n")
                f.write(f"  Valid samples: {len(chamfer_dists)}\n\n")
                
                if hausdorff_dists:
                    f.write("Hausdorff Distance Statistics:\n")
                    f.write(f"  Mean: {np.mean(hausdorff_dists):.6f}\n")
                    f.write(f"  Median: {np.median(hausdorff_dists):.6f}\n")
                    f.write(f"  Standard Deviation: {np.std(hausdorff_dists):.6f}\n")
                    f.write(f"  Min: {np.min(hausdorff_dists):.6f}\n")
                    f.write(f"  Max: {np.max(hausdorff_dists):.6f}\n")
                    f.write(f"  Valid samples: {len(hausdorff_dists)}\n\n")
                
                f.write("=" * 50 + "\n\n")
            
            # Individual results
            f.write("INDIVIDUAL RESULTS\n")
            f.write("=" * 30 + "\n\n")
            
            for result in results:
                f.write(f"Model: {result['model_name']}\n")
                f.write(f"  IoU: {result['iou']:.4f}\n")
                f.write(f"  Chamfer Distance: {result['chamfer_distance']:.6f}\n")
                f.write(f"  Hausdorff Distance: {result['hausdorff_distance']:.6f}\n")
                f.write(f"  Original: {result['original_path']}\n")
                f.write(f"  Reconstruction: {result['reconstruction_path']}\n\n")
        
        print(f"Results saved to: {self.output_dir}/metrics_results.txt")


def main():
    """Main function to run the analysis"""
    # Configure paths - CHANGE THESE TO YOUR DIRECTORIES
    original_dir = "/mnt/data/sdiaz/final_framework/aux/asteroids/og"
    reconstruction_dir = "/mnt/data/sdiaz/final_framework/aux/asteroids/rec"
    output_dir = "/mnt/data/sdiaz/final_framework/aux/3d_metrics"
    
    # Create and run analysis
    analyzer = UnifiedMetrics(original_dir, reconstruction_dir, output_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
