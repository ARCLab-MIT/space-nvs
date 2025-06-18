import trimesh
import open3d as o3d
import numpy as np

def load_mesh_to_pointcloud(file_path, n_points=10000):
    mesh = trimesh.load(file_path, force='mesh')
    mesh.apply_scale(1 / mesh.scale)  # normalize scale
    sampled_points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return np.array(sampled_points)


def align_point_clouds(source_points, target_points):
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



from scipy.spatial import cKDTree, distance

def chamfer_distance(pcd1, pcd2):
    tree1 = cKDTree(pcd1)
    tree2 = cKDTree(pcd2)
    d1, _ = tree1.query(pcd2)
    d2, _ = tree2.query(pcd1)
    return np.mean(d1**2) + np.mean(d2**2)

def hausdorff_distance(pcd1, pcd2):
    forward = distance.directed_hausdorff(pcd1, pcd2)[0]
    backward = distance.directed_hausdorff(pcd2, pcd1)[0]
    return max(forward, backward)




pcd1 = load_mesh_to_pointcloud("/mnt/data/sdiaz/tencent_model/data/og/002.obj")
pcd2 = load_mesh_to_pointcloud("/mnt/data/sdiaz/tencent_model/data/recons/002.glb")

aligned_pcd2, aligned_pcd1 = align_point_clouds(pcd2, pcd1)

cd = chamfer_distance(aligned_pcd1, aligned_pcd2)
hd = hausdorff_distance(aligned_pcd1, aligned_pcd2)

print(f"Chamfer Distance: {cd:.6f}")
print(f"Hausdorff Distance: {hd:.6f}")










import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

def colorize_by_distance(source_points, target_points):
    # Calculate distance from reconstructed point to nearest original point
    tree = cKDTree(target_points)
    distances, _ = tree.query(source_points)
    return distances

def save_colored_pointcloud_image(points, distances, output_path="outputs/comparison/colored_view.png"):
    # Normalize distances
    norm_distances = (distances - np.min(distances)) / (np.ptp(distances) + 1e-8)
    colors = cm.viridis(norm_distances)[:, :3]  # RGB (without alpha)

    # 2D projection (XY axis, can be changed to XZ or YZ)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(points[:, 0], points[:, 1], c=colors, s=1)
    ax.set_title("Distance per point to original model")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Image saved to: {os.path.abspath(output_path)}")

# Calculate distances and save image
distances = colorize_by_distance(aligned_pcd2, aligned_pcd1)
save_colored_pointcloud_image(aligned_pcd2, distances)




def save_colored_pointcloud_ply(points, distances, output_path="outputs/comparison/colored_model.ply"):
    import matplotlib.cm as cm
    import matplotlib.colors as colors

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
    print(f"Colored 3D model saved to: {os.path.abspath(output_path)}")

# Calculate distances and save 3D model
distances = colorize_by_distance(aligned_pcd2, aligned_pcd1)
save_colored_pointcloud_ply(aligned_pcd2, distances)
