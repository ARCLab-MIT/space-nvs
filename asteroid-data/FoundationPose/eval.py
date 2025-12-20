import numpy as np
import json
from scipy.spatial.transform import Rotation as R
import os

def load_predicted_pose(txt_file):
    """Load a single predicted 4x4 pose from txt file."""
    matrix = []
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = list(map(float, line.split()))
            matrix.append(row)
    if len(matrix) != 4 or any(len(r) != 4 for r in matrix):
        raise ValueError(f"Invalid pose shape in {txt_file}")
    return np.array(matrix)

def load_gt_pose(json_file):
    """Load ground truth 4x4 camera_pose_world from JSON."""
    with open(json_file, 'r') as f:
        data = json.load(f)
        pose = np.array(data['camera_pose_world'])
    return pose

def pose_errors(pred_pose, gt_pose):
    """Compute translation and rotation errors between two 4x4 poses."""
    t_pred = pred_pose[:3, 3]
    t_gt = gt_pose[:3, 3]
    t_error = np.linalg.norm(t_pred - t_gt)

    R_pred = pred_pose[:3, :3]
    R_gt = gt_pose[:3, :3]
    R_diff = R.from_matrix(R_pred.T @ R_gt)
    r_error = R_diff.magnitude() * (180/np.pi)  # degrees

    return t_error, r_error

def evaluate(pred_dir, gt_dir):
    t_errors = []
    r_errors = []

    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.txt')]
    pred_files.sort()

    for pf in pred_files:
        base_name = os.path.splitext(pf)[0]
        pred_path = os.path.join(pred_dir, pf)
        gt_path = os.path.join(gt_dir, base_name + '.json')

        if not os.path.exists(gt_path):
            print(f"Warning: GT file {gt_path} not found. Skipping.")
            continue

        pred_pose = load_predicted_pose(pred_path)
        gt_pose = load_gt_pose(gt_path)

        t_err, r_err = pose_errors(pred_pose, gt_pose)
        t_errors.append(t_err)
        r_errors.append(r_err)

    print("Translation Error (units same as input poses):")
    print(f"  Avg: {np.mean(t_errors):.4f}")
    print(f"  Max: {np.max(t_errors):.4f}")
    print(f"  Min: {np.min(t_errors):.4f}")

    print("Rotation Error (degrees):")
    print(f"  Avg: {np.mean(r_errors):.4f}")
    print(f"  Max: {np.max(r_errors):.4f}")
    print(f"  Min: {np.min(r_errors):.4f}")

if __name__ == "__main__":
    pred_dir = "debug/ob_in_cam"   # directory containing predicted .txt files
    gt_dir = "asteroid_example/my_object/meta"      # directory containing ground truth .json files
    evaluate(pred_dir, gt_dir)
