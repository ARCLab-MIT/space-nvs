import os
import sys
import json
import numpy as np

def convert_transforms_to_npy(root_dir):
    """
    1. For each subdirectory in `root_dir`, look for `transforms_train.json`.
    2. For each frame in the JSON, save `transform_matrix` to a .npy file.
    3. .npy filename is derived from `file_path` in the JSON.
    """

    # Ensure the root directory exists
    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a valid directory.")
        return

    # Loop over each subdirectory in root_dir
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue  # skip files at this level

        # Look for transforms_train.json in this subdirectory
        transforms_file = os.path.join(folder_path, "transforms_train.json")
        if not os.path.isfile(transforms_file):
            print(f"No transforms_train.json in {folder_path}, skipping.")
            continue

        print(f"Found transforms_train.json in {folder_path}, processing...")

        # Load the JSON data
        with open(transforms_file, "r") as f:
            data = json.load(f)

        # "frames" should be a list of objects with "file_path" & "transform_matrix"
        frames = data.get("frames", [])
        if not frames:
            print(f"No frames found in {transforms_file}, skipping.")
            continue

        for frame in frames:
            # file_path might be "r_0", "1", or even "./train/r_0"
            raw_name = frame.get("file_path", "")
            if raw_name.startswith("./"):
                raw_name = raw_name[2:]  # remove leading "./" if present

            # Convert transform_matrix to a NumPy array
            transform_matrix = np.array(frame["transform_matrix"], dtype=np.float32)

            # Example: if raw_name="r_0", we'll produce "r_0.npy"
            # If raw_name="train/r_0", we produce "train/r_0.npy"
            # We'll store .npy in the same subdirectory as transforms_train.json
            base_name = os.path.basename(raw_name)  # e.g. "r_0" from "train/r_0"
            npy_path = os.path.join(folder_path, f"{base_name}.npy")
            # Ensure subfolders exist if raw_name includes a slash
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)

            # Save the transform as .npy
            np.save(npy_path, transform_matrix)
            print(f"  -> Saved {npy_path}")

    print("Conversion complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_json_to_npy.py <root_dir>")
        sys.exit(1)

    root_dir = sys.argv[1]
    convert_transforms_to_npy(root_dir)
