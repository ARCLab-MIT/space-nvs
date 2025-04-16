import os
import json
import bpy
import math
import time
import subprocess
from mathutils import Matrix, Vector
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'
RENDER_SCRIPT = '/mnt/data/xinghui/space-nvs/zero123_render.py'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')


def main():
    print('Checking blender...')
    _install_blender()

    original_data_dir = "/mnt/data/xinghui/space-nvs/3D_models/asteroid_catalog"  # Change to your original_data directory
    new_data_dir = "/mnt/data/xinghui/space-nvs/3D_models/asteroid_frames"  # Change to your new_data directory
    os.makedirs(new_data_dir, exist_ok=True)

    obj_files = [f for f in os.listdir(original_data_dir) if f.endswith('.obj')][:5]

    MAX_PROCESSES = min(os.cpu_count(), 4)
    processes = []

    for idx, obj_file in enumerate(tqdm(obj_files)):
        obj_path = os.path.join(original_data_dir, obj_file)
        output_dir = os.path.join(new_data_dir, f"object_{idx + 1}")
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            "python3", RENDER_SCRIPT,
            "--",  # Everything after this goes to Python
            obj_path,
            output_dir
        ]
        print(cmd)

        p = subprocess.Popen(cmd)
        processes.append(p)

        if len(processes) >= MAX_PROCESSES:
            # Wait for the oldest process to finish
            if not processes[0].poll():
                processes[0].wait()
            processes.pop(0)

    # Wait for remaining
    for p in processes:
        p.wait()

if __name__ == "__main__":
    main()