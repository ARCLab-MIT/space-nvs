import os
import json
import bpy
import math
import sys
import time
from mathutils import Matrix, Vector

def matrix_to_list(matrix):
    """Convert a Blender matrix to a list of lists for JSON serialization."""
    return [list(row) for row in matrix]

def add_track_to_constraint(obj, target):
    """Add a Track To constraint to the object to make it point at the target."""
    constraint = obj.constraints.new(type='TRACK_TO')
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

def scale_to_unit_size(obj):
    """Scale the object to fit within a unit size bounding box."""
    dimensions = obj.dimensions
    max_dimension = max(dimensions)
    scale_factor = 1.0 / max_dimension
    obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.context.view_layer.update()

def setup_scene(obj_filepath):
    # Clear existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    bpy.ops.wm.obj_import(filepath=obj_filepath)
    imported_object = bpy.context.selected_objects[0]

    # Scale the object to a unit size
    scale_to_unit_size(imported_object)

    # Add an empty at the origin for the camera to track
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    empty = bpy.context.active_object

    # Add camera
    bpy.ops.object.camera_add(location=(0, -3, 1))
    camera = bpy.context.active_object
    add_track_to_constraint(camera, empty)

    # Set this camera as the active camera for the scene
    bpy.context.scene.camera = camera

    # Adjust camera to fit the entire object
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action='DESELECT')
    imported_object.select_set(True)
    bpy.context.view_layer.objects.active = imported_object
    bpy.ops.view3d.camera_to_view_selected()
    
    # Create an area light
    bpy.ops.object.light_add(type='AREA', location=(5, 5, 5))
    key_light = bpy.context.active_object
    key_light.data.energy = 1000 
    add_track_to_constraint(key_light, empty)

    # Add a fill light
    bpy.ops.object.light_add(type='AREA', location=(-5, -5, 5))
    fill_light = bpy.context.active_object
    fill_light.data.energy = 500 
    add_track_to_constraint(fill_light, empty)

    # Add a rim light
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 5))
    rim_light = bpy.context.active_object
    rim_light.data.energy = 750
    add_track_to_constraint(rim_light, empty)

    # Adjust area light sizes
    for light in [key_light, fill_light, rim_light]:
        light.data.size = 5  

    # Set render resolution to 256x256
    bpy.context.scene.render.resolution_x = 256
    bpy.context.scene.render.resolution_y = 256
    bpy.context.scene.render.resolution_percentage = 100

    # ----- Comment below section out if using CPU ----- #
    # Set blender preferences to GPU
    bpy.data.scenes[0].render.engine = "CYCLES"

    # Set the device_type
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA" # or "OPENCL"

    # Set the device and feature set
    bpy.context.scene.cycles.device = "GPU"

    # get_devices() to let Blender detects GPU device
    bpy.context.preferences.addons["cycles"].preferences.get_devices()

    return camera

def render_and_save(camera, frame, output_dir):
    # Ensure the camera is set as active
    bpy.context.scene.camera = camera

    # Update the scene
    bpy.context.view_layer.update()

    # Render and save image
    bpy.context.scene.render.filepath = os.path.join(output_dir, '%03d.png' % frame)
    try:
        bpy.ops.render.render(write_still=True)
    except Exception as e:
        print(f"Error rendering frame {frame}: {str(e)}")

def process_model(obj_filepath, output_dir):
    camera = setup_scene(obj_filepath)

    if not camera:
        print("Failed to set up the scene. No camera created.")
        return

    frames = []
    num_frames = 16  # Number of views to generate per plane

    # Function to handle camera rotation and rendering
    def rotate_and_render(start_frame, axis):
        for i in range(num_frames):
            angle = (2 * math.pi * i) / num_frames
            if axis == 'xy':
                camera.location = (3 * math.cos(angle), 3 * math.sin(angle), 1)
            elif axis == 'yz':
                camera.location = (1, 3 * math.cos(angle), 3 * math.sin(angle))
            elif axis == 'xz':
                camera.location = (3 * math.cos(angle), 1, 3 * math.sin(angle))
            
            bpy.context.view_layer.update()
            
            frame = start_frame + i
            render_and_save(camera, frame, output_dir)

            # Get and store camera matrix
            matrix = get_camera_matrix(camera)
            frames.append({
                "file_path": f"./train/r_{frame}",
                "transform_matrix": matrix_to_list(matrix.transposed())
            })

    # Rotate camera around object in different planes
    rotate_and_render(0, 'xy')
    rotate_and_render(num_frames, 'yz')
    rotate_and_render(2 * num_frames, 'xz')

    # Save JSON file
    json_data = {
        "camera_angle_x": math.pi / 2,  
        "frames": frames
    }

    with open(os.path.join(output_dir, "transforms_train.json"), "w") as f:
        json.dump(json_data, f, indent=4)

def get_camera_matrix(camera):
    # Get camera matrix
    return camera.matrix_world.copy()


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    obj_path = argv[0]
    output_dir = argv[1]
    
    process_model(obj_path, output_dir)

if __name__ == "__main__":
    main()