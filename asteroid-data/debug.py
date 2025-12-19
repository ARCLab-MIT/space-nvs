import cv2
import trimesh

img = cv2.imread("data/my_object/images/depth/img_000000.png")
# TODO check the sample mustard bottle data for reference

h, w, c = img.shape

for y in range(h):
    for x in range(w):
        b, g, r = img[y, x]
        if b != 0 or g != 0 or r != 0:
            print(f"({x},{y}) = B:{b} G:{g} R:{r}")


mesh = trimesh.load("Eros Gaskell 50k poly.obj")
# 3D asteroid catalog: https://3d-asteroids.space/asteroids/433-Eros
mesh.apply_scale(1e3) # likely in km instead of m

# mesh.show()

# print(f"Volume (m³): {mesh.volume}")
# print(f"Surface area (m²): {mesh.area}")
# print(f"Centroid: {mesh.center_mass}")
