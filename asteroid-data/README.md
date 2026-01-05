# Data Generation for FoundationPose

* Given an .obj file, run ```foundation_pose_data.py``` to generate rgb, depth, and segementation masks for the target object, as well as metadata.
* ```moving_cam.py```: gathers images of an object by moving the camera around, one can modify the path the camera takes
* ```moving_obj.py```: gathers images of an object by moving the camera and object around, one can modify the paths 
* ```debug.py```: ignore 

## FoundationPose Steps

I had to make modifications to some code in FoundationPose, detailed in the ```FoundationPose``` folder. Substituting the corresponding files in FoundationPose github seemed to work (along with downloading weights and such). Explanations may be found in Google Drive slides.

Steps to run (install requires more steps, check FoundationPose GitHub):
1. New terminal: ```cd docker/```
2. ```docker pull shingarey/foundationpose_custom_cuda121:latest```
3. ```cd ..```
4. ```bash docker/run_container.sh```
5. New terminal: ```docker exec -it foundationpose bash```
6. ```cd mnt/data/your_kerb```
7. Run FoundationPose: ```python run_demo_remote.py```
8. Run evaluation: ```python eval.py```
