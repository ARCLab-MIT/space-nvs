## Zero123XL Repository changes

### zero123/requirements.txt
- At line 5, change the version of `opencv-python` to 4.10.0.84

### zero123/configs/sd-objaverse-finetune-c_concat-256.yaml
- At line 75, change the data directory path to your own
```
data:
  target: ldm.data.simple.ObjaverseDataModuleFromConfig
  params:
    root_dir: 'view_whole_sphere'    # change this to your own directory
```

### zero123/ldm/data/simple.py
- At line 210 and 216, change the data directory to your own and change the number of views to 48 to match the blender script.
```
class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='.objaverse/hf-objaverse-v1/views',    # change this to your own directory
        image_transforms=[],
        ext="png",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=12,  # change to 48
        validation=False
        ) -> None:
```

- At lines 234-235, since the blender script does not produce a `valid_paths.json` file, you can replace it with the below code
```
try:
    with open(os.path.join(root_dir, 'valid_paths.json')) as f:
        self.paths = json.load(f)
except:
    self.paths = [f'object_{i}' for i in range(1, 211)]
```
The training/validation split can be adjusted on lines 239 and 241.

- At lines 258-261, change the matrix multiplication code inside the get_T function to the following to work with our generated npy files:
```
R, T = target_RT[:3, :3], target_RT[3, :3]
T_target = -R.T @ T

R, T = cond_RT[:3, :3], cond_RT[3, :3]
T_cond = -R.T @ T
```

This should allow the finetuning to run successfully.