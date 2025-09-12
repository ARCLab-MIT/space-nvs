## Setup
First, clone the TRELLIS repository and cd into the directory:
```sh
git clone
cd TRELLIS/
```

Create a new conda environment and download the necessary dependencies by running the below commands:
```sh
conda create -n trellis python=3.10
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge cxx-compiler
. ./setup.sh --basic --train --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
```
Note: the version numbers in the above commands may differ based on which CUDA driver version you have. Some higher CUDA versions are not supported by the setup.sh script, so it is recommended that you download cuda-toolkit in conda as shown.

## Running inference
To reconstruct a 3D model with the vanilla TRELLIs model, simply run `python example.py`.

To use a different checkpoint to replace some part of the TRELLIS model, add this code in place of the pipeline creation in `example.py`:

```python
pretrained_path = "JeffreyXiang/TRELLIS-image-large/ckpts"
model_dict = {
    "sparse_structure_decoder": models.from_pretrained(f"{pretrained_path}/ss_dec_conv3d_16l8_fp16").cuda(),
    "sparse_structure_flow_model": models.from_pretrained(f"{pretrained_path}/ss_flow_img_dit_L_16l8_fp16").cuda(),
    "slat_decoder_gs": models.from_pretrained(f"{pretrained_path}/slat_dec_gs_swin8_B_64l8gs32_fp16").cuda(),
    "slat_decoder_rf": models.from_pretrained(f"{pretrained_path}/slat_dec_rf_swin8_B_64l8r16_fp16").cuda(),
    "slat_decoder_mesh": models.from_pretrained(f"{pretrained_path}/slat_dec_mesh_swin8_B_64l8m256c_fp16").cuda(),
    "slat_flow_model": models.from_pretrained(f"{pretrained_path}/slat_flow_img_dit_L_64l8p2_fp16").cuda(),
}

# replace with path to your checkpoint
ckpt_paths = {
    "sparse_structure_decoder": "outputs/ss_vae_ast/ckpts/decoder_step0600000.pt",
}

for name, ckpt in ckpt_paths.items():
    model_dict[name].load_state_dict(torch.load(dist_utils.read_file_dist(ckpt), weights_only=True))

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
for name, model in model_dict.items():
    pipeline.models[name] = model
pipeline.cuda()
```

## Dataset Creation
Download `build_dataset.py` and put it under `dataset_toolkits/datasets` (next to all the other files that look like `<dataset_name>.py`).

You should now be able to follow through the dataset creation instructions in DATASET.md in the trellis github, but using `build_dataset` in place of `<SUBSET>`.

## Training
To train, first follow the setup instructions in the TRELLIS repository. 

To finetune from the pretrained TRELLIS checkpoint, replace the train.py file in the original TRELLIS repository with the train.py file in this folder. The command to run is
```sh
python train.py --config configs/vae/ss_vae_conv3d_16l8_fp16.json --output_dir <OUTPUT_DIR> --data_dir <DATASET_DIR> --pretrained --num_gpus 2
```
Add arguments as you need following the instructions in the original TRELLIS repository.