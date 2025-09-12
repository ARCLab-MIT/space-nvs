conda create -n ouroboros python=3.10
conda activate ouroboros
conda install nvidia/label/cuda-12.6.0::cuda-toolkit
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -y \
  numpy=1.23.5 \
  matplotlib \
  scikit-learn \
  imageio \
  pandas \
  protobuf=3.20.3 \
  tqdm \
  rich \

pip install \
  PyMCubes \
  xatlas \
  lightning>=2.2.1 \
  hydra-core==1.3.2 \
  hydra-colorlog==1.2.0 \
  hydra-optuna-sweeper==1.2.0 \
  omegaconf==2.3.0 \
  rootutils==1.0.7 \
  wandb==0.15.11 \
  tensorboard==2.14.1 \
  tensorboard-data-server==0.7.1 \
  diffusers==0.25.0 \
  transformers==4.28.1 \
  accelerate==0.23.0 \
  einops==0.6.1 \
  opencv-python-headless==4.7.0.72 \
  google==3.0.0 \
  googleapis-common-protos==1.59.0 \
  timm==0.9.7 \
  kornia==0.7.0 \
  imageio[ffmpeg] \
  imageio[pyav] \
  jaxtyping \
  nerfacc \
  mesh2sdf \
  transforms3d \
  plyfile==0.8.1 \
  wrapt==1.15.0 \
  httpx
pip install \
  git+https://github.com/ashawkey/diff-gaussian-rasterization.git \
  git+https://gitlab.inria.fr/bkerbl/simple-knn.git \
  git+https://github.com/NVlabs/nvdiffrast.git
