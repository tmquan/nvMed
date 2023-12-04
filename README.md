# nvMed
Novel View Medical Image Reconstruction

```
git clone https://github.com/tmquan/nvmed
conda create -n py311 python=3.11
conda activate py311
pip install -U pip 
pip install -U transformers
curl https://sh.rustup.rs -sSf | sh
source "$HOME/.cargo/env"
rustc -V
pip install -U plotly
pip install -U diffusers
pip install -U lightning
pip install -U tensorboard

pip install -U monai
pip install -U einops
pip install -U lmdb
pip install -U mlflow
pip install -U clearml
pip install -U scikit-image
pip install -U pytorch-ignite
pip install -U pandas
pip install -U pynrrd
pip install -U gdown
pip install -U itk

pip install -U git+https://github.com/Project-MONAI/GenerativeModels.git 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu114
```


```
python -c 'import torch; print(torch.cuda.is_available())'
python -c 'import torch; torch.randn(1, device="cuda:0")'
python -c "import torch; import monai; from monai.config import print_config; print_config()"
python -c "import torch; import pytorch3d; from monai.config import print_config; print_config()"
```


```
env CUDA_VISIBLE_DEVICES='4,5,6,7' CUDA_LAUNCH_BLOCKING=1  python main_nvmed_inv.py --accelerator='gpu' --devices=4 --batch_size=2 --lr=1e-4 --epochs=201 --logsdir=/home/quantm/logs/b0_nvmed --datadir=/home/quantm/data --train_samples=4000 --val_samples=800 --n_pts_per_ray=400 --vol_shape=256 --img_shape=256 --fov_depth=256 --alpha=1 --theta=1 --gamma=1 --delta=0.002 --omega=1 --lamda=1 --sh=0 --pe=0 --prediction_type='sample' --amp --strategy=auto --backbone='efficientnet-b0' --resample --phase=ctonly 

env CUDA_VISIBLE_DEVICES='4,5,6,7' CUDA_LAUNCH_BLOCKING=1  python main_nvmed_inv.py --accelerator='gpu' --devices=4 --batch_size=2 --lr=1e-4 --epochs=201 --logsdir=/home/quantm/logs/b0_nvmed --datadir=/home/quantm/data --train_samples=4000 --val_samples=800 --n_pts_per_ray=256 --vol_shape=256 --img_shape=256 --fov_depth=256 --alpha=1 --theta=1 --gamma=1 --delta=0.002 --omega=1 --lamda=1 --sh=0 --pe=0 --prediction_type='sample' --amp --strategy=auto --backbone='efficientnet-b0' --resample --phase=ctxray --tfunc --perceptual --ckpt=/home/quantm/logs/b0_nvmed_resample_ctonly/epoch=149-step=75000.ckpt 

```