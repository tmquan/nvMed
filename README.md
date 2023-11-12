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