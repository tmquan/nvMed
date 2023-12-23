import os
import warnings

warnings.filterwarnings("ignore")
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print(rlimit)
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, rlimit[1]))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.distributed.fsdp.wrap import wrap

torch.set_float32_matmul_precision("medium")

from typing import Optional, NamedTuple
from lightning_fabric.utilities.seed import seed_everything
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger

from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from argparse import ArgumentParser
from tqdm.auto import tqdm

from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch3d.renderer.cameras import (
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    look_at_view_transform,
)

from monai.losses import PerceptualLoss
from monai.losses import SSIMLoss
from monai.networks.nets import Unet
from monai.networks.layers.factories import Norm

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

from datamodule import UnpairedDataModule, ExampleDataModule
from dvr.renderer import ReverseXRayVolumeRenderer 
from dvr.renderer import normalized
from dvr.renderer import standardized

backbones = {
    "efficientnet-b0": (16, 24, 40, 112, 320),
    "efficientnet-b1": (16, 24, 40, 112, 320),
    "efficientnet-b2": (16, 24, 48, 120, 352),
    "efficientnet-b3": (24, 32, 48, 136, 384),
    "efficientnet-b4": (24, 32, 56, 160, 448),
    "efficientnet-b5": (24, 40, 64, 176, 512),
    "efficientnet-b6": (32, 40, 72, 200, 576),
    "efficientnet-b7": (32, 48, 80, 224, 640),
    "efficientnet-b8": (32, 56, 88, 248, 704),
    "efficientnet-b9": (32, 64, 96, 256, 800),
    "efficientnet-b0": (64, 128, 256, 512, 1024),
    "efficientnet-l2": (72, 104, 176, 480, 1376),
}


def make_cameras_dea(
    dist: torch.Tensor, 
    elev: torch.Tensor, 
    azim: torch.Tensor, 
    fov: int = 40, 
    znear: int = 4.0, 
    zfar: int = 8.0, 
    is_orthogonal: bool = False
):
    assert dist.device == elev.device == azim.device
    _device = dist.device
    R, T = look_at_view_transform(dist=dist.float(), elev=elev.float() * 90, azim=azim.float() * 180)
    if is_orthogonal:
        return FoVOrthographicCameras(R=R, T=T, znear=znear, zfar=zfar).to(_device)
    return FoVPerspectiveCameras(R=R, T=T, fov=fov, znear=znear, zfar=zfar).to(_device)

def init_weights(net, init_type = 'normal', gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain = gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain = gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
    
    
class InverseXrayVolumeRenderer(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        img_shape=256,
        vol_shape=256,
        fov_depth=256,
        sh=0,
        pe=0,
        backbone="efficientnet-b9",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_shape = img_shape
        self.vol_shape = vol_shape
        self.fov_depth = fov_depth
        self.pe = pe
        self.sh = sh
        assert backbone in backbones.keys()

        self.net2d3d = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,  # Condition with straight/hidden view
            out_channels=self.fov_depth,
            num_channels=backbones[backbone],
            attention_levels=[False, False, False, True, True],
            norm_num_groups=8,
            num_res_blocks=2,
            with_conditioning=True,
            cross_attention_dim=12,  # flatR | flatT
        )
        
        # self.net3d3d = nn.Sequential(
        #     Unet(
        #         spatial_dims=3, 
        #         in_channels=1, 
        #         out_channels=1, 
        #         channels=backbones[backbone], 
        #         strides=(2, 2, 2, 2, 2), 
        #         num_res_units=2, 
        #         kernel_size=3, 
        #         up_kernel_size=3, 
        #         act=("LeakyReLU", {"inplace": True}), 
        #         norm=Norm.BATCH,
        #         dropout=0.5
        #     ),
        # )
        
    def forward(self, image2d, cameras, resample=False, timesteps=None, is_training=False):
        _device = image2d.device
        B = image2d.shape[0]
        dtype = image2d.dtype
        if timesteps is None:
            timesteps = torch.zeros((B), device=_device).long()
        
        R = cameras.R
        inv = torch.cat([torch.inverse(R).reshape(B, 1, -1), torch.zeros((B, 1, 3), device=_device)], dim=-1)
        
        # Run forward pass
        fov = self.net2d3d(
            x=image2d, 
            context=inv,
            timesteps=timesteps,
        ).view(-1, 1, self.fov_depth, self.img_shape, self.img_shape)

        if resample:
            grd = F.affine_grid(inv, fov.size()).type(dtype)
            return F.grid_sample(fov, grd)
        else:
            return fov 
        
class FlexiblePerceptualLoss(PerceptualLoss):
    def __init__(self,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.shape != source.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from source ({source.shape})")
        self.perceptual_function.eval()
        if len(source.shape) == 5:
            # Compute 2.5D approach
            loss_sagittal = self._calculate_axis_loss(source, target, spatial_axis=2)
            loss_coronal = self._calculate_axis_loss(source, target, spatial_axis=3)
            loss_axial = self._calculate_axis_loss(source, target, spatial_axis=4)
            loss = loss_sagittal + loss_axial + loss_coronal
        elif len(source.shape) == 4:
            # 2D and real 3D cases
            loss = self.perceptual_function(source, target)

        return torch.mean(loss)
    
class NVMLightningModule(LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super().__init__()
        self.lr = hparams.lr

        self.ckpt = hparams.ckpt
        self.strict = hparams.strict
        self.phase = hparams.phase
        self.img_shape = hparams.img_shape
        self.vol_shape = hparams.vol_shape
        self.fov_depth = hparams.fov_depth
        self.seed = hparams.seed
        
        self.lpips = hparams.lpips
        self.alpha = hparams.alpha
        self.gamma = hparams.gamma
        self.delta = hparams.delta
        self.theta = hparams.theta
        self.omega = hparams.omega
        self.lamda = hparams.lamda
        self.tfunc = hparams.tfunc
        
        self.timesteps = hparams.timesteps
        self.resample = hparams.resample
        self.perceptual2d = hparams.perceptual2d
        self.perceptual3d = hparams.perceptual3d
        self.perceptual = hparams.perceptual
        self.ssim = hparams.ssim
        
        self.logsdir = hparams.logsdir
        self.sh = hparams.sh
        self.pe = hparams.pe

        self.n_pts_per_ray = hparams.n_pts_per_ray
        self.weight_decay = hparams.weight_decay
        self.batch_size = hparams.batch_size
        self.devices = hparams.devices
        self.backbone = hparams.backbone
        
        self.save_hyperparameters()

        self.fwd_renderer = ReverseXRayVolumeRenderer(
            image_width=self.img_shape, 
            image_height=self.img_shape, 
            n_pts_per_ray=self.n_pts_per_ray, 
            min_depth=4.0, 
            max_depth=8.0, 
            ndc_extent=1.0,
        )

        if self.tfunc:
            # Define the embedding layer
            self.data_dim = 4096
            self.feat_dim = 1
            self.emb_function = nn.Embedding(self.data_dim, self.feat_dim)
            # Initialize the embedding weights linearly
            lin_weight = torch.linspace(0, self.data_dim, self.data_dim).unsqueeze(1) / self.data_dim
            # one_weight = torch.ones_like(lin_weight)
            self.emb_function.weight = nn.Parameter(lin_weight.repeat(1, self.feat_dim))

        self.inv_renderer = InverseXrayVolumeRenderer(
            in_channels=1, 
            out_channels=self.sh ** 2 if self.sh > 0 else 1, 
            vol_shape=self.vol_shape, 
            img_shape=self.img_shape, 
            sh=self.sh, 
            pe=self.pe, 
            backbone=self.backbone,
        )
        # init_weights(self.inv_renderer, init_type="normal")
        
        # @ Diffusion 
        self.unet2d_model = DiffusionModelUNet(
            spatial_dims=2, 
            in_channels=1, 
            out_channels=1, 
            num_channels=backbones[self.backbone], 
            attention_levels=[False, False, False, True, True], 
            norm_num_groups=16, 
            num_res_blocks=2, 
            with_conditioning=True, 
            cross_attention_dim=12, # Condition with straight/hidden view  # flatR | flatT
        )
                
        self.ddpmsch = DDPMScheduler(
            num_train_timesteps=self.timesteps, 
            schedule="scaled_linear_beta", 
            prediction_type=hparams.prediction_type, 
            beta_start=0.0005, 
            beta_end=0.0195,
        )
        self.ddimsch = DDIMScheduler(
            num_train_timesteps=self.timesteps, 
            schedule="scaled_linear_beta", 
            prediction_type=hparams.prediction_type, 
            beta_start=0.0005, 
            beta_end=0.0195, 
            clip_sample=True,
        )
        self.ddimsch.set_timesteps(num_inference_steps=100)
        self.inferer = DiffusionInferer(scheduler=self.ddimsch)
                
        if self.ckpt:
            print("Loading.. ", self.ckpt)
            checkpoint = torch.load(self.ckpt, map_location=torch.device("cpu"))["state_dict"]
            state_dict = {k: v for k, v in checkpoint.items() if k in self.state_dict()}
            self.load_state_dict(state_dict, strict=self.strict)

        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.maeloss = nn.L1Loss(reduction="mean")
        
        if self.perceptual2d:
            self.pctloss = FlexiblePerceptualLoss(
                spatial_dims=2, 
                network_type="radimagenet_resnet50", 
                # network_type="resnet50", 
                # network_type="medicalnet_resnet50_23datasets", 
                is_fake_3d=False, # fake_3d_ratio=20/256, # 0.0625, # 16/256
                pretrained=True,
            )
        if self.perceptual:
            self.pctloss = FlexiblePerceptualLoss(
                spatial_dims=3, 
                network_type="radimagenet_resnet50", 
                # network_type="resnet50", 
                # network_type="medicalnet_resnet50_23datasets", 
                is_fake_3d=True, fake_3d_ratio=20/256, # 0.0625, # 16/256
                pretrained=True,
            )
        if self.ssim:
            self.s2dloss = SSIMLoss(spatial_dims=2)
            self.s3dloss = SSIMLoss(spatial_dims=3)
            
    def correct_window(self, T_old, a_min=-1024, a_max=3071, b_min=-512, b_max=3071):
        # Calculate the range for the old and new scales
        range_old = a_max - a_min
        range_new = b_max - b_min
        # Reverse the incorrect scaling
        T_raw = (T_old * range_old) + a_min
        # Apply the correct scaling
        T_new = (T_raw - b_min) / range_new
        return T_new.clamp_(0, 1)
        # windowed = self.correct_window(image3d, a_min=-1024, a_max=3071, b_min=-600, b_max=3071)
       
    def forward_screen(self, image3d, cameras):
        if self.tfunc:
            # Ensure the range
            density = image3d.clamp(0, 1)
            B, C, D, H, W = density.shape
            # Bin the data
            binning = density * (self.data_dim - 1)
            binning = binning.long().clamp_(0, self.data_dim - 1)
            # Apply the embedding
            flatten = self.emb_function(binning.flatten()) 
            # Reshape to the original tensor shape
            density = flatten.view((B, -1, D, H, W))
            # Ensuring the output is in the range (0, 1)
            density = torch.clamp_(density, 0, 1)
            return self.fwd_renderer(density, cameras)
        else:
            image3d = self.correct_window(image3d, a_min=-1024, a_max=3071, b_min=-512, b_max=3071)
            return self.fwd_renderer(image3d, cameras)

    def forward_volume(self, image2d, cameras, n_views=[2, 1], resample=False, timesteps=None, is_training=False):
        _device = image2d.device
        B = image2d.shape[0]
        assert B == sum(n_views)  # batch must be equal to number of projections
        
        # Transpose on the fly to make it homogeneous
        if self.resample:
            image2d = torch.flip(image2d, [2, 3])
        else:
            image2d = torch.flip(image2d, [3])
        image2d = image2d.transpose(2, 3)
        
        results = self.inv_renderer(image2d, cameras, resample, timesteps, is_training)
        return results

    def forward_timing(self, image2d, cameras, n_views=[2, 1], resample=False, timesteps=None):
        _device = image2d.device
        B = image2d.shape[0]
        assert B == sum(n_views)  # batch must be equal to number of projections
        if timesteps is None:
            timesteps = torch.zeros((B,), device=_device).long()
        # viewpts = torch.cat([cameras.R.reshape(B, 1, -1), cameras.T.reshape(B, 1, -1),], dim=-1,)
        # R = cameras.R
        # # T = cameras.T.unsqueeze_(-1)
        # T = torch.zeros_like(cameras.T.unsqueeze_(-1))
        # # mat = torch.cat([R, T], dim=-1)
        # print(R.shape, T.shape)
        # inv = torch.cat([torch.inverse(R), -T], dim=-1)
        R = cameras.R
        inv = torch.cat([torch.inverse(R).reshape(B, 1, -1), torch.zeros((B, 1, 3), device=_device)], dim=-1)
        results = self.unet2d_model(x=image2d * 2.0 - 1.0, context=inv, timesteps=timesteps,) * 0.5 + 0.5
        return results
    
    def _common_step(self, batch, batch_idx, stage: Optional[str] = "evaluation"):
        image3d = batch["image3d"]
        image2d = batch["image2d"]
        _device = batch["image3d"].device
        batchsz = image2d.shape[0]

        # Construct the random cameras, -1 and 1 are the same point in azimuths
        dist_random = 6.0 * torch.ones(self.batch_size, device=_device)
        elev_random = torch.rand_like(dist_random) - 0.5
        azim_random = torch.rand_like(dist_random) * 2 - 1  # [0 1) to [-1 1)
        view_random = make_cameras_dea(dist_random, elev_random, azim_random, fov=20, znear=4, zfar=8)

        dist_hidden = 6.0 * torch.ones(self.batch_size, device=_device)
        elev_hidden = torch.zeros(self.batch_size, device=_device)
        azim_hidden = torch.zeros(self.batch_size, device=_device)
        view_hidden = make_cameras_dea(dist_hidden, elev_hidden, azim_hidden, fov=20, znear=4, zfar=8)

        # Construct the samples in 2D
        # with torch.no_grad():
        figure_xr_hidden = image2d
        figure_ct_random = self.forward_screen(image3d=image3d, cameras=view_random)
        figure_ct_hidden = self.forward_screen(image3d=image3d, cameras=view_hidden)
        
        view_concat = join_cameras_as_batch([view_random, view_hidden])
        
        # # Reconstruct the Encoder-Decoder
        # volume_dx_concat = self.forward_volume(
        #     image2d=torch.cat([figure_ct_random, figure_ct_hidden]), 
        #     cameras=view_concat,
        #     n_views=[1, 1]*batchsz,
        #     resample=self.resample,
        #     timesteps=None, 
        #     is_training=(stage=="train"),
        # )
        # volume_ct_random_inverse, volume_ct_hidden_inverse = torch.split(volume_dx_concat, batchsz)
        
        # # Reconstruct the Encoder-Decoder
        # volume_xr_hidden_inverse = self.forward_volume(
        #     image2d=torch.cat([figure_xr_hidden]), 
        #     cameras=join_cameras_as_batch([view_hidden]), 
        #     n_views=[1]*batchsz,
        #     resample=self.resample,
        #     timesteps=None, 
        #     is_training=(stage=="train"),
        # )
        
        # # with torch.no_grad():
        # figure_xr_hidden_inverse_random = self.forward_screen(image3d=volume_xr_hidden_inverse, cameras=view_random)
        # figure_xr_hidden_inverse_hidden = self.forward_screen(image3d=volume_xr_hidden_inverse, cameras=view_hidden)
        # figure_ct_random_inverse_random = self.forward_screen(image3d=volume_ct_random_inverse, cameras=view_random)
        # figure_ct_random_inverse_hidden = self.forward_screen(image3d=volume_ct_random_inverse, cameras=view_hidden)
        # figure_ct_hidden_inverse_random = self.forward_screen(image3d=volume_ct_hidden_inverse, cameras=view_random)
        # figure_ct_hidden_inverse_hidden = self.forward_screen(image3d=volume_ct_hidden_inverse, cameras=view_hidden)

        # if self.sh > 0:
        #     volume_xr_hidden_inverse = volume_xr_hidden_inverse.sum(dim=1, keepdim=True)
        #     volume_ct_random_inverse = volume_ct_random_inverse.sum(dim=1, keepdim=True)
        #     volume_ct_hidden_inverse = volume_ct_hidden_inverse.sum(dim=1, keepdim=True)

        # @ Diffusion step: 2 kinds of blending
        timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (batchsz,), device=_device).long()  # 3 views

        # figure_xr_latent_hidden = torch.randn_like(image2d)
        volume_xr_latent = torch.randn_like(image3d) * 0.5 + 0.5
        figure_xr_latent_hidden = self.forward_screen(image3d=volume_xr_latent, cameras=view_hidden)
        figure_xr_interp_hidden = self.ddpmsch.add_noise(original_samples=image2d * 2.0 - 1.0, noise=figure_xr_latent_hidden * 2.0 - 1.0, timesteps=timesteps) * 0.5 + 0.5

        volume_ct_latent = torch.randn_like(image3d) * 0.5 + 0.5
        figure_ct_latent_random = self.forward_screen(image3d=volume_ct_latent, cameras=view_random)
        figure_ct_latent_hidden = self.forward_screen(image3d=volume_ct_latent, cameras=view_hidden)
        volume_ct_interp = self.ddpmsch.add_noise(original_samples=image3d * 2.0 - 1.0, noise=volume_ct_latent * 2.0 - 1.0, timesteps=timesteps) * 0.5 + 0.5
        figure_ct_interp_random = self.forward_screen(image3d=volume_ct_interp, cameras=view_random)
        figure_ct_interp_hidden = self.forward_screen(image3d=volume_ct_interp, cameras=view_hidden)
  
        # Run the backward diffusion (denoising + reproject)
        figure_dx_output = self.forward_timing(
            image2d=torch.cat([figure_ct_interp_random, figure_ct_interp_hidden]), 
            cameras=view_concat,
            n_views=[1, 1] * batchsz, 
            timesteps=timesteps.repeat(2),
        )
        figure_ct_output_random, figure_ct_output_hidden = torch.split(figure_dx_output, batchsz)
        figure_xr_output_hidden = self.forward_timing(
            image2d=figure_xr_interp_hidden, 
            cameras=view_hidden, 
            n_views=[1] * batchsz, 
            timesteps=timesteps.repeat(1),
        )
        
        # Reconstruct the Encoder-Decoder
        volume_dx_output = self.forward_volume(
            image2d=torch.cat([figure_ct_output_random, figure_ct_output_hidden]), 
            cameras=view_concat,
            n_views=[1, 1]*batchsz,
            resample=self.resample,
            timesteps=None, 
            is_training=(stage=="train"),
        )
        volume_ct_random_output, volume_ct_hidden_output = torch.split(volume_dx_output, batchsz)
        
        volume_xr_hidden_output = self.forward_volume(
            image2d=figure_xr_output_hidden, 
            cameras=view_hidden, 
            n_views=[1] * batchsz, 
            resample=self.resample,
            timesteps=None, 
            is_training=(stage=="train"),
        )

        figure_xr_hidden_output_random = self.forward_screen(image3d=volume_xr_hidden_output, cameras=view_random)
        figure_xr_hidden_output_hidden = self.forward_screen(image3d=volume_xr_hidden_output, cameras=view_hidden)
        figure_ct_random_output_random = self.forward_screen(image3d=volume_ct_random_output, cameras=view_random)
        figure_ct_random_output_hidden = self.forward_screen(image3d=volume_ct_random_output, cameras=view_hidden)
        figure_ct_hidden_output_random = self.forward_screen(image3d=volume_ct_hidden_output, cameras=view_random)
        figure_ct_hidden_output_hidden = self.forward_screen(image3d=volume_ct_hidden_output, cameras=view_hidden)

        if self.ddpmsch.prediction_type == "sample":
            figure_xr_target_hidden = figure_xr_hidden
            figure_ct_target_random = figure_ct_random
            figure_ct_target_hidden = figure_ct_hidden
            volume_ct_target = image3d
        elif self.ddpmsch.prediction_type == "epsilon":
            figure_xr_target_hidden = figure_xr_latent_hidden
            figure_ct_target_random = figure_ct_latent_random
            figure_ct_target_hidden = figure_ct_latent_hidden
            volume_ct_target = volume_ct_latent
        elif self.ddpmsch.prediction_type == "v_prediction":
            figure_xr_target_hidden = self.ddpmsch.get_velocity(figure_xr_hidden, figure_xr_latent_hidden, timesteps)
            figure_ct_target_random = self.ddpmsch.get_velocity(figure_ct_random, figure_ct_latent_random, timesteps)
            figure_ct_target_hidden = self.ddpmsch.get_velocity(figure_ct_hidden, figure_ct_latent_hidden, timesteps)
            volume_ct_target = self.ddpmsch.get_velocity(image3d, volume_ct_latent, timesteps)

        if self.phase == "ctonly":
            im2d_loss_dif = (
                  self.maeloss(figure_ct_output_random, figure_ct_target_random)
                + self.maeloss(figure_ct_output_hidden, figure_ct_target_hidden)
                + self.maeloss(figure_ct_hidden_output_random, figure_ct_target_random)
                + self.maeloss(figure_ct_hidden_output_hidden, figure_ct_target_hidden)
                + self.maeloss(figure_ct_random_output_random, figure_ct_target_random)
                + self.maeloss(figure_ct_random_output_hidden, figure_ct_target_hidden)
            )

            im3d_loss_dif = (
                  self.maeloss(volume_ct_random_output, volume_ct_target)
                + self.maeloss(volume_ct_hidden_output, volume_ct_target)
            )
        if self.phase == "ctxray":
            im2d_loss_dif = (
                  self.maeloss(figure_xr_output_hidden, figure_xr_target_hidden)
                + self.maeloss(figure_ct_output_random, figure_ct_target_random)
                + self.maeloss(figure_ct_output_hidden, figure_ct_target_hidden)
                + self.maeloss(figure_xr_hidden_output_hidden, figure_xr_target_hidden)
                + self.maeloss(figure_ct_hidden_output_random, figure_ct_target_random)
                + self.maeloss(figure_ct_hidden_output_hidden, figure_ct_target_hidden)
                + self.maeloss(figure_ct_random_output_random, figure_ct_target_random)
                + self.maeloss(figure_ct_random_output_hidden, figure_ct_target_hidden)
            )
            im3d_loss_dif = (
                  self.maeloss(volume_ct_random_output, volume_ct_target)
                + self.maeloss(volume_ct_hidden_output, volume_ct_target)
            )

        # Visualization step
        if batch_idx == 0:
            # Sampling step for X-ray
            with torch.no_grad():
                # Construct the context pose to diffusion model
                # pose_random = torch.cat([view_random.R.reshape(batchsz, 1, -1), view_random.T.reshape(batchsz, 1, -1),], dim=-1,)
                # pose_hidden = torch.cat([view_hidden.R.reshape(batchsz, 1, -1), view_hidden.T.reshape(batchsz, 1, -1),], dim=-1,)
                R = view_hidden.R
                inv = torch.cat([torch.inverse(R).reshape(batchsz, 1, -1), torch.zeros((batchsz, 1, 3), device=_device)], dim=-1)
        
                volume_xr_latent = torch.randn_like(image3d) * 0.5 + 0.5
                figure_xr_latent_hidden = self.forward_screen(image3d=volume_xr_latent, cameras=view_hidden)
                figure_xr_sample_hidden = self.inferer.sample(input_noise=figure_xr_latent_hidden * 2.0 - 1.0, 
                                                              diffusion_model=self.unet2d_model, 
                                                              conditioning=inv.view(batchsz, 1, -1), 
                                                              scheduler=self.ddimsch, 
                                                              verbose=False,) * 0.5 + 0.5
                volume_xr_sample_hidden = self.forward_volume(image2d=figure_xr_sample_hidden, 
                                                              cameras=view_hidden, 
                                                              n_views=[1] * batchsz, timesteps=None)
                # print(volume_xr_sample_hidden.shape)
                figure_xr_sample_hidden_random = self.forward_screen(image3d=volume_xr_sample_hidden, cameras=view_random)
                figure_xr_sample_hidden_hidden = self.forward_screen(image3d=volume_xr_sample_hidden, cameras=view_hidden)

            zeros = torch.zeros_like(image2d)
            viz2d = torch.cat([
                torch.cat([
                    image2d, 
                    # volume_xr_hidden_inverse[..., self.vol_shape // 2, :], 
                    # figure_xr_hidden_inverse_random, 
                    # figure_xr_hidden_inverse_hidden, 
                    volume_xr_hidden_output[..., self.vol_shape // 2, :], 
                    figure_xr_hidden_output_random, 
                    figure_xr_hidden_output_hidden, 
                    image3d[..., self.vol_shape // 2, :], 
                    figure_ct_random, 
                    figure_ct_hidden,
                ], dim=-2).transpose(2, 3),
                torch.cat([
                    zeros,
                    # volume_ct_random_inverse[..., self.vol_shape // 2, :],
                    # figure_ct_random_inverse_random,
                    # figure_ct_random_inverse_hidden,
                    # volume_ct_random_output[..., self.vol_shape // 2, :],
                    # figure_ct_random_output_random,
                    # figure_ct_random_output_hidden,
                    volume_xr_sample_hidden[..., self.vol_shape // 2, :], 
                    figure_xr_sample_hidden_random, 
                    figure_xr_sample_hidden_hidden, 
                    volume_ct_hidden_output[..., self.vol_shape // 2, :],
                    figure_ct_hidden_output_random,
                    figure_ct_hidden_output_hidden,
                ], dim=-2).transpose(2, 3),
            ], dim=-2)
            tensorboard = self.logger.experiment
            grid2d = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=True, nrow=1, padding=0)
            tensorboard.add_image(f"{stage}_df_samples", grid2d, self.current_epoch * self.batch_size + batch_idx)
            if self.tfunc:
                tensorboard.add_histogram(f"{stage}_emb_function", self.emb_function.weight, self.current_epoch * self.batch_size + batch_idx)
                       
        if self.phase == "ctonly":
            im3d_loss = im3d_loss_dif
            self.log(f"{stage}_im3d_loss", im3d_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
            im2d_loss = im2d_loss_dif
            self.log(f"{stage}_im2d_loss", im2d_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
            loss = self.alpha * im3d_loss + self.gamma * im2d_loss
            

        if self.phase == "ctxray":
            im3d_loss = im3d_loss_dif
            self.log(f"{stage}_im3d_loss", im3d_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
            im2d_loss = im2d_loss_dif
            self.log(f"{stage}_im2d_loss", im2d_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
            loss = self.alpha * im3d_loss + self.gamma * im2d_loss
            
        # if self.current_epoch < 21: # Warmup
        #     loss += 0.1 * self.maeloss(volume_xr_hidden_output, image3d) 
        #     loss += 0.1 * self.maeloss(figure_xr_hidden_output_random, figure_ct_random) 
        if self.perceptual2d and stage=="train":
            pc2d_loss = self.pctloss(figure_xr_hidden_output_random, figure_ct_random) \
                      + self.pctloss(figure_xr_hidden_output_hidden, figure_ct_hidden) \
                      + self.pctloss(figure_xr_hidden_output_hidden, image2d) 
            self.log(f"{stage}_pc2d_loss", pc2d_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
            loss += self.delta * pc2d_loss    
        if self.perceptual3d and stage=="train":
            pc3d_loss = self.p3dloss(volume_xr_hidden_output, image3d)
            self.log(f"{stage}_pc3d_loss", pc3d_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
            loss += self.delta * pc3d_loss 
        if self.perceptual and stage=="train":
            pc2d_loss = self.pctloss(figure_xr_hidden_output_random, figure_ct_random) \
                      + self.pctloss(figure_xr_hidden_output_hidden, figure_ct_hidden) \
                      + self.pctloss(figure_xr_hidden_output_hidden, image2d) 
            self.log(f"{stage}_pc2d_loss", pc2d_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
            loss += self.delta * pc2d_loss    
            pc3d_loss = self.pctloss(volume_xr_hidden_output, image3d) 
            self.log(f"{stage}_pc3d_loss", pc3d_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
            loss += self.delta * pc3d_loss    
        if self.ssim and stage=="train":
            si2d_loss = self.s2dloss(figure_xr_hidden_output_random, figure_ct_random) \
                      + self.s2dloss(figure_xr_hidden_output_hidden, figure_ct_hidden) \
                      + self.s2dloss(figure_xr_hidden_output_hidden, image2d) 
            self.log(f"{stage}_si2d_loss", si2d_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
            loss += self.omega * si2d_loss    
            si3d_loss = self.s3dloss(volume_xr_hidden_output, image3d) 
            self.log(f"{stage}_si3d_loss", si3d_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
            loss += self.omega * si3d_loss    
            
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, stage="train")
        self.train_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, stage="validation")
        self.validation_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        loss = torch.stack(self.train_step_outputs).mean()
        self.log(f"train_loss_epoch", loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.train_step_outputs.clear()  # free memory

    def on_validation_epoch_end(self):
        loss = torch.stack(self.validation_step_outputs).mean()
        self.log(f"validation_loss_epoch", loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                # {"params": self.fwd_renderer.parameters()},
                # {"params": self.emb_function.parameters()}, 
                {"params": self.unet2d_model.parameters()},
                {"params": self.inv_renderer.parameters()},
            ],
            # self.unet2d_model.parameters(),
            # self.inv_renderer.parameters(),
            # self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999)
            # 
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--conda_env", type=str, default="Unet")
    parser.add_argument("--notification_email", type=str, default="quantm88@gmail.com")
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)

    # Model arguments
    parser.add_argument("--sh", type=int, default=0, help="degree of spherical harmonic (2, 3)")
    parser.add_argument("--pe", type=int, default=0, help="positional encoding (0 - 8)")
    parser.add_argument("--n_pts_per_ray", type=int, default=400, help="Sampling points per ray")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--img_shape", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--vol_shape", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--fov_depth", type=int, default=256, help="raymarch num of the tensor")
    parser.add_argument("--epochs", type=int, default=301, help="number of epochs")
    parser.add_argument("--train_samples", type=int, default=1000, help="training samples")
    parser.add_argument("--val_samples", type=int, default=400, help="validation samples")
    parser.add_argument("--test_samples", type=int, default=400, help="test samples")
    parser.add_argument("--timesteps", type=int, default=180, help="timesteps for diffusion")
    parser.add_argument("--amp", action="store_true", help="train with mixed precision or not")
    parser.add_argument("--test", action="store_true", help="test")
    parser.add_argument("--lpips", action="store_true", help="train with lpips xray ct random")
    parser.add_argument("--strict", action="store_true", help="checkpoint loading")
    parser.add_argument("--perceptual2d", action="store_true", help="perceptual 2d loss")
    parser.add_argument("--perceptual3d", action="store_true", help="perceptual 3d loss")
    parser.add_argument("--perceptual", action="store_true", help="perceptual 2d/fake 3d loss")
    parser.add_argument("--ssim", action="store_true", help="ssim")
    parser.add_argument("--resample", action="store_true", help="resample")
    parser.add_argument("--phase", type=str, default="direct", help="direct|cyclic|paired")

    parser.add_argument("--alpha", type=float, default=1.0, help="vol loss")
    parser.add_argument("--gamma", type=float, default=1.0, help="img loss")
    parser.add_argument("--delta", type=float, default=1.0, help="vgg loss")
    parser.add_argument("--theta", type=float, default=1.0, help="cam loss")
    parser.add_argument("--omega", type=float, default=1.0, help="cam cond")
    parser.add_argument("--lamda", type=float, default=1.0, help="cam roto")
    parser.add_argument("--tfunc", action="store_true", help="tfunc function")

    parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--logsdir", type=str, default="logs", help="logging directory")
    parser.add_argument("--datadir", type=str, default="data", help="data directory")
    parser.add_argument("--strategy", type=str, default="auto", help="training strategy")
    parser.add_argument("--backbone", type=str, default="efficientnet-b7", help="Backbone for network")
    parser.add_argument("--prediction_type", type=str, default="sample", help="prediction_type for network")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")

    # parser = Trainer.add_argparse_args(parser)

    # Collect the hyper parameters
    hparams = parser.parse_args()

    # Seed the application
    seed_everything(42)

    # Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{hparams.logsdir}_{hparams.phase}",
        monitor="validation_loss_epoch",
        auto_insert_metric_name=True,
        save_top_k=-1,
        save_last=True,
        every_n_epochs=10,
    )
    lr_callback = LearningRateMonitor(logging_interval="step")

    # Logger
    tensorboard_logger = TensorBoardLogger(save_dir=f"{hparams.logsdir}_{hparams.phase}", log_graph=True)
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    early_stop_callback = EarlyStopping(
        monitor="validation_loss_epoch",  # The quantity to be monitored
        min_delta=0.00,  # Minimum change in the monitored quantity to qualify as an improvement
        patience=20,  # Number of epochs with no improvement after which training will be stopped
        verbose=True,  # Whether to print logs in stdout
        mode="min",  # In 'min' mode, training will stop when the quantity monitored has stopped decreasing
    )
    callbacks = [
        lr_callback,
        checkpoint_callback,
        early_stop_callback,
    ]
    if hparams.strategy != "fsdp":
        callbacks.append(swa_callback)
    # Init model with callbacks
    trainer = Trainer(
        accelerator=hparams.accelerator,
        devices=hparams.devices,
        max_epochs=hparams.epochs,
        logger=[tensorboard_logger],
        callbacks=callbacks,
        # accumulate_grad_batches=4,
        strategy="auto",  # hparams.strategy, #"auto", #"ddp_find_unused_parameters_true",
        precision=16 if hparams.amp else 32,
        # profiler="advanced",
    )

    # Create data module
    train_image3d_folders = [
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/NSCLC/processed/train/images"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MELA2022/raw/train/images"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MELA2022/raw/val/images"),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Imagenglab/processed/train/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/train/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/val/images'),
    ]

    train_label3d_folders = []

    train_image2d_folders = [
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/VinDr/v1/processed/train/images/"),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'),
    ]

    train_label2d_folders = []

    val_image3d_folders = [
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/NSCLC/processed/train/images"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MELA2022/raw/train/images"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MELA2022/raw/val/images"),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Imagenglab/processed/train/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/train/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/val/images'),
    ]

    val_image2d_folders = [
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/VinDr/v1/processed/test/images/"),
    ]

    test_image3d_folders = val_image3d_folders
    test_image2d_folders = val_image2d_folders

    datamodule = UnpairedDataModule(
        train_image3d_folders=train_image3d_folders,
        train_image2d_folders=train_image2d_folders,
        val_image3d_folders=val_image3d_folders,
        val_image2d_folders=val_image2d_folders,
        test_image3d_folders=test_image3d_folders,
        test_image2d_folders=test_image2d_folders,
        train_samples=hparams.train_samples,
        val_samples=hparams.val_samples,
        test_samples=hparams.test_samples,
        batch_size=hparams.batch_size,
        img_shape=hparams.img_shape,
        vol_shape=hparams.vol_shape,
    )
    datamodule.setup(seed=hparams.seed)

    # Comment here
    # Example
    #
    # hparams = parser.parse_args()
    # # Create data module
    # train_image3d_folders = [os.path.join(hparams.datadir, "Visualization/image3d"),]
    # train_image2d_folders = [os.path.join(hparams.datadir, "Visualization/image2d"),]
    # val_image3d_folders = [os.path.join(hparams.datadir, "Visualization/image3d"),]
    # val_image2d_folders = [os.path.join(hparams.datadir, "Visualization/image2d"),]
    # test_image3d_folders = [os.path.join(hparams.datadir, "Visualization/image3d"),]
    # test_image2d_folders = [os.path.join(hparams.datadir, "Visualization/image2d"),]

    # datamodule = ExampleDataModule(
    #     train_image3d_folders=train_image3d_folders,
    #     train_image2d_folders=train_image2d_folders,
    #     val_image3d_folders=val_image3d_folders,
    #     val_image2d_folders=val_image2d_folders,
    #     test_image3d_folders=test_image3d_folders,
    #     test_image2d_folders=test_image2d_folders,
    #     img_shape=hparams.img_shape,
    #     vol_shape=hparams.vol_shape,
    #     batch_size=hparams.batch_size,
    # )
    # datamodule.setup(seed=hparams.seed)
    
    ####### Test camera mu and bandwidth ########
    # test_random_uniform_cameras(hparams, datamodule)
    #############################################

    model = NVMLightningModule(hparams=hparams)
    # model = model.load_from_checkpoint(hparams.ckpt, strict=False) if hparams.ckpt is not None else model
    
    if hparams.test:
        trainer.test(
            model,
            dataloaders=datamodule.test_dataloader(),
            ckpt_path=hparams.ckpt
        )

    else:
        trainer.fit(
            model,
            # compiled_model,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader(),
            # datamodule=datamodule,
            ckpt_path=hparams.ckpt if hparams.ckpt is not None and hparams.strict else None,  # "some/path/to/my_checkpoint.ckpt"
        )

    # serve
