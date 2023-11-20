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
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger

from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from argparse import ArgumentParser
from tqdm.auto import tqdm

from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch3d.renderer.cameras import (
    FoVPerspectiveCameras,
    look_at_view_transform,
)

from monai.losses import PerceptualLoss
from monai.networks.nets import Unet
from monai.networks.layers.factories import Norm

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

from datamodule import UnpairedDataModule, ExampleDataModule
from dvr.renderer import ReverseXRayVolumeRenderer

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
    # "efficientnet-b9": (64, 128, 256, 512, 1024),
    "efficientnet-l2": (72, 104, 176, 480, 1376),
}

class InverseXrayVolumeRenderer(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        img_shape=256,
        vol_shape=256,
        fov_depth=256,
        fwd_renderer=None,
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
        self.fwd_renderer = fwd_renderer
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
        
        self.net3d3d = nn.Sequential(
            Unet(
                spatial_dims=3, 
                in_channels=1, 
                out_channels=1, 
                channels=backbones[backbone], 
                strides=(2, 2, 2, 2, 2), 
                num_res_units=2, 
                kernel_size=3, 
                up_kernel_size=3, 
                act=("LeakyReLU", {"inplace": True}), 
                norm=Norm.BATCH,
                # dropout=0.5
            ),
        )
        
    def forward(self, image2d, cameras, timesteps=None, is_training=False):
        _device = image2d.device
        batch = image2d.shape[0]
        dtype = image2d.dtype
        if timesteps is None:
            timesteps = torch.zeros((batch), device=_device).long()
        
        R = cameras.R
        T = torch.zeros_like(cameras.T.unsqueeze_(-1))
        
        mat = torch.cat([R, T], dim=-1)
        inv = torch.cat([torch.inverse(R), -T], dim=-1)
        
        mid = self.net2d3d(
            x=image2d,
            context=mat.reshape(batch, 1, -1),
            timesteps=timesteps,
        ).view(-1, 1, self.fov_depth, self.img_shape, self.img_shape)
        # mid = self.net2d3d(image2d).view(-1, 1, self.fov_depth, self.img_shape, self.img_shape)

        grd = F.affine_grid(inv, mid.size()).type(dtype)
        
        mid_resample = F.grid_sample(mid, grd)
        
        # if is_training:
        #     # Randomly return out_resample or out_explicit
        #     rng = torch.rand(1).item()
        #     if rng > 0.5:
        #         out = self.net3d3d(mid)
        #         out_resample = F.grid_sample(out, grd)
        #     else:
        #         out_resample = self.net3d3d(mid_resample)
        # else:
        #     out_resample = self.net3d3d(mid_resample)
        out_resample = self.net3d3d(mid_resample)
        return out_resample, mid_resample

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
    
        self.timesteps = hparams.timesteps

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
            min_depth=3.0, 
            max_depth=9.0, 
            ndc_extent=1.0,
            tffunction=True,
            data_dim=16000,
            feat_dim=1,
        )

        self.inv_renderer = InverseXrayVolumeRenderer(
            in_channels=1, 
            out_channels=self.sh ** 2 if self.sh > 0 else 1, 
            vol_shape=self.vol_shape, 
            img_shape=self.img_shape, 
            sh=self.sh, 
            pe=self.pe, 
            backbone=self.backbone,
            fwd_renderer=self.fwd_renderer,
        )

        if self.ckpt:
            print("Loading.. ", self.ckpt)
            checkpoint = torch.load(self.ckpt, map_location=torch.device("cpu"))["state_dict"]
            state_dict = {k: v for k, v in checkpoint.items() if k in self.state_dict()}
            self.load_state_dict(state_dict, strict=self.strict)

        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.maeloss = nn.L1Loss(reduction="mean")
        # self.pctloss = PerceptualLoss(
        #     spatial_dims=2, 
        #     network_type="radimagenet_resnet50", 
        #     # network_type="resnet50", 
        #     # network_type="medicalnet_resnet50_23datasets", 
        #     is_fake_3d=False, 
        #     pretrained=True,
        # )

    def forward_screen(self, image3d, cameras):
        return self.fwd_renderer(image3d, cameras)

    def forward_volume(self, image2d, cameras, n_views=[2, 1], has_middle=False, timesteps=None, is_training=False):
        _device = image2d.device
        B = image2d.shape[0]
        assert B == sum(n_views)  # batch must be equal to number of projections
        results, middles = self.inv_renderer(image2d, cameras, timesteps, is_training)
        if has_middle:
            return results, middles
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
        view_random = make_cameras_dea(dist_random, elev_random, azim_random, fov=20, znear=3, zfar=9)

        dist_hidden = 6.0 * torch.ones(self.batch_size, device=_device)
        elev_hidden = torch.zeros(self.batch_size, device=_device)
        azim_hidden = torch.zeros(self.batch_size, device=_device)
        view_hidden = make_cameras_dea(dist_hidden, elev_hidden, azim_hidden, fov=20, znear=3, zfar=9)

        # Construct the samples in 2D
        figure_xr_hidden = image2d
        figure_ct_random = self.forward_screen(image3d=image3d, cameras=view_random)
        figure_ct_hidden = self.forward_screen(image3d=image3d, cameras=view_hidden)
        
        # Reconstruct the Encoder-Decoder
        volume_dx_concat, middle_dx_concat = self.forward_volume(
            image2d=torch.cat([figure_xr_hidden, figure_ct_random, figure_ct_hidden]), 
            cameras=join_cameras_as_batch([view_hidden, view_random, view_hidden]), 
            n_views=[1, 1, 1]*batchsz,
            has_middle=True, 
            is_training=(stage=="train"),
        )
        volume_xr_hidden_inverse, volume_ct_random_inverse, volume_ct_hidden_inverse = torch.split(volume_dx_concat, batchsz)
        middle_xr_hidden_inverse, middle_ct_random_inverse, middle_ct_hidden_inverse = torch.split(middle_dx_concat, batchsz)

        with torch.no_grad():
            figure_xr_hidden_inverse_random = self.forward_screen(image3d=volume_xr_hidden_inverse, cameras=view_random)
        figure_xr_hidden_inverse_hidden = self.forward_screen(image3d=volume_xr_hidden_inverse, cameras=view_hidden)
        figure_ct_random_inverse_random = self.forward_screen(image3d=volume_ct_random_inverse, cameras=view_random)
        figure_ct_random_inverse_hidden = self.forward_screen(image3d=volume_ct_random_inverse, cameras=view_hidden)
        figure_ct_hidden_inverse_random = self.forward_screen(image3d=volume_ct_hidden_inverse, cameras=view_random)
        figure_ct_hidden_inverse_hidden = self.forward_screen(image3d=volume_ct_hidden_inverse, cameras=view_hidden)

        if self.sh > 0:
            volume_xr_hidden_inverse = volume_xr_hidden_inverse.sum(dim=1, keepdim=True)
            volume_ct_random_inverse = volume_ct_random_inverse.sum(dim=1, keepdim=True)
            volume_ct_hidden_inverse = volume_ct_hidden_inverse.sum(dim=1, keepdim=True)

        # Visualization step
        if batch_idx == 0:
            zeros = torch.zeros_like(image2d)
            viz2d = torch.cat([
                torch.cat([
                    image2d, 
                    volume_xr_hidden_inverse[..., self.vol_shape // 2, :], 
                    figure_xr_hidden_inverse_random, 
                    figure_xr_hidden_inverse_hidden, 
                    image3d[..., self.vol_shape // 2, :], 
                    figure_ct_random, 
                    figure_ct_hidden,
                ], dim=-2).transpose(2, 3),
                torch.cat([
                    zeros,
                    volume_ct_random_inverse[..., self.vol_shape // 2, :],
                    figure_ct_random_inverse_random,
                    figure_ct_random_inverse_hidden,
                    volume_ct_hidden_inverse[..., self.vol_shape // 2, :],
                    figure_ct_hidden_inverse_random,
                    figure_ct_hidden_inverse_hidden,
                ], dim=-2).transpose(2, 3),
            ], dim=-2)
            tensorboard = self.logger.experiment
            grid2d = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=True, nrow=1, padding=0)
            tensorboard.add_image(f"{stage}_df_samples", grid2d, self.current_epoch * self.batch_size + batch_idx)

        # if self.phase == "direct":
        #     im3d_loss_inv = self.maeloss(volume_ct_hidden_inverse, image3d)
        #     im3d_loss = im3d_loss_inv
        #     self.log(f"{stage}_im3d_loss", im3d_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
        #     loss = self.alpha * im3d_loss
        if self.phase == "ctonly":
            im3d_loss_inv = self.maeloss(volume_ct_hidden_inverse, image3d) \
                          + self.maeloss(middle_ct_hidden_inverse, image3d) \
                          + self.maeloss(volume_ct_random_inverse, image3d) \
                          + self.maeloss(middle_ct_random_inverse, image3d) 
            im3d_loss = im3d_loss_inv
            self.log(f"{stage}_im3d_loss", im3d_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
            im2d_loss_inv = self.maeloss(figure_ct_hidden_inverse_hidden, figure_ct_hidden) \
                          + self.maeloss(figure_ct_hidden_inverse_random, figure_ct_random) \
                          + self.maeloss(figure_ct_random_inverse_hidden, figure_ct_hidden) \
                          + self.maeloss(figure_ct_random_inverse_random, figure_ct_random) 
            im2d_loss = im2d_loss_inv
            self.log(f"{stage}_im2d_loss", im2d_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
            loss = self.alpha * im3d_loss + self.gamma * im2d_loss
        if self.phase == "ctxray":
            im3d_loss_inv = self.maeloss(volume_ct_hidden_inverse, image3d) \
                          + self.maeloss(middle_ct_hidden_inverse, image3d) \
                          + self.maeloss(volume_ct_random_inverse, image3d) \
                          + self.maeloss(middle_ct_random_inverse, image3d) 
            im3d_loss = im3d_loss_inv
            self.log(f"{stage}_im3d_loss", im3d_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
            im2d_loss_inv = self.maeloss(figure_ct_hidden_inverse_hidden, figure_ct_hidden) \
                          + self.maeloss(figure_ct_hidden_inverse_random, figure_ct_random) \
                          + self.maeloss(figure_ct_random_inverse_hidden, figure_ct_hidden) \
                          + self.maeloss(figure_ct_random_inverse_random, figure_ct_random) \
                          + self.maeloss(figure_xr_hidden_inverse_hidden, figure_xr_hidden) # Check here
                          
            im2d_loss = im2d_loss_inv
            self.log(f"{stage}_im2d_loss", im2d_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
            loss = self.alpha * im3d_loss + self.gamma * im2d_loss
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
            # [
            #     # {"params": self.fwd_renderer.parameters()},
            #     {"params": self.inv_renderer.parameters()},
            #     # {'params': self.unet2d_model.parameters()}, # Add diffusion model, remove lpips model
            # ],
            # lr=self.lr,
            # betas=(0.9, 0.999)
            self.parameters(), lr=self.lr, betas=(0.9, 0.999)
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
    parser.add_argument("--phase", type=str, default="direct", help="direct|cyclic|paired")

    parser.add_argument("--alpha", type=float, default=1.0, help="vol loss")
    parser.add_argument("--gamma", type=float, default=1.0, help="img loss")
    parser.add_argument("--delta", type=float, default=1.0, help="vgg loss")
    parser.add_argument("--theta", type=float, default=1.0, help="cam loss")
    parser.add_argument("--omega", type=float, default=1.0, help="cam cond")
    parser.add_argument("--lamda", type=float, default=1.0, help="cam roto")

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
        dirpath=f"{hparams.logsdir}",
        monitor="validation_loss_epoch",
        auto_insert_metric_name=True,
        save_top_k=-1,
        save_last=True,
        every_n_epochs=10,
    )
    lr_callback = LearningRateMonitor(logging_interval="step")

    # Logger
    tensorboard_logger = TensorBoardLogger(save_dir=f"{hparams.logsdir}", log_graph=True)
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    callbacks = [
        lr_callback,
        checkpoint_callback,
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
        accumulate_grad_batches=4,
        strategy="auto",  # hparams.strategy, #"auto", #"ddp_find_unused_parameters_true",
        precision=16 if hparams.amp else 32,
        profiler="advanced",
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
