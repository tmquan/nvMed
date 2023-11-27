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

from typing import Any, Callable, Dict, Optional, Tuple, List
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

from datamodule import UnpairedDataModule, ExampleDataModule
# from dvr.renderer import ReverseXRayVolumeRenderer


def minimized(x, eps=1e-8):
    return (x + eps) / (x.max() + eps)

def normalized(x, eps=1e-8):
    return (x - x.min() + eps) / (x.max() - x.min() + eps)

def standardized(x, eps=1e-8):
    return (x - x.mean()) / (x.std() + eps)  # 1e-6 to avoid zero division

def equalized(x, eps=1e-8):
    return equalize(x)

def make_cameras_dea(
    dist: torch.Tensor, 
    elev: torch.Tensor, 
    azim: torch.Tensor, 
    fov: int = 40, 
    znear: int = 4.0, 
    zfar: int = 8.0, 
    is_orthogonal: bool = False
):
    assert dist.device==elev.device==azim.device
    _device = dist.device
    R, T = look_at_view_transform(dist=dist.float(), elev=elev.float() * 90, azim=azim.float() * 180)
    if is_orthogonal:
        return FoVOrthographicCameras(R=R, T=T, znear=znear, zfar=zfar).to(_device)
    return FoVPerspectiveCameras(R=R, T=T, fov=fov, znear=znear, zfar=zfar).to(_device)

from omegaconf import OmegaConf
from PIL import Image
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.implicitron.dataset.utils import DATASET_TYPE_KNOWN, DATASET_TYPE_UNKNOWN
from pytorch3d.implicitron.dataset.rendered_mesh_dataset_map_provider import RenderedMeshDatasetMapProvider

from pytorch3d.implicitron.models.generic_model import GenericModel
from pytorch3d.implicitron.models.implicit_function.base import ImplicitFunctionBase, ImplicitronRayBundle
from pytorch3d.implicitron.models.renderer.raymarcher import AccumulativeRaymarcherBase, RaymarcherBase
from pytorch3d.implicitron.models.renderer.base import BaseRenderer, RendererOutput, EvaluationMode, ImplicitFunctionWrapper
from pytorch3d.implicitron.models.renderer.multipass_ea import MultiPassEmissionAbsorptionRenderer
from pytorch3d.implicitron.models.renderer.ray_point_refiner import RayPointRefiner
from pytorch3d.implicitron.tools.config import get_default_args, registry, remove_unused_components, run_auto_creation
from pytorch3d.renderer.implicit.renderer import VolumeSampler
from pytorch3d.vis.plotly_vis import plot_batch_individually, plot_scene
   
# from pytorch3d.implicitron.models.renderer.base import BaseRenderer
from pytorch3d.structures import Volumes
from pytorch3d.renderer import (
    VolumeRenderer,
    NDCMultinomialRaysampler,
)
from pytorch3d.renderer.implicit.raymarching import (
    _check_density_bounds,
    _check_raymarcher_inputs,
    _shifted_cumprod,
)

@registry.register
class AbsorptionEmissionRaymarcher(  # pyre-ignore: 13
    AccumulativeRaymarcherBase, torch.nn.Module
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __post_init__(self):
        """
        Args:
            surface_thickness: Denotes the overlap between the absorption
                function and the density function.
        """
        bg_color = torch.tensor(self.bg_color)
        if bg_color.ndim != 1:
            raise ValueError(f"bg_color (shape {bg_color.shape}) should be a 1D tensor")

        self.register_buffer("_bg_color", bg_color, persistent=False)

        self._capping_function: Callable[[_TTensor], _TTensor] = {
            "exponential": lambda x: 1.0 - torch.exp(-x),
            "cap1": lambda x: x.clamp(max=1.0),
        }[self.capping_function_type]

        self._weight_function: Callable[[_TTensor, _TTensor], _TTensor] = {
            "product": lambda curr, acc: curr * acc,
            "minimum": lambda curr, acc: torch.minimum(curr, acc),
        }[self.weight_function_type]

    # pyre-fixme[14]: `forward` overrides method defined in `RaymarcherBase`
    #  inconsistently.
    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: torch.Tensor,
        aux: Dict[str, Any],
        ray_lengths: torch.Tensor,
        ray_deltas: Optional[torch.Tensor] = None,
        density_noise_std: float = 0.0,
        **kwargs,
    ) -> RendererOutput:
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray, 1)`.
            rays_features: Per-ray feature values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            aux: a dictionary with extra information.
            ray_lengths: Per-ray depth values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            ray_deltas: Optional differences between consecutive elements along the ray bundle
                represented with a tensor of shape `(..., n_points_per_ray)`. If None,
                these differences are computed from ray_lengths.
            density_noise_std: the magnitude of the noise added to densities.

        Returns:
            features: A tensor of shape `(..., feature_dim)` containing
                the rendered features for each ray.
            depth: A tensor of shape `(..., 1)` containing estimated depth.
            opacities: A tensor of shape `(..., 1)` containing rendered opacities.
            weights: A tensor of shape `(..., n_points_per_ray)` containing
                the ray-specific non-negative opacity weights. In general, they
                don't sum to 1 but do not overcome it, i.e.
                `(weights.sum(dim=-1) <= 1.0).all()` holds.
        """
        _check_raymarcher_inputs(
            rays_densities,
            rays_features,
            ray_lengths,
            z_can_be_none=True,
            features_can_be_none=False,
            density_1d=True,
        )

        if ray_deltas is None:
            ray_lengths_diffs = torch.diff(ray_lengths, dim=-1)
            if self.replicate_last_interval:
                last_interval = ray_lengths_diffs[..., -1:]
            else:
                last_interval = torch.full_like(
                    ray_lengths[..., :1], self.background_opacity
                )
            deltas = torch.cat((ray_lengths_diffs, last_interval), dim=-1)
        else:
            deltas = ray_deltas

        rays_densities = rays_densities[..., 0]

        if density_noise_std > 0.0:
            noise: _TTensor = torch.randn_like(rays_densities).mul(density_noise_std)
            rays_densities = rays_densities + noise
        if self.density_relu:
            rays_densities = torch.relu(rays_densities)

        weighted_densities = deltas * rays_densities
        capped_densities = self._capping_function(weighted_densities)

        rays_opacities = self._capping_function(
            torch.cumsum(weighted_densities, dim=-1)
        )
        opacities = rays_opacities[..., -1:]
        # absorption_shifted = (-rays_opacities + 1.0).roll(
        #     self.surface_thickness, dims=-1
        # )
        # absorption_shifted[..., : self.surface_thickness] = 1.0
        absorption_shifted = _shifted_cumprod((1.0 + eps) - rays_opacities.flip(dims=(-1,)), shift=-self.surface_thickness).flip(dims=(-1,))  
        weights = self._weight_function(capped_densities, absorption_shifted)
        features = (weights[..., None] * rays_features).sum(dim=-2)
        depth = (weights * ray_lengths)[..., None].sum(dim=-2)

        alpha = opacities if self.blend_output else 1
        if self._bg_color.shape[-1] not in [1, features.shape[-1]]:
            raise ValueError("Wrong number of background color channels.")
        features = alpha * features + (1 - opacities) * self._bg_color

        return RendererOutput(
            features=features,
            depths=depth,
            masks=opacities,
            weights=weights,
            aux=aux,
        )
        
    # def forward(self, rays_densities: torch.Tensor, rays_features: torch.Tensor, eps: float = 1e-10, **kwargs,) -> torch.Tensor:
    #     _check_raymarcher_inputs(
    #         rays_densities, rays_features, None, z_can_be_none=True, features_can_be_none=False, density_1d=True,
    #     )
    #     _check_density_bounds(rays_densities)
    #     rays_densities = rays_densities[..., 0]
        
    #     # absorption = _shifted_cumprod(
    #     #     (1.0 + eps) - rays_densities, shift=self.surface_thickness
    #     # )
    #     # weights = rays_densities * absorption
    #     # features = (weights[..., None] * rays_features).sum(dim=-2)
    #     # opacities = 1.0 - torch.prod(1.0 - rays_densities, dim=-1, keepdim=True)
        
    #     # Reverse the direction of the absorption to match X-ray detector
    #     absorption = _shifted_cumprod((1.0 + eps) - rays_densities.flip(dims=(-1,)), shift=-self.surface_thickness).flip(dims=(-1,))  
    #     weights = rays_densities * absorption
    #     features = (weights[..., None] * rays_features).sum(dim=-2)
    #     opacities = 1.0 - torch.prod(1.0 - rays_densities, dim=-1, keepdim=True)
    #     return torch.cat((features, opacities), dim=-1)

@registry.register
class SinglePassEmissionAbsorptionRenderer(  # pyre-ignore: 13
    MultiPassEmissionAbsorptionRenderer, torch.nn.Module
):
    """
    Implements the multi-pass rendering function, in particular,
    with emission-absorption ray marching used in NeRF [1]. First, it evaluates
    opacity-based ray-point weights and then optionally (in case more implicit
    functions are given) resamples points using importance sampling and evaluates
    new weights.

    During each ray marching pass, features, depth map, and masks
    are integrated: Let o_i be the opacity estimated by the implicit function,
    and d_i be the offset between points `i` and `i+1` along the respective ray.
    Ray marching is performed using the following equations::

        ray_opacity_n = cap_fn(sum_i=1^n cap_fn(d_i * o_i)),
        weight_n = weight_fn(cap_fn(d_i * o_i), 1 - ray_opacity_{n-1}),

    and the final rendered quantities are computed by a dot-product of ray values
    with the weights, e.g. `features = sum_n(weight_n * ray_features_n)`.

    By default, for the EA raymarcher from [1] (
        activated with `self.raymarcher_class_type="EmissionAbsorptionRaymarcher"`
    )::

        cap_fn(x) = 1 - exp(-x),
        weight_fn(x) = w * x.

    Note that the latter can altered by changing `self.raymarcher_class_type`,
    e.g. to "CumsumRaymarcher" which implements the cumulative-sum raymarcher
    from NeuralVolumes [2].

    Settings:
        n_pts_per_ray_fine_training: The number of points sampled per ray for the
            fine rendering pass during training.
        n_pts_per_ray_fine_evaluation: The number of points sampled per ray for the
            fine rendering pass during evaluation.
        stratified_sampling_coarse_training: Enable/disable stratified sampling in the
            refiner during training. Only matters if there are multiple implicit
            functions (i.e. in GenericModel if num_passes>1).
        stratified_sampling_coarse_evaluation: Enable/disable stratified sampling in
            the refiner during evaluation. Only matters if there are multiple implicit
            functions (i.e. in GenericModel if num_passes>1).
        append_coarse_samples_to_fine: Add the fine ray points to the coarse points
            after sampling.
        density_noise_std_train: Standard deviation of the noise added to the
            opacity field.
        return_weights: Enables returning the rendering weights of the EA raymarcher.
            Setting to `True` can lead to a prohibitivelly large memory consumption.
        blurpool_weights: Use blurpool defined in [3], on the input weights of
            each implicit_function except the first (implicit_functions[0]).
        sample_pdf_eps: Padding applied to the weights (alpha in equation 18 of [3]).
        raymarcher_class_type: The type of self.raymarcher corresponding to
            a child of `RaymarcherBase` in the registry.
        raymarcher: The raymarcher object used to convert per-point features
            and opacities to a feature render.

    References:
        [1] Mildenhall, Ben, et al. "Nerf: Representing Scenes as Neural Radiance
            Fields for View Synthesis." ECCV 2020.
        [2] Lombardi, Stephen, et al. "Neural Volumes: Learning Dynamic Renderable
            Volumes from Images." SIGGRAPH 2019.
        [3] Jonathan T. Barron, et al. "Mip-NeRF: A Multiscale Representation
            for Anti-Aliasing Neural Radiance Fields." ICCV 2021.

    """

    raymarcher_class_type: str = "AbsorptionEmissionRaymarcher"
    raymarcher: RaymarcherBase

    n_pts_per_ray_fine_training: int = 256
    n_pts_per_ray_fine_evaluation: int = 256
    stratified_sampling_coarse_training: bool = True
    stratified_sampling_coarse_evaluation: bool = False
    append_coarse_samples_to_fine: bool = True
    density_noise_std_train: float = 0.0
    return_weights: bool = False
    blurpool_weights: bool = False
    sample_pdf_eps: float = 1e-5

    def __post_init__(self):
        self._refiners = {
            EvaluationMode.TRAINING: RayPointRefiner(
                n_pts_per_ray=self.n_pts_per_ray_fine_training,
                random_sampling=self.stratified_sampling_coarse_training,
                add_input_samples=self.append_coarse_samples_to_fine,
                blurpool_weights=self.blurpool_weights,
                sample_pdf_eps=self.sample_pdf_eps,
            ),
            EvaluationMode.EVALUATION: RayPointRefiner(
                n_pts_per_ray=self.n_pts_per_ray_fine_evaluation,
                random_sampling=self.stratified_sampling_coarse_evaluation,
                add_input_samples=self.append_coarse_samples_to_fine,
                blurpool_weights=self.blurpool_weights,
                sample_pdf_eps=self.sample_pdf_eps,
            ),
        }
        run_auto_creation(self)

    def forward(
        self,
        ray_bundle: ImplicitronRayBundle,
        implicit_functions: List[ImplicitFunctionWrapper],
        evaluation_mode: EvaluationMode = EvaluationMode.EVALUATION,
        **kwargs,
    ) -> RendererOutput:
        """
        Args:
            ray_bundle: A `ImplicitronRayBundle` object containing the parametrizations of the
                sampled rendering rays.
            implicit_functions: List of ImplicitFunctionWrappers which
                define the implicit functions to be used sequentially in
                the raymarching step. The output of raymarching with
                implicit_functions[n-1] is refined, and then used as
                input for raymarching with implicit_functions[n].
            evaluation_mode: one of EvaluationMode.TRAINING or
                EvaluationMode.EVALUATION which determines the settings used for
                rendering

        Returns:
            instance of RendererOutput
        """
        if not implicit_functions:
            raise ValueError("EA renderer expects implicit functions")

        return self._run_raymarcher(
            ray_bundle,
            implicit_functions,
            None,
            evaluation_mode,
        )

    def _run_raymarcher(
        self, ray_bundle, implicit_functions, prev_stage, evaluation_mode
    ):
        density_noise_std = (
            self.density_noise_std_train
            if evaluation_mode == EvaluationMode.TRAINING
            else 0.0
        )

        ray_deltas = (
            None if ray_bundle.bins is None else torch.diff(ray_bundle.bins, dim=-1)
        )
        output = self.raymarcher(
            *implicit_functions[0](ray_bundle=ray_bundle),
            ray_lengths=ray_bundle.lengths,
            ray_deltas=ray_deltas,
            density_noise_std=density_noise_std,
        )
        output.prev_stage = prev_stage

        weights = output.weights
        if not self.return_weights:
            output.weights = None

        # we may need to make a recursive call
        if len(implicit_functions) > 1:
            fine_ray_bundle = self._refiners[evaluation_mode](ray_bundle, weights)
            output = self._run_raymarcher(
                fine_ray_bundle,
                implicit_functions[1:],
                output,
                evaluation_mode,
            )

        return output
        
class ReverseXRayVolumeRenderer(BaseRenderer, nn.Module):
    def __init__(
        self, 
        image_width: int = 256, 
        image_height: int = 256, 
        n_pts_per_ray: int = 320, 
        min_depth: float = 3.0, 
        max_depth: float = 9.0, 
        ndc_extent: float = 1.0, 
        data_dim: int = 1024,
        feat_dim: int = 3,
        stratified_sampling: bool = False,
    ):
        super().__init__()
        self.n_pts_per_ray = n_pts_per_ray
        from dvr.raymarcher import AbsorptionEmissionRaymarcher as DirectAbsorptionEmissionRaymarcher
        self.raymarcher = DirectAbsorptionEmissionRaymarcher()  # FrontToBack
        self.raysampler = NDCMultinomialRaysampler(image_width=image_width, 
                                                   image_height=image_height, 
                                                   n_pts_per_ray=n_pts_per_ray, 
                                                   min_depth=min_depth, 
                                                   max_depth=max_depth, 
                                                   stratified_sampling=stratified_sampling,)
        self.renderer = VolumeRenderer(raysampler=self.raysampler, raymarcher=self.raymarcher,)
        self.ndc_extent = ndc_extent
        
    def forward(self, 
        image3d, 
        cameras, 
        opacity=None, 
        norm_type="minimized", 
        scaling_factor=0.1, 
        is_grayscale=True, 
        return_bundle=False
    ) -> torch.Tensor:
        
        features = image3d.repeat(1, 3, 1, 1, 1) if image3d.shape[1]==1 else image3d
        if opacity is None:
            densities = torch.ones_like(image3d[:, [0]]) * scaling_factor
        else:
            densities = opacity * scaling_factor
        # print(image3d.shape, densities.shape)
        shape = max(image3d.shape[1], image3d.shape[2])
        self.volumes = Volumes(
            features=features,
            densities=densities,
            voxel_size=2.0*float(self.ndc_extent)/shape,
            # volume_translation = [-0.5, -0.5, -0.5],
        )
        
        screen_RGBA, bundle = self.renderer(cameras=cameras, volumes=self.volumes)  # [...,:3]

        screen_RGBA = screen_RGBA.permute(0, 3, 2, 1)  # 3 for NeRF
        if is_grayscale:
            screen_RGB = screen_RGBA[:, :].mean(dim=1, keepdim=True)
        else:
            screen_RGB = screen_RGBA[:, :]

        if norm_type=="minimized":
            screen_RGB = minimized(screen_RGB)
        elif norm_type=="normalized":
            screen_RGB = normalized(screen_RGB)
        elif norm_type=="standardized":
            screen_RGB = normalized(standardized(screen_RGB))

        if return_bundle:
            return screen_RGB, bundle
        return screen_RGB

class InverseXrayVolumeRenderer(ImplicitFunctionBase, nn.Module):
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
        
        self.gm = GenericModel(
            num_passes=1,
            render_features_dimensions=3,
            # sampling_mode_training="fullgrid", 
            # sampling_mode_evaluation="fullgrid", 
            # implicit_function_class_type="VoxelGridImplicitFunction",
            # voxel_grid_class_type = "FullResolutionVoxelGrid",
            # voxel_grid_density_args = {
            #     "extents": (1.0, 1.0, 1.0), 
            #     "translation": (0.0, 0.0, 0.0),
            # },
            # raysampler_class_type = "NDCMultinomialRaysampler",
            # raysampler_class_type = "AdaptiveRaySampler",
            # raysampler_AdaptiveRaySampler_args = {"scene_extent": 4.0}
            raysampler_AdaptiveRaySampler_args = {
                "scene_extent": 1.0,
                "n_pts_per_ray_training": self.fov_depth, 
                "n_pts_per_ray_evaluation": self.fov_depth, 
            },
            # renderer_class_type = "SinglePassEmissionAbsorptionRenderer",
            # renderer_MultiPassEmissionAbsorptionRenderer_args = {
            #     "raymarcher_class_type": "AbsorptionEmissionRaymarcher",
            #     # "raymarcher_AbsorptionEmissionRaymarcher_args": {},
            # },
            render_image_height=self.img_shape,
            render_image_width=self.img_shape,
            image_feature_extractor_class_type="ResNetFeatureExtractor",
            image_feature_extractor_ResNetFeatureExtractor_args = {
                "add_images": True,
                "add_masks": True,
                "first_max_pool": False,
                "image_rescale": 0.32,
                "l2_norm": True,
                "name": "resnet101",
                "normalize_image": True,
                "pretrained": True,
                "stages": (1, 2, 3, 4),
                "proj_dim": 16,
            },
            loss_weights = { 
                "loss_rgb_mse": 1.0,
                "loss_rgb_huber": 0.0,
                "loss_mask_bce": 0.0,
            },
        )

        # In this case we can get the equivalent DictConfig cfg object to the way gm is configured as follows
        cfg = OmegaConf.structured(self.gm)
        # We can display the configuration in use as follows.
        remove_unused_components(cfg)
        yaml = OmegaConf.to_yaml(cfg, sort_keys=False)
        # Specify the file path
        file_path = "config.yaml"
        # Write the YAML content to the file
        with open(file_path, "w") as file:
            file.write(yaml)
            
    def forward(self, frames, evaluation_mode=EvaluationMode.EVALUATION):
        return self.gm(**frames, evaluation_mode=evaluation_mode)

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
        self.perceptual2d = hparams.perceptual2d
        self.perceptual3d = hparams.perceptual3d
        
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
            min_depth=2.0, 
            max_depth=10.0, 
            ndc_extent=1.0,
            data_dim=4096,
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
        )
        # init_weights(self.inv_renderer, init_type="xavier")
        
        if self.ckpt:
            print("Loading.. ", self.ckpt)
            checkpoint = torch.load(self.ckpt, map_location=torch.device("cpu"))["state_dict"]
            state_dict = {k: v for k, v in checkpoint.items() if k in self.state_dict()}
            self.load_state_dict(state_dict, strict=self.strict)

        self.train_step_outputs = []
        self.validation_step_outputs = []

    def forward_screen(self, image3d, cameras):
        return self.fwd_renderer(image3d, cameras)

    def _set_framedata(self, image2d, cameras, number=None, name=None, category=None):
        if image2d.shape[1] == 1:
            image2d = image2d.repeat(1, 3, 1, 1)
        batchsz = image2d.shape[0]
        opacity = torch.ones_like(image2d[:,[0]])
        framedata = FrameData(
            frame_number=number,
            sequence_name=name,
            sequence_category=category,
            camera=cameras,
            # pyre-ignore
            image_size_hw=torch.tensor(image2d.shape[-2:], dtype=torch.long),
            image_rgb=image2d,
            fg_probability=opacity,
            frame_type=[DATASET_TYPE_UNKNOWN]*batchsz,
        )
        return framedata
    
    def _collate_framedata(self, framedata1, framedata2):
        # Check if framedata1 and framedata2 have compatible shapes
        if framedata1.image_rgb.shape != framedata2.image_rgb.shape:
            raise ValueError("The image shapes of framedata1 and framedata2 must be the same.")

        # Concatenate the attributes of framedata1 and framedata2
        concatenated_framedata = FrameData(
            frame_number=[framedata1.frame_number, framedata2.frame_number],
            sequence_name=[framedata1.sequence_name, framedata2.sequence_name],
            sequence_category=[framedata1.sequence_category, framedata2.sequence_category],
            camera=join_cameras_as_batch([framedata1.camera, framedata2.camera]),
            image_size_hw=framedata1.image_size_hw,  # Assuming this is the same for both
            image_rgb=torch.cat([framedata1.image_rgb, framedata2.image_rgb], dim=0),
            fg_probability=torch.cat([framedata1.fg_probability, framedata2.fg_probability], dim=0),
            frame_type=framedata1.frame_type + framedata2.frame_type,
        )
        return concatenated_framedata
        
    def _common_step(self, batch, batch_idx, stage: Optional[str] = "evaluation"):
        image3d = batch["image3d"]
        image2d = batch["image2d"]
        _device = batch["image3d"].device
        batchsz = image2d.shape[0]

        image3d_pth = batch["image3d_pth"]
        image2d_pth = batch["image2d_pth"]
        image3d_idx = batch["image3d_idx"]
        image2d_idx = batch["image2d_idx"]
        # Construct the random cameras, -1 and 1 are the same point in azimuths
        dist_random = 6.0 * torch.ones(self.batch_size, device=_device)
        elev_random = torch.rand_like(dist_random) - 0.5
        azim_random = torch.rand_like(dist_random) * 2 - 1  # [0 1) to [-1 1)
        view_random = make_cameras_dea(dist_random, elev_random, azim_random, fov=20, znear=2, zfar=10)

        dist_hidden = 6.0 * torch.ones(self.batch_size, device=_device)
        elev_hidden = torch.zeros(self.batch_size, device=_device)
        azim_hidden = torch.zeros(self.batch_size, device=_device)
        view_hidden = make_cameras_dea(dist_hidden, elev_hidden, azim_hidden, fov=20, znear=2, zfar=10)

        # Construct the samples in 2D
        # with torch.no_grad():
        figure_xr_hidden = image2d
        figure_ct_random = self.forward_screen(image3d=image3d, cameras=view_random)
        # figure_ct_hidden = self.forward_screen(image3d=image3d, cameras=view_hidden)
        frames_xr_hidden = self._set_framedata(image2d=figure_xr_hidden, cameras=view_hidden, number=None, name=image2d_pth, category='xr')
        frames_ct_random = self._set_framedata(image2d=figure_ct_random, cameras=view_random, number=None, name=image3d_pth, category='ct')
        frames = self._collate_framedata(frames_xr_hidden, frames_ct_random)

        if stage=="train":
            output = self.inv_renderer.forward(frames, evaluation_mode=EvaluationMode.TRAINING)
        else:
            output = self.inv_renderer.forward(frames, evaluation_mode=EvaluationMode.EVALUATION)
            image_rgb = output["images_render"]
            estim_rgb = frames["image_rgb"]
            # For 'frames' dictionary
            print("Frames dictionary:")
            for key, value in frames.items():
                if isinstance(value, torch.Tensor):
                    print(f"Key: {key}, Shape: {value.shape}")

            # For 'output' dictionary
            print("\nOutput dictionary:")
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    print(f"Key: {key}, Shape: {value.shape}")
            import pdb; pdb.set_trace()
        return 0
        # cameras = join_cameras_as_batch([view_hidden, view_random, view_hidden])
        # figures = torch.cat([figure_xr_hidden, figure_ct_random, figure_ct_hidden]).repeat(1, 3, 1, 1)
        # opacity = torch.ones_like(torch.cat([figure_xr_hidden, figure_ct_random, figure_ct_hidden]))
        # frames = {cameras, figures,  opacity}
        # frames = FrameData(
        #     frame_number=None,
        #     sequence_name=None,
        #     sequence_category=None,
        #     camera=cameras,
        #     # pyre-ignore
        #     image_size_hw=torch.tensor(figures.shape[-2:], dtype=torch.long),
        #     image_rgb=figures,
        #     fg_probability=opacity,
        #     frame_type=[DATASET_TYPE_UNKNOWN]*batchsz,
        # )
        
        # if stage=="train":
        #     output = self.inv_renderer.forward(frames, evaluation_mode=EvaluationMode.TRAINING)
        # else:
        #     output = self.inv_renderer.forward(frames, evaluation_mode=EvaluationMode.EVALUATION)

            # # For 'frames' dictionary
            # print("Frames dictionary:")
            # for key, value in frames.items():
            #     if isinstance(value, torch.Tensor):
            #         print(f"Key: {key}, Shape: {value.shape}")

            # # For 'output' dictionary
            # print("\nOutput dictionary:")
            # for key, value in output.items():
            #     if isinstance(value, torch.Tensor):
            #         print(f"Key: {key}, Shape: {value.shape}")
                    
        #     import pdb; pdb.set_trace()
            
        #     image_rgb = output["images_render"][[0]]
        #     estim_rgb = frames["image_rgb"][[0]]
        #     # print(image3d.shape)
        #     # print(image_rgb.shape)
        #     # print(estim_rgb.shape)
        #     # Visualization step
        #     if batch_idx == 0:
        #         zeros = torch.zeros_like(image2d)
        #         viz2d = torch.cat([
        #             torch.cat([
        #                 # image3d[..., self.vol_shape // 2, :].repeat(1, 3, 1, 1), 
        #                 image_rgb, 
        #                 estim_rgb
        #             ], dim=-2).transpose(2, 3),
        #         ], dim=-2)
        #         tensorboard = self.logger.experiment
        #         grid2d = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=True, nrow=1, padding=0)
        #         tensorboard.add_image(f"{stage}_df_samples", grid2d, self.current_epoch * self.batch_size + batch_idx)
            
        # loss = output["objective"]
        # self.log(f"{stage}_loss", loss, on_step=(stage=="train"), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
        # return loss
        
        # # with torch.no_grad():
        # figure_xr_hidden_inverse_random = self.forward_screen(image3d=volume_xr_hidden_inverse, cameras=view_random)
        # figure_xr_hidden_inverse_hidden = self.forward_screen(image3d=volume_xr_hidden_inverse, cameras=view_hidden)
        # figure_ct_random_inverse_random = self.forward_screen(image3d=volume_ct_random_inverse, cameras=view_random)
        # figure_ct_random_inverse_hidden = self.forward_screen(image3d=volume_ct_random_inverse, cameras=view_hidden)
        # figure_ct_hidden_inverse_random = self.forward_screen(image3d=volume_ct_hidden_inverse, cameras=view_random)
        # figure_ct_hidden_inverse_hidden = self.forward_screen(image3d=volume_ct_hidden_inverse, cameras=view_hidden)
            
        # return loss

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
                {"params": self.inv_renderer.parameters()},
            ],
            lr=self.lr,
            betas=(0.9, 0.999)
            # 
        )
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
        return [optimizer], [scheduler]


if __name__=="__main__":
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
    parser.add_argument("--tfunc", action="store_true", help="learnable mapping")
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
        patience=10,  # Number of epochs with no improvement after which training will be stopped
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
    
    hparams = parser.parse_args()
    # Create data module
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
