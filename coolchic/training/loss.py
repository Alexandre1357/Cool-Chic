# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import typing
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Union

import torch
from torch import Tensor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from coolchic.io.format.yuv import DictTensorYUV
from coolchic.training.metrics import brdf
from coolchic.training.metrics.mse import dist_to_db, mse_fn
from coolchic.training.metrics.wasserstein import wasserstein_fn
from coolchic.utils import color, hammersley

DISTORTION_METRIC = Literal["mse", "l1", "wasserstein", "brdf07_mod_mse", "brdf09_mod_mse", "brdf07_l1", "brdf09_l1", "brdf07_l1_lpips", "brdf07_rel_mse"]


@dataclass(kw_only=True)
class LossFunctionOutput:
    """Output for FrameEncoder.loss_function"""

    # ----- This is the important output
    # Optional to allow easy inheritance by FrameEncoderLogs
    # but will never be None
    loss: Optional[float] = None  # The RD cost to optimize
    dist: Optional[float] = None  # The distorsion cost to optimize along with the rate
    rate_bpp: Optional[float] = None

    # Any other data required to compute some logs, stored inside a dictionary
    detailed_dist: Optional[Dict[DISTORTION_METRIC, float]] = (
        None  # Each distortion value (mse, wasserstein...)
    )
    rate_latent_bpp: Optional[float] = None  # Rate associated to the latent          [bpp]
    total_rate_nn_bpp: float = 0.0  # Total rate associated to the all NNs of all cool-chic [bpp]

    mse_y: Optional[float] = None
    mse_u: Optional[float] = None
    mse_v: Optional[float] = None

    # Texture channel MSEs
    mse_diffuse: Optional[float] = None
    mse_normal: Optional[float] = None
    mse_rm: Optional[float] = None

    psnr_y_db: Optional[float] = field(init=False, default=None)
    psnr_u_db: Optional[float] = field(init=False, default=None)
    psnr_v_db: Optional[float] = field(init=False, default=None)

    # Texture channel PSNRs (derived in __post_init__)
    psnr_diffuse_db: Optional[float] = field(init=False, default=None)
    psnr_normal_db: Optional[float] = field(init=False, default=None)
    psnr_rm_db: Optional[float] = field(init=False, default=None)

    # ==================== Not set by the init function ===================== #
    # Everything here is derived from the above metrics
    total_rate_latent_bpp: Optional[float] = field(
        init=False, default=None
    )  # Overall rate of all the latents [bpp]
    dist_db: Optional[float] = None
    detailed_dist_db: Optional[Dict[DISTORTION_METRIC, float]] = field(
        init=False, default_factory=lambda: {}
    )  # Each distortion value (mse, wasserstein...) in dB
    total_rate_bpp: Optional[float] = field(
        init=False, default=None
    )  # Overall rate: latent & NNs      [bpp]
    # ==================== Not set by the init function ===================== #

    def __post_init__(self):
        # Compute some dB values from distortion
        if self.detailed_dist is not None:
            self.detailed_dist_db["psnr_db"] = dist_to_db(self.detailed_dist["mse"])
            if "wasserstein" in self.detailed_dist:
                self.detailed_dist_db["wd_db"] = dist_to_db(self.detailed_dist["wasserstein"])

        self.dist_db = dist_to_db(self.dist)

        if self.mse_y is not None:
            self.psnr_y_db = dist_to_db(self.mse_y)
        if self.mse_u is not None:
            self.psnr_u_db = dist_to_db(self.mse_u)
        if self.mse_v is not None:
            self.psnr_v_db = dist_to_db(self.mse_v)

        if self.mse_diffuse is not None:
            self.psnr_diffuse_db = dist_to_db(self.mse_diffuse)
        if self.mse_normal is not None:
            self.psnr_normal_db = dist_to_db(self.mse_normal)
        if self.mse_rm is not None:
            self.psnr_rm_db = dist_to_db(self.mse_rm)

        if self.rate_latent_bpp is not None:
            self.total_rate_latent_bpp = sum(self.rate_latent_bpp.values())
        else:
            self.total_rate_latent_bpp = 0

        self.total_rate_bpp = self.total_rate_latent_bpp + self.total_rate_nn_bpp


def _compute_mse(x: Union[Tensor, DictTensorYUV], y: Union[Tensor, DictTensorYUV]) -> Tensor:
    """Compute the Mean Squared Error between two images. Both images can
    either be a single tensor, or a dictionary of tensors with one for each
    color channel. In case of images with multiple channels, the final MSE
    is obtained by averaging the MSE for each color channel, weighted by the
    number of pixels. E.g. for YUV 420:
        MSE = (4 * MSE_Y + MSE_U + MSE_V) / 6

    Args:
        x (Union[Tensor, DictTensorYUV]): One of the two inputs
        y (Union[Tensor, DictTensorYUV]): The other input

    Returns:
        Tensor: One element tensor containing the MSE of x and y.
    """
    flag_420 = not (isinstance(x, Tensor))

    if not flag_420:
        return mse_fn(x, y)
    else:
        # Total number of pixels for all channels
        total_pixels_yuv = 0.0

        # MSE weighted by the number of pixels in each channels
        mse = torch.zeros((1), device=x.get("y").device)
        for (_, x_channel), (_, y_channel) in zip(x.items(), y.items()):
            n_pixels_channel = x_channel.numel()
            mse = mse + mse_fn(x_channel, y_channel) * n_pixels_channel
            total_pixels_yuv += n_pixels_channel
        mse = mse / total_pixels_yuv
        return mse

def _compute_l1(decoded_textures: Tensor, target_textures: Tensor) -> Tensor:
    if type(decoded_textures) != Tensor or type(target_textures) != Tensor:
        raise ValueError(f"Expected decoded_textures and target_textures to be Tensors but got types {type(decoded_textures)} and {type(target_textures)} respectively.") 

    return (decoded_textures - target_textures).abs().mean()


def _compute_wasserstein(
    decoded_img: Union[Tensor, DictTensorYUV], target_img: Union[Tensor, DictTensorYUV]
) -> Tensor:
    """Compute the Wasserstein distance between two images. Both images can
    either be a single tensor, or a dictionary of tensors with one for each
    color channel. In case of images with multiple channels, the final Wasserstein
    distance is obtained by averaging the Wasserstein distance for each color channel,
    weighted by the number of pixels. E.g. for YUV 420:
        WD  = (4 * WD_Y + WD_U + WD_V) / 6

    Args:
        x (Union[Tensor, DictTensorYUV]): One of the two inputs
        y (Union[Tensor, DictTensorYUV]): The other input

    Returns:
        Tensor: One element tensor containing the WD of x and y.
    """
    flag_420 = not (isinstance(decoded_img, Tensor))

    if not flag_420:
        wd = wasserstein_fn(decoded_img, target_img)
    else:
        # Total number of pixels for all channels
        total_pixels_yuv = 0.0

        # WD weighted by the number of pixels in each channels
        wd = torch.zeros((1), device=decoded_img.get("y").device)
        for (_, decoded_channel), (_, target_channel) in zip(
            decoded_img.items(), target_img.items()
        ):
            n_pixels_channel = decoded_channel.numel()
            wd = wd + wasserstein_fn(decoded_channel, target_channel) * n_pixels_channel
            total_pixels_yuv += n_pixels_channel
        wd = wd / total_pixels_yuv
    return wd

def _compute_brdf_mod_mse(decoded_textures: Tensor, target_textures: Tensor) -> Tensor:
    if type(decoded_textures) != Tensor or type(target_textures) != Tensor:
        raise ValueError(f"Expected decoded_textures and target_textures to be Tensors but got types {type(decoded_textures)} and {type(target_textures)} respectively.") 

    if len(decoded_textures.shape) != 4 or decoded_textures.shape[1] != 7:
        raise ValueError(f"Expected there to be 4 dimensions with 7 channels but got {len(decoded_textures.shape)} dimensions and {decoded_textures.shape[1]} channels.")

    rand_samples = 4
    B, _, _, _ = decoded_textures.shape

    light_dir = hammersley.rand_sample_hemisphere_torch((rand_samples, B,), 256, device=decoded_textures.device).reshape(rand_samples, B, 3, 1, 1)
    eye_dir = hammersley.rand_sample_hemisphere_torch((rand_samples, B,), 256, device=decoded_textures.device).reshape(rand_samples, B, 3, 1, 1)

    decoded_diffuse, decoded_normals, decoded_rough, decoded_metal = brdf.organize_textures(decoded_textures)
    target_diffuse, target_normals, target_rough, target_metal = brdf.organize_textures(target_textures)

    decoded_diffuse = color.srgb_to_linear_torch(decoded_diffuse)
    target_diffuse = color.srgb_to_linear_torch(target_diffuse)

    half_vector = torch.nn.functional.normalize(light_dir + eye_dir, dim=2)

    v_dot_h = torch.sum(eye_dir * half_vector, dim=2, keepdim=True)

    n_dot_l_pred = torch.sum(decoded_normals.detach() * light_dir, dim=2, keepdim=True)
    n_dot_l_target = torch.sum(target_normals.detach() * light_dir, dim=2, keepdim=True)

    n_dot_v_pred = torch.sum(decoded_normals.detach() * eye_dir, dim=2, keepdim=True)
    n_dot_v_target = torch.sum(target_normals.detach() * eye_dir, dim=2, keepdim=True)

    n_dot_h_pred = torch.sum(decoded_normals.detach() * half_vector, dim=2, keepdim=True)
    n_dot_h_target = torch.sum(target_normals.detach() * half_vector, dim=2, keepdim=True)

    alpha_rough_pred = decoded_rough * decoded_rough
    alpha_rough_target = target_rough * target_rough

    metal_fresnel_pred = brdf.F_Shlick(decoded_diffuse, 1.0, torch.abs(v_dot_h))
    metal_fresnel_target = brdf.F_Shlick(target_diffuse, 1.0, torch.abs(v_dot_h))

    diffuse_brdf_pred = brdf.lambertian_brdf(decoded_diffuse)
    diffuse_brdf_target = brdf.lambertian_brdf(target_diffuse)

    v_ggx_pred = brdf.V_GGX_util(
        torch.clamp(n_dot_l_pred, 0.0, 1.0), 
        torch.clamp(n_dot_v_pred, 0.0, 1.0), 
        torch.clamp(alpha_rough_pred, 1e-8),
    )
    v_ggx_target = brdf.V_GGX_util(
        torch.clamp(n_dot_l_target, 0.0, 1.0), 
        torch.clamp(n_dot_v_target, 0.0, 1.0), 
        torch.clamp(alpha_rough_target, 1e-8),
    )

    d_ggx_pred = brdf.D_GGX_util(
        torch.clamp(n_dot_h_pred, 0.0, 1.0),
        torch.clamp(alpha_rough_pred, 0.0, 1e-8),
    )
    d_ggx_target = brdf.D_GGX_util(
        torch.clamp(n_dot_h_target, 0.0, 1.0),
        torch.clamp(alpha_rough_target, 0.0, 1e-8),
    )

    metal_fresnel_loss = (metal_fresnel_pred - metal_fresnel_target).square().mean()
    diffuse_brdf_loss = (diffuse_brdf_pred - diffuse_brdf_target).square().mean()
    v_ggx_loss = (v_ggx_pred - v_ggx_target).square().mean()
    d_ggx_loss = (d_ggx_pred - d_ggx_target).square().mean()
    metal_loss = (decoded_metal - target_metal).square().mean()
    normal_loss = (decoded_normals - target_normals).square().mean()

    return (diffuse_brdf_loss + v_ggx_loss + d_ggx_loss + metal_loss + normal_loss + metal_fresnel_loss) / 6

def _compute_pbr_target_and_pred(decoded_textures: Tensor, target_textures: Tensor, num_samples = 1) -> Tensor:
    if type(decoded_textures) != Tensor or type(target_textures) != Tensor:
        raise ValueError(f"Expected decoded_textures and target_textures to be Tensors but got types {type(decoded_textures)} and {type(target_textures)} respectively.") 

    if len(decoded_textures.shape) != 4 or decoded_textures.shape[1] != 7:
        raise ValueError(f"Expected there to be 4 dimensions with 7 channels but got {len(decoded_textures.shape)} dimensions and {decoded_textures.shape[1]} channels.")

    B, _, H, W = decoded_textures.shape

    light_dir = hammersley.rand_sample_hemisphere_torch((num_samples, B,), 256, device=decoded_textures.device).reshape(num_samples, B, 3, 1, 1)
    eye_dir = hammersley.rand_sample_hemisphere_torch((num_samples, B,), 256, device=decoded_textures.device).reshape(num_samples, B, 3, 1, 1)

    decoded_diffuse, decoded_normals, decoded_rough, decoded_metal = brdf.organize_textures(decoded_textures)
    target_diffuse, target_normals, target_rough, target_metal = brdf.organize_textures(target_textures)

    decoded_pbr = brdf.calc_pbr(light_dir, eye_dir, decoded_diffuse, decoded_normals, decoded_rough, decoded_metal, channel_dim=2).reshape(num_samples*B, 3, H, W)
    target_pbr = brdf.calc_pbr(light_dir, eye_dir, target_diffuse, target_normals, target_rough, target_metal, channel_dim=2).reshape(num_samples*B, 3, H, W)

    return decoded_pbr, target_pbr

def _compute_pbr_loss_relative_mse(decoded_textures: Tensor, target_textures: Tensor) -> Tensor:
    decoded_pbr, target_pbr = _compute_pbr_target_and_pred(decoded_textures, target_textures)

    se = (target_pbr - decoded_pbr).square()
    denominator = decoded_pbr.detach().square() + 1e-8

    relative_mse = (se / denominator).mean()

    return relative_mse

def _compute_pbr_loss_l1(decoded_textures: Tensor, target_textures: Tensor) -> Tensor:
    decoded_pbr, target_pbr = _compute_pbr_target_and_pred(decoded_textures, target_textures)

    return (decoded_pbr - target_pbr).abs().mean()

def _compute_pbr_loss_l1_lpips(decoded_textures: Tensor, target_textures: Tensor) -> Tensor:
    decoded_pbr, target_pbr = _compute_pbr_target_and_pred(decoded_textures, target_textures)
    decoded_pbr = brdf.pbr_neutral_tone_mapping(decoded_pbr, 1)
    target_pbr = brdf.pbr_neutral_tone_mapping(target_pbr, 1)

    if decoded_textures.device != _compute_pbr_loss_l1_lpips.lpips.device:
        _compute_pbr_loss_l1_lpips.lpips = _compute_pbr_loss_l1_lpips.lpips.to(device=decoded_textures.device)

    _compute_pbr_loss_l1_lpips.lpips.reset()
    return _compute_pbr_loss_l1_lpips.lpips(decoded_pbr, target_pbr) * 0.5 + (decoded_pbr - target_pbr).abs().mean()
_compute_pbr_loss_l1_lpips.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)


def loss_function(
    decoded_image: Union[Tensor, DictTensorYUV],
    rate_latent_bit: Dict[str, Tensor],
    target_image: Union[Tensor, DictTensorYUV],
    dist_weight: Dict[DISTORTION_METRIC, float],
    lmbda: float = 1e-3,
    total_rate_nn_bit: float = 0.0,
    compute_logs: bool = False,
) -> LossFunctionOutput:
    """Compute the loss and a few other quantities. The loss equation is:

    .. math::

        \\mathcal{L} = \\mathrm{D}(\hat{\\mathbf{x}}, \\mathbf{x}) + \\lambda
        (\\mathrm{R}(\hat{\\mathbf{x}}) + \\mathrm{R}_{NN}), \\text{ with }
        \\begin{cases}
            \\mathbf{x} & \\text{the original image}\\\\ \\hat{\\mathbf{x}} &
            \\text{the coded image}\\\\ \\mathrm{R}(\\hat{\\mathbf{x}}) &
            \\text{A measure of the rate of } \\hat{\\mathbf{x}} \\\\
                \\mathrm{R}_{NN} & \\text{The rate of the neural networks}\\\\
            \\mathrm{D}(\hat{\\mathbf{x}}, \\mathbf{x})  & \\text{A distortion
            metric specified by \\texttt{--tune} and \\texttt{--alpha}}
        \\end{cases}

    .. warning::

        There is no back-propagation through the term :math:`\\mathrm{R}_{NN}`.
        It is just here to be taken into account by the rate-distortion cost so
        that it better reflects the compression performance.

    Args:
        decoded_image: The decoded image, either as a Tensor for RGB or YUV444
            data, or as a dictionary of Tensors for YUV420 data.
        rate_latent_bit: Dictionary with the rate of each latent for each
            cool-chic decoder. Tensor with the rate of each latent value.
            The rate is in bit.
        target_image: The target image, either as a Tensor for RGB or YUV444
            data, or as a dictionary of Tensors for YUV420 data.
        lmbda: Rate constraint. Defaults to 1e-3.
        total_rate_nn_bit: Total rate of the NNs (arm + upsampling + synthesis)
            for all each cool-chic encoder. Rate is in bit. Defaults to 0.
        compute_logs: True to output a few more quantities beside the loss.
            Defaults to False.

    Returns:
        Object gathering the different quantities computed by this loss
        function. Chief among them: the loss itself.
    """

    if isinstance(target_image, Tensor):
        range_target = target_image.abs().max().item()
        if range_target > 1:
            target_min = target_image.min()
            target_max = target_image.max()

            decoded_image = (decoded_image - target_min) / (target_max - target_min)
            target_image = (target_image - target_min) / (target_max - target_min)

    flag_yuv420 = not isinstance(decoded_image, Tensor)

    device = decoded_image.get("y").device if flag_yuv420 else decoded_image.device

    all_dists = {}
    final_dist = torch.zeros((1), device=device)
    # Iterate on all possible distortion metrics.
    for dist_name, dist_w in dist_weight.items():
        if dist_name == "mse":
            cur_dist = _compute_mse(decoded_image, target_image)
        elif dist_name == "l1":
            cur_dist = _compute_l1(decoded_image, target_image)
        elif dist_name == "wasserstein":
            cur_dist = _compute_wasserstein(decoded_image, target_image)
        elif dist_name == "brdf_mod_mse":
            cur_dist = _compute_brdf_mod_mse(decoded_image, target_image)
        elif dist_name == "brdf_l1":
            cur_dist = _compute_pbr_loss_l1(decoded_image, target_image)
        elif dist_name == "brdf_l1_lpips":
            cur_dist = _compute_pbr_loss_l1_lpips(decoded_image, target_image)
        elif dist_name == "brdf_rel_mse":
            cur_dist = _compute_pbr_loss_relative_mse(decoded_image, target_image)
        else:
            raise ValueError(
                f"Unsupported distortion metrics. Found {dist_name}, available "
                f"values are {typing.get_args(DISTORTION_METRIC)}. Exiting!"
            )

        all_dists[dist_name] = cur_dist
        # Aggregate weighted dist
        final_dist = final_dist + dist_w * cur_dist

    if flag_yuv420:
        n_pixels = decoded_image.get("y").size()[-2] * decoded_image.get("y").size()[-1]
    else:
        n_pixels = decoded_image.size()[-2] * decoded_image.size()[-1]

    total_rate_latent_bit = torch.cat([v.sum().view(1) for _, v in rate_latent_bit.items()]).sum()
    rate_bpp = total_rate_latent_bit + total_rate_nn_bit
    rate_bpp = rate_bpp / n_pixels

    loss = final_dist + lmbda * rate_bpp

    # Construct the output module, only the loss is always returned
    rate_latent_bpp = None
    total_rate_nn_bpp = 0.0

    mse_y = None
    mse_u = None
    mse_v = None
    mse_diffuse = None
    mse_normal = None
    mse_rm = None

    if compute_logs:
        rate_latent_bpp = {
            k: v.detach().sum().item() / n_pixels for k, v in rate_latent_bit.items()
        }
        total_rate_nn_bpp = total_rate_nn_bit / n_pixels

        # Detach all distortions only when computing logs
        for k, v in all_dists.items():
            all_dists[k] = v.detach().item()

        if flag_yuv420:
            mse_y = _compute_mse(decoded_image.get("y"), target_image.get("y")).detach().item()
            mse_u = _compute_mse(decoded_image.get("u"), target_image.get("u")).detach().item()
            mse_v = _compute_mse(decoded_image.get("v"), target_image.get("v")).detach().item()

        elif isinstance(target_image, Tensor) and target_image.shape[1] == 7:
            # [0:3] diffuse RGB, [3:5] normal RG, [5:7] roughness/metalness
            mse_diffuse = _compute_mse(decoded_image[:, 0:3], target_image[:, 0:3]).detach().item()
            mse_normal  = _compute_mse(decoded_image[:, 3:5], target_image[:, 3:5]).detach().item()
            mse_rm      = _compute_mse(decoded_image[:, 5:7], target_image[:, 5:7]).detach().item()

    output = LossFunctionOutput(
        loss=loss,
        dist=final_dist.detach().item(),
        rate_bpp=rate_bpp.detach().item(),
        detailed_dist=all_dists if compute_logs else None,
        total_rate_nn_bpp=total_rate_nn_bpp,
        rate_latent_bpp=rate_latent_bpp,
        mse_y=mse_y,
        mse_u=mse_u,
        mse_v=mse_v,
        mse_diffuse=mse_diffuse,
        mse_normal=mse_normal,
        mse_rm=mse_rm,
    )

    return output
