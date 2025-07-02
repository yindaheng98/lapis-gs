import torch
from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.dataset.colmap import colmap_init
from gaussian_splatting.trainer import AbstractTrainer
from gaussian_splatting.trainer.extensions import ScaleRegularizeTrainerWrapper
from lapisgs.dataset import colmap_concat
from lapisgs.trainer.extensions.reduced_3dgs import *


def prepare_gaussians(sh_degree: int, source: str, device: str, trainable_camera: bool = False, load_ply: str = None) -> GaussianModel:
    from reduced_3dgs.prepare import prepare_gaussians as legacy_prepare_gaussians
    gaussians = legacy_prepare_gaussians(sh_degree=sh_degree, source=source, device=device, trainable_camera=trainable_camera, load_ply=load_ply)
    fixed_size = None
    if load_ply:
        fixed_size = gaussians.get_xyz.shape[0]
        colmap_concat(gaussians, source)
        torch.cuda.empty_cache()
    return gaussians, fixed_size


modes = {
    "base": LapisFullTrainer,
    "camera": CameraLapisFullTrainer,
    "shculling": SHCullingLapisFullTrainer,
    "camera-shculling": CameraSHCullingLapisFullTrainer,
}


def prepare_trainer(gaussians: GaussianModel, dataset: CameraDataset, mode: str, load_ply: str = None, with_scale_reg=False, configs={}) -> AbstractTrainer:
    if not load_ply:
        from reduced_3dgs.prepare import prepare_trainer as legacy_prepare_trainer
        modemap = {
            "base": "densify-pruning",
            "camera": "camera-densify-pruning",
            "shculling": "densify-prune-shculling",
            "camera-shculling": "camera-densify-prune-shculling",
        }
        trainer, _ = legacy_prepare_trainer(
            gaussians=gaussians, dataset=dataset, mode=modemap[mode], with_scale_reg=with_scale_reg, quantize=False, configs=configs)
        return trainer
    constructor = modes[mode]
    if with_scale_reg:
        constructor = lambda *args, **kwargs: ScaleRegularizeTrainerWrapper(modes[mode], *args, **kwargs)
    trainer = constructor(
        gaussians,
        scene_extent=dataset.scene_extent(),
        dataset=dataset,
        **configs
    )
    return trainer
