from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.trainer import AbstractTrainer
from gaussian_splatting.trainer.extensions import ScaleRegularizeTrainerWrapper
from lapisgs.trainer.extensions.reduced_3dgs import *

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
