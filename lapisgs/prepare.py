from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.trainer import AbstractTrainer
from gaussian_splatting.trainer.extensions import ScaleRegularizeTrainerWrapper
from lapisgs.dataset import RescaleJSONCameraDataset, RescaleColmapCameraDataset, RescaleTrainableCameraDataset, RescaleColmapTrainableCameraDataset
from lapisgs.trainer import Trainer, CameraTrainer, DepthTrainer, DepthCameraTrainer


def prepare_dataset(source: str, device: str, trainable_camera: bool = False, load_camera: str = None, load_depth=False, rescale_factor=1.0) -> CameraDataset:
    if trainable_camera:
        dataset = (RescaleTrainableCameraDataset.from_json(load_camera, load_depth=load_depth, rescale_factor=rescale_factor)
                   if load_camera else RescaleColmapTrainableCameraDataset(source, load_depth=load_depth, rescale_factor=rescale_factor)).to(device)
    else:
        dataset = (RescaleJSONCameraDataset(load_camera, load_depth=load_depth, rescale_factor=rescale_factor)
                   if load_camera else RescaleColmapCameraDataset(source, load_depth=load_depth, rescale_factor=rescale_factor)).to(device)
    return dataset


modes = {
    "base": DepthTrainer,
    "camera": DepthCameraTrainer,
    "nodepth-base": Trainer,
    "nodepth-camera": CameraTrainer,
}


def prepare_trainer(gaussians: GaussianModel, dataset: CameraDataset, mode: str, trainable_camera: bool = False, load_ply: str = None, with_scale_reg=False, configs={}) -> AbstractTrainer:
    if not load_ply:
        from gaussian_splatting.prepare import prepare_trainer as legacy_prepare_trainer
        modemap = {
            "base": "densify",
            "camera": "camera-densify",
            "nodepth-base": "nodepth-densify",
            "nodepth-camera": "nodepth-camera-densify",
        }
        return legacy_prepare_trainer(
            gaussians=gaussians, dataset=dataset, mode=modemap[mode], trainable_camera=trainable_camera, load_ply=load_ply, with_scale_reg=with_scale_reg, configs=configs)
    constructor = modes[mode]
    if with_scale_reg:
        constructor = lambda *args, **kwargs: ScaleRegularizeTrainerWrapper(modes[mode], *args, **kwargs)
    if trainable_camera:
        trainer = constructor(
            gaussians,
            scene_extent=dataset.scene_extent(),
            dataset=dataset,
            **configs
        )
    else:
        trainer = constructor(
            gaussians,
            scene_extent=dataset.scene_extent(),
            **configs
        )
    return trainer
