from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.trainer import DepthTrainerWrapper, CameraTrainerWrapper
from lapisgs.dataset import RescaleTrainableCameraDataset
from .densifier import BasePartialDensificationTrainer
from .opacity_reset import PartialOpacityResetTrainerWrapper


def LapisTrainer(model: GaussianModel, dataset: CameraDataset, opacity_lr=0.05, **configs):
    return PartialOpacityResetTrainerWrapper(BasePartialDensificationTrainer, model, dataset, opacity_lr=opacity_lr, **configs)


def DepthLapisTrainer(model: GaussianModel, dataset: CameraDataset, **configs):
    return DepthTrainerWrapper(LapisTrainer, model, dataset, **configs)


def LapisCameraTrainer(model: CameraTrainableGaussianModel, dataset: RescaleTrainableCameraDataset, **configs):
    return CameraTrainerWrapper(
        LapisTrainer,
        model, dataset, **configs
    )


def DepthLapisCameraTrainer(model: CameraTrainableGaussianModel, dataset: RescaleTrainableCameraDataset, **configs):
    return CameraTrainerWrapper(
        DepthLapisTrainer,
        model, dataset, **configs
    )


# Aliases for default trainers
Trainer = LapisTrainer
CameraTrainer = LapisCameraTrainer
DepthTrainer = DepthLapisTrainer
DepthCameraTrainer = DepthLapisCameraTrainer
