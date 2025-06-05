from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.trainer import DepthTrainerWrapper, CameraTrainerWrapper
from lapisgs.dataset import RescaleTrainableCameraDataset
from .densifier import PartialDensificationTrainer
from .opacity_reset import PartialOpacityResetTrainerWrapper


def LapisTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return PartialOpacityResetTrainerWrapper(PartialDensificationTrainer, model, scene_extent, *args, **kwargs)


def DepthLapisTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return DepthTrainerWrapper(LapisTrainer, model, scene_extent, *args, **kwargs)


def LapisCameraTrainer(model: CameraTrainableGaussianModel, scene_extent: float, dataset: RescaleTrainableCameraDataset, *args, **kwargs):
    return CameraTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: LapisTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset, *args, **kwargs
    )


def DepthLapisCameraTrainer(model: CameraTrainableGaussianModel, scene_extent: float, dataset: RescaleTrainableCameraDataset, *args, **kwargs):
    return CameraTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: DepthLapisTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset, *args, **kwargs
    )
