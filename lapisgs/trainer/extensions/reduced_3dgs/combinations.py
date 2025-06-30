from typing import List
from gaussian_splatting import GaussianModel, Camera
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import CameraTrainerWrapper, NoopDensifier, DepthTrainerWrapper
from reduced_3dgs.shculling import VariableSHGaussianModel, SHCullingTrainerWrapper
from reduced_3dgs.combinations import CameraTrainableVariableSHGaussianModel
from lapisgs.trainer import LapisTrainer, PartialOpacityResetTrainerWrapper
from .trainer import PartialPrunerInDensifyTrainerWrapper
from .importance import ImportancePruner


def PartialFullPrunerInDensifyTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: List[Camera],
        *args,
        importance_prune_from_iter=15000,
        importance_prune_until_iter=20000,
        importance_prune_interval: int = 1000,
        importance_score_resize=None,
        importance_prune_type="comprehensive",
        importance_prune_percent=0.1,
        importance_prune_thr_important_score=None,
        importance_prune_thr_v_important_score=3.0,
        importance_prune_thr_max_v_important_score=None,
        importance_prune_thr_count=1,
        importance_prune_thr_T_alpha=1.0,
        importance_prune_thr_T_alpha_avg=0.001,
        importance_v_pow=0.1,
        **kwargs):
    return PartialPrunerInDensifyTrainerWrapper(
        lambda model, scene_extent, dataset: ImportancePruner(
            NoopDensifier(model),
            dataset,
            importance_prune_from_iter=importance_prune_from_iter,
            importance_prune_until_iter=importance_prune_until_iter,
            importance_prune_interval=importance_prune_interval,
            importance_score_resize=importance_score_resize,
            importance_prune_type=importance_prune_type,
            importance_prune_percent=importance_prune_percent,
            importance_prune_thr_important_score=importance_prune_thr_important_score,
            importance_prune_thr_v_important_score=importance_prune_thr_v_important_score,
            importance_prune_thr_max_v_important_score=importance_prune_thr_max_v_important_score,
            importance_prune_thr_count=importance_prune_thr_count,
            importance_prune_thr_T_alpha=importance_prune_thr_T_alpha,
            importance_prune_thr_T_alpha_avg=importance_prune_thr_T_alpha_avg,
            importance_v_pow=importance_v_pow,
        ),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def DepthPartialFullPrunerInDensifyTrainer(model: GaussianModel, scene_extent: float, dataset: CameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(
        PartialFullPrunerInDensifyTrainer,
        model, scene_extent, dataset,
        *args, **kwargs)


def PartialOpacityResetPrunerInDensifyTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return PartialOpacityResetTrainerWrapper(
        lambda model, scene_extent, *args, **kwargs: DepthPartialFullPrunerInDensifyTrainer(model, scene_extent, dataset, *args, **kwargs),
        model, scene_extent,
        *args, **kwargs
    )


LapisFullTrainer = PartialOpacityResetPrunerInDensifyTrainer


def SHCullingLapisTrainer(
    model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return SHCullingTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: LapisTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def SHCullingLapisFullTrainer(
    model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return SHCullingTrainerWrapper(
        PartialOpacityResetPrunerInDensifyTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraLapisFullTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        PartialOpacityResetPrunerInDensifyTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraSHCullingLapisTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        SHCullingLapisTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraSHCullingLapisFullTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        SHCullingLapisFullTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )
