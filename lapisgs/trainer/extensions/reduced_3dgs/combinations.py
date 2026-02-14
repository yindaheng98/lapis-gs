from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import CameraTrainerWrapper, NoopDensifier, DepthTrainerWrapper
from reduced_3dgs.shculling import VariableSHGaussianModel, SHCullingTrainerWrapper
from reduced_3dgs.combinations import CameraTrainableVariableSHGaussianModel
from lapisgs.trainer import LapisTrainer, PartialOpacityResetTrainerWrapper
from .trainer import PartialPrunerInDensifyTrainerWrapper
from .importance import ImportancePruner


def PartialFullPrunerInDensifyTrainer(
        model: GaussianModel,
        dataset: CameraDataset,
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
        **configs):
    return PartialPrunerInDensifyTrainerWrapper(
        lambda model, dataset: ImportancePruner(
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
        model, dataset,
        **configs
    )


def DepthPartialFullPrunerInDensifyTrainer(model: GaussianModel, dataset: CameraDataset, **configs):
    return DepthTrainerWrapper(
        PartialFullPrunerInDensifyTrainer,
        model, dataset,
        **configs)


def PartialOpacityResetPrunerInDensifyTrainer(
        model: GaussianModel,
        dataset: CameraDataset,
        **configs):
    return PartialOpacityResetTrainerWrapper(
        DepthPartialFullPrunerInDensifyTrainer,
        model, dataset,
        **configs
    )


LapisFullTrainer = PartialOpacityResetPrunerInDensifyTrainer


def SHCullingLapisTrainer(
    model: VariableSHGaussianModel,
        dataset: CameraDataset,
        **configs):
    return SHCullingTrainerWrapper(
        LapisTrainer,
        model, dataset,
        **configs
    )


def SHCullingLapisFullTrainer(
    model: VariableSHGaussianModel,
        dataset: CameraDataset,
        **configs):
    return SHCullingTrainerWrapper(
        PartialOpacityResetPrunerInDensifyTrainer,
        model, dataset,
        **configs
    )


def CameraLapisFullTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        dataset: TrainableCameraDataset,
        **configs):
    return CameraTrainerWrapper(
        PartialOpacityResetPrunerInDensifyTrainer,
        model, dataset,
        **configs
    )


def CameraSHCullingLapisTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        dataset: TrainableCameraDataset,
        **configs):
    return CameraTrainerWrapper(
        SHCullingLapisTrainer,
        model, dataset,
        **configs
    )


def CameraSHCullingLapisFullTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        dataset: TrainableCameraDataset,
        **configs):
    return CameraTrainerWrapper(
        SHCullingLapisFullTrainer,
        model, dataset,
        **configs
    )
