from typing import Callable
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import AbstractDensifier, DepthTrainerWrapper, NoopDensifier
from lapisgs.trainer import SplitClonePartialDensifierTrainerWrapper
from reduced_3dgs.pruning import BasePruner

# Reduced 3DGS


def PartialPrunerInDensifyTrainerWrapper(
        noargs_base_densifier_constructor: Callable[[GaussianModel, CameraDataset], AbstractDensifier],
        model: GaussianModel,
        dataset: CameraDataset,
        prune_from_iter=1000,
        prune_until_iter=15000,
        prune_interval: int = 100,
        box_size=1.,
        lambda_mercy=1.,
        mercy_minimum=3,
        mercy_type='redundancy_opacity',
        **configs):
    return SplitClonePartialDensifierTrainerWrapper(
        lambda model, dataset: BasePruner(
            noargs_base_densifier_constructor(model, dataset),
            dataset,
            prune_from_iter=prune_from_iter,
            prune_until_iter=prune_until_iter,
            prune_interval=prune_interval,
            box_size=box_size,
            lambda_mercy=lambda_mercy,
            mercy_minimum=mercy_minimum,
            mercy_type=mercy_type,
        ),
        model,
        dataset,
        **configs
    )


def PartialPrunerInDensifyTrainer(
        model: GaussianModel,
        dataset: CameraDataset,
        **configs):
    return PartialPrunerInDensifyTrainerWrapper(
        lambda model, dataset: NoopDensifier(model),
        model, dataset,
        **configs
    )


# Depth trainer

def DepthPrunerInDensifyTrainer(model: GaussianModel, dataset: TrainableCameraDataset, **configs):
    return DepthTrainerWrapper(
        PartialPrunerInDensifyTrainer,
        model, dataset,
        **configs)


LapisReducedTrainer = DepthPrunerInDensifyTrainer
BaseLapisReducedTrainer = PartialPrunerInDensifyTrainer
