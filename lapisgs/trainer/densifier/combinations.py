from typing import Callable
from gaussian_splatting import GaussianModel

from gaussian_splatting.trainer.densifier import AbstractDensifier, NoopDensifier, OpacityPruner
from .trainer import SplitClonePartialDensifierTrainerWrapper


def PartialDensificationTrainerWrapper(
        noargs_base_densifier_constructor: Callable[[GaussianModel, float], AbstractDensifier],
        model: GaussianModel,
        scene_extent: float,
        *args,
        prune_from_iter=1000,
        prune_until_iter=15000,
        prune_interval=100,
        prune_screensize_threshold=20,
        prune_percent_too_big=1,
        prune_opacity_threshold=0.005,
        **kwargs):
    # https://github.com/yindaheng98/gaussian-splatting/blob/56576b647d9c5bd05300f5640cd03a8c75a760bc/gaussian_splatting/trainer/densifier/combinations.py#L10
    return SplitClonePartialDensifierTrainerWrapper(
        lambda model, scene_extent: OpacityPruner(
            noargs_base_densifier_constructor(model, scene_extent),
            scene_extent,
            prune_from_iter=prune_from_iter,
            prune_until_iter=prune_until_iter,
            prune_interval=prune_interval,
            prune_screensize_threshold=prune_screensize_threshold,
            prune_percent_too_big=prune_percent_too_big,
            prune_opacity_threshold=prune_opacity_threshold
        ),
        model, scene_extent,
        *args, **kwargs
    )


def PartialDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        *args, **kwargs):
    # https://github.com/yindaheng98/gaussian-splatting/blob/56576b647d9c5bd05300f5640cd03a8c75a760bc/gaussian_splatting/trainer/densifier/combinations.py#L38
    return PartialDensificationTrainerWrapper(
        lambda model, scene_extent: NoopDensifier(model),
        model,
        scene_extent,
        *args, **kwargs
    )
