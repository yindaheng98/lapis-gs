from typing import Callable
from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import DensificationTrainer
from gaussian_splatting.trainer.densifier import AbstractDensifier, SplitCloneDensifier


class PartialDensificationTrainer(DensificationTrainer):

    def __init__(
            self, model: GaussianModel,
            scene_extent: float,
            densifier: AbstractDensifier,
            *args,
            fix_xyz=True,
            fix_features_dc=True,
            fix_features_rest=True,
            fix_scaling=True,
            fix_rotation=True,
            fix_opacity=False,
            **kwargs
    ):
        super().__init__(model, scene_extent, densifier, *args, **kwargs)
        self.size_fixed_gs = model.get_xyz.shape[0]
        self.fix_xyz = fix_xyz
        self.fix_features_dc = fix_features_dc
        self.fix_features_rest = fix_features_rest
        self.fix_scaling = fix_scaling
        self.fix_rotation = fix_rotation
        self.fix_opacity = fix_opacity

    def optim_step(self):
        # freeze all other parameters except opacity in first `size_fixed_gs` points
        # https://github.com/nus-vv-streams/lapis-gs/blob/12dcda37ed43838d7407b28675bc26b7364ae431/scene/gaussian_model.py#L322
        if self.model._xyz.grad is not None and self.fix_xyz:  # grad is None when after pruning or densification
            self.model._xyz.grad[:self.size_fixed_gs] = 0
        if self.model._features_dc.grad is not None and self.fix_features_dc:
            self.model._features_dc.grad[:self.size_fixed_gs] = 0
        if self.model._features_rest.grad is not None and self.fix_features_rest:
            self.model._features_rest.grad[:self.size_fixed_gs] = 0
        if self.model._scaling.grad is not None and self.fix_scaling:
            self.model._scaling.grad[:self.size_fixed_gs] = 0
        if self.model._rotation.grad is not None and self.fix_rotation:
            self.model._rotation.grad[:self.size_fixed_gs] = 0
        if self.model._opacity.grad is not None and self.fix_opacity:
            self.model._opacity.grad[:self.size_fixed_gs] = 0
        return super().optim_step()

    def remove_points(self, rm_mask):
        rm_mask[:self.size_fixed_gs] = False
        return super().remove_points(rm_mask)


def SplitClonePartialDensifierTrainerWrapper(
        noargs_base_densifier_constructor: Callable[[GaussianModel, float], AbstractDensifier],
        model: GaussianModel,
        scene_extent: float,
        *args,
        densify_from_iter=500,
        densify_until_iter=15000,
        densify_interval=100,
        densify_grad_threshold=0.0002,
        densify_percent_dense=0.01,
        densify_percent_too_big=0.8,
        fix_xyz=True,
        fix_features_dc=True,
        fix_features_rest=True,
        fix_scaling=True,
        fix_rotation=True,
        fix_opacity=False,
        **kwargs):
    # https://github.com/yindaheng98/gaussian-splatting/blob/56576b647d9c5bd05300f5640cd03a8c75a760bc/gaussian_splatting/trainer/densifier/densifier.py#L149
    densifier = noargs_base_densifier_constructor(model, scene_extent)
    densifier = SplitCloneDensifier(
        densifier,
        scene_extent,
        densify_from_iter=densify_from_iter,
        densify_until_iter=densify_until_iter,
        densify_interval=densify_interval,
        densify_grad_threshold=densify_grad_threshold,
        densify_percent_dense=densify_percent_dense,
        densify_percent_too_big=densify_percent_too_big
    )
    return PartialDensificationTrainer(
        model, scene_extent,
        densifier,
        *args,
        fix_xyz=fix_xyz,
        fix_features_dc=fix_features_dc,
        fix_features_rest=fix_features_rest,
        fix_scaling=fix_scaling,
        fix_rotation=fix_rotation,
        fix_opacity=fix_opacity,
        **kwargs
    )
