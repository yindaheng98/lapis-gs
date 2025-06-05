from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import DensificationTrainer
from gaussian_splatting.trainer.densifier import AbstractDensifier


class PartialDensificationTrainer(DensificationTrainer):

    def __init__(
            self, model: GaussianModel,
            scene_extent: float,
            densifier: AbstractDensifier,
            *args, **kwargs
    ):
        super().__init__(model, scene_extent, densifier, *args, **kwargs)
        self.size_fixed_gs = model.get_xyz.shape[0]

    def optim_step(self):
        # freeze all other parameters except opacity in first `size_fixed_gs` points
        # https://github.com/nus-vv-streams/lapis-gs/blob/12dcda37ed43838d7407b28675bc26b7364ae431/scene/gaussian_model.py#L322
        self.model._xyz.grad[:self.size_fixed_gs] = 0
        self.model._features_dc.grad[:self.size_fixed_gs] = 0
        self.model._features_rest.grad[:self.size_fixed_gs] = 0
        self.model._scaling.grad[:self.size_fixed_gs] = 0
        self.model._rotation.grad[:self.size_fixed_gs] = 0
        return super().optim_step()

    def remove_points(self, rm_mask):
        rm_mask[:self.size_fixed_gs] = False
        return super().remove_points(rm_mask)
