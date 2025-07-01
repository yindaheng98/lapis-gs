from typing import Callable
import torch

from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import AbstractTrainer
from gaussian_splatting.trainer.opacity_reset import OpacityResetter, replace_tensor_to_optimizer


class PartialOpacityResetter(OpacityResetter):
    def __init__(
            self,
            base_trainer: AbstractTrainer,
            *args,
            reset_fixed_opacity_to=None,
            **kwargs
    ):
        super().__init__(base_trainer, *args, **kwargs)
        self.size_fixed_gs = base_trainer.model.get_opacity.shape[0]
        assert reset_fixed_opacity_to is None or reset_fixed_opacity_to < 1, "reset_fixed_opacity_to should be less than 1"
        self.reset_fixed_opacity_to = reset_fixed_opacity_to

    def optim_step(self):
        with torch.no_grad():
            if self.opacity_reset_from_iter <= self.curr_step <= self.opacity_reset_until_iter and self.curr_step % self.opacity_reset_interval == 0:
                # https://github.com/nus-vv-streams/lapis-gs/blob/12dcda37ed43838d7407b28675bc26b7364ae431/scene/gaussian_model.py#L226
                opacities_new_dynamic = self.model.inverse_opacity_activation(torch.min(self.model.get_opacity[self.size_fixed_gs:], torch.ones_like(self.model.get_opacity[self.size_fixed_gs:])*0.01))
                opacities_new_static = self.model.inverse_opacity_activation(torch.ones_like(self.model.get_opacity[:self.size_fixed_gs])*self.reset_fixed_opacity_to) if self.reset_fixed_opacity_to else self.model._opacity[:self.size_fixed_gs]
                opacities_new = torch.cat((opacities_new_static, opacities_new_dynamic), dim=0)
                optimizable_tensors = replace_tensor_to_optimizer(self.optimizer, opacities_new, "opacity")
                self.model._opacity = optimizable_tensors["opacity"]
                torch.cuda.empty_cache()
        return super().optim_step()


def PartialOpacityResetTrainerWrapper(
        base_trainer_constructor: Callable[..., AbstractTrainer],
        model: GaussianModel,
        scene_extent: float,
        *args,
        opacity_reset_from_iter=3000,
        opacity_reset_until_iter=15000,
        opacity_reset_interval=3000,
        reset_fixed_opacity_to=None,
        **kwargs) -> PartialOpacityResetter:
    return PartialOpacityResetter(
        base_trainer=base_trainer_constructor(model, scene_extent, *args, **kwargs),
        opacity_reset_from_iter=opacity_reset_from_iter,
        opacity_reset_until_iter=opacity_reset_until_iter,
        opacity_reset_interval=opacity_reset_interval,
        reset_fixed_opacity_to=reset_fixed_opacity_to,
    )
