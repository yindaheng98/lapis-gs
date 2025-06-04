import torch
from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import BaseTrainer


class OpacityOnlyTrainer(BaseTrainer):
    def __init__(
            self, model: GaussianModel,
            lambda_dssim=0.8,
            opacity_lr=0.05,
    ):
        super().__init__()
        self.lambda_dssim = lambda_dssim
        params = [{'params': [model._opacity], 'lr': opacity_lr, "name": "opacity"}]
        self._model = model
        self._optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        self._schedulers = {}
        self._curr_step = 0
