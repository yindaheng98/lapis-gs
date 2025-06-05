import os
from typing import Tuple
import torch
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.trainer import *
from gaussian_splatting.trainer.extensions import ScaleRegularizeTrainerWrapper
from gaussian_splatting.train import save_cfg_args, training, basemodes, shliftmodes, prepare_gaussians
from lapisgs.dataset import RescaleJSONCameraDataset, RescaleColmapCameraDataset, RescaleTrainableCameraDataset, RescaleColmapTrainableCameraDataset
from lapisgs.trainer import Trainer, CameraTrainer, DepthTrainer, DepthCameraTrainer

modes = {
    "base": DepthTrainer,
    "camera": DepthCameraTrainer,
    "nodepth-base": Trainer,
    "nodepth-camera": CameraTrainer,
}


def prepare_dataset(source: str, device: str, mode: str, load_camera: str = None, load_depth=False, rescale_factor=1.0):
    match mode:
        case "base" | "nodepth":
            dataset = (RescaleJSONCameraDataset(load_camera, load_depth=load_depth, rescale_factor=rescale_factor) if load_camera else RescaleColmapCameraDataset(source, load_depth=load_depth, rescale_factor=rescale_factor)).to(device)
        case "camera" | "nodepth-camera":
            dataset = (RescaleTrainableCameraDataset.from_json(load_camera, load_depth=load_depth, rescale_factor=rescale_factor)
                       if load_camera else RescaleColmapTrainableCameraDataset(source, load_depth=load_depth, rescale_factor=rescale_factor)).to(device)
        case _:
            raise ValueError(f"Unknown mode: {mode}")
    return dataset


def prepare_trainer(gaussians: GaussianModel, dataset: CameraDataset, mode: str, load_ply: str = None, with_scale_reg=False, configs={}) -> AbstractTrainer:
    if not load_ply:
        from gaussian_splatting.train import prepare_trainer as legacy_prepare_trainer
        modemap = {
            "base": "densify",
            "camera": "camera-densify",
            "nodepth-base": "nodepth-densify",
            "nodepth-camera": "nodepth-camera-densify",
        }
        return legacy_prepare_trainer(
            gaussians=gaussians, dataset=dataset, mode=modemap[mode], load_ply=load_ply, with_scale_reg=with_scale_reg, configs=configs)
    constructor = modes[mode]
    if with_scale_reg:
        constructor = lambda *args, **kwargs: ScaleRegularizeTrainerWrapper(modes[mode], *args, **kwargs)
    match mode:
        case "base" | "nodepth":
            trainer = constructor(
                gaussians,
                scene_extent=dataset.scene_extent(),
                **configs
            )
        case "camera" | "nodepth-camera":
            trainer = constructor(
                gaussians,
                scene_extent=dataset.scene_extent(),
                dataset=dataset,
                **configs
            )
        case _:
            raise ValueError(f"Unknown mode: {mode}")
    return gaussians, trainer


def prepare_training(sh_degree: int, source: str, device: str, mode: str, load_ply: str = None, load_camera: str = None, load_depth=False, rescale_factor=1.0, with_scale_reg=False, configs={}) -> Tuple[CameraDataset, GaussianModel, AbstractTrainer]:
    dataset = prepare_dataset(source=source, device=device, mode=mode, load_camera=load_camera, load_depth=load_depth, rescale_factor=rescale_factor)
    gaussians = prepare_gaussians(sh_degree=sh_degree, source=source, device=device, mode=mode, load_ply=load_ply)
    trainer = prepare_trainer(gaussians=gaussians, dataset=dataset, mode=mode, load_ply=load_ply, with_scale_reg=with_scale_reg, configs=configs)
    return dataset, gaussians, trainer


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", default=30000, type=int)
    parser.add_argument("-l", "--foundation_gs_path", default=None, type=str)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--no_depth_data", action="store_true")
    parser.add_argument("--rescale_factor", default=1.0, type=float)
    parser.add_argument("--with_scale_reg", action="store_true")
    parser.add_argument("--mode", choices=sorted(list(set(list(modes.keys()) + list(basemodes.keys()) + list(shliftmodes.keys())))), default="base")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("-o", "--option", default=[], action='append', type=str)
    args = parser.parse_args()
    save_cfg_args(args.destination, args.sh_degree, args.source)
    torch.autograd.set_detect_anomaly(False)

    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    dataset, gaussians, trainer = prepare_training(
        sh_degree=args.sh_degree, source=args.source, device=args.device, mode=args.mode,
        load_ply=args.foundation_gs_path, load_camera=args.load_camera, load_depth=not args.no_depth_data, rescale_factor=args.rescale_factor, with_scale_reg=args.with_scale_reg, configs=configs)
    dataset.save_cameras(os.path.join(args.destination, "cameras.json"))
    torch.cuda.empty_cache()
    training(
        dataset=dataset, gaussians=gaussians, trainer=trainer,
        destination=args.destination, iteration=args.iteration, save_iterations=args.save_iterations,
        device=args.device)
