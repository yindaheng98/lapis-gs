import os
import torch
from gaussian_splatting.train import save_cfg_args, training
from lapisgs.prepare_reduced import modes
from lapisgs.train_reduced import prepare_training


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", default=30000, type=int)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--no_depth_data", action="store_true")
    parser.add_argument("--resolution_scales", action="append", type=int, default=[8, 4, 2, 1])
    parser.add_argument("--with_scale_reg", action="store_true")
    parser.add_argument("--mode", choices=sorted(list(modes.keys())), default="base")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("-o", "--option", default=[], action='append', type=str)
    args = parser.parse_args()
    torch.autograd.set_detect_anomaly(False)

    load_camera = None
    foundation_gs_path = None
    for resolution in args.resolution_scales:
        destination = os.path.join(args.destination, f"{resolution}x")
        save_cfg_args(destination, args.sh_degree, args.source)

        configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
        dataset, gaussians, trainer = prepare_training(
            sh_degree=args.sh_degree, source=args.source, device=args.device, mode=args.mode, trainable_camera="camera" in args.mode,
            load_ply=foundation_gs_path, load_camera=load_camera, load_depth=not args.no_depth_data, rescale_factor=1/resolution, with_scale_reg=args.with_scale_reg, configs=configs)
        dataset.save_cameras(os.path.join(destination, "cameras.json"))
        torch.cuda.empty_cache()
        training(
            dataset=dataset, gaussians=gaussians, trainer=trainer,
            destination=destination, iteration=args.iteration, save_iterations=args.save_iterations,
            device=args.device)
        load_camera = os.path.join(destination, "cameras.json")
        foundation_gs_path = os.path.join(destination, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
