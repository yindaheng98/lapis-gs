import torch
import os
from gaussian_splatting.prepare import prepare_gaussians
from gaussian_splatting.render import rendering
from lapisgs.prepare import prepare_dataset


def prepare_rendering(sh_degree: int, source: str, device: str, trainable_camera: bool = False, load_ply: str = None, load_camera: str = None, load_depth=False, rescale_factor=1.0):
    dataset = prepare_dataset(source=source, device=device, trainable_camera=trainable_camera, load_camera=load_camera, load_depth=load_depth, rescale_factor=rescale_factor)
    gaussians = prepare_gaussians(sh_degree=sh_degree, source=source, device=device, trainable_camera=trainable_camera, load_ply=load_ply)
    return dataset, gaussians


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", required=True, type=int)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--rescale_factor", default=1.0, type=float)
    parser.add_argument("--mode", choices=["base", "camera"], default="base")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--no_rescale_depth_gt", action="store_true")
    parser.add_argument("--save_depth_pcd", action="store_true")
    args = parser.parse_args()
    load_ply = os.path.join(args.destination, "point_cloud", "iteration_" + str(args.iteration), "point_cloud.ply")
    save = os.path.join(args.destination, "ours_{}".format(args.iteration))
    with torch.no_grad():
        dataset, gaussians = prepare_rendering(
            sh_degree=args.sh_degree, source=args.source, device=args.device, trainable_camera=args.mode == "camera",
            load_ply=load_ply, load_camera=args.load_camera, load_depth=True, rescale_factor=args.rescale_factor)
        rendering(dataset, gaussians, save, save_pcd=args.save_depth_pcd, rescale_depth_gt=not args.no_rescale_depth_gt)
