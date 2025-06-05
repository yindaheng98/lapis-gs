from gaussian_splatting import CameraTrainableGaussianModel
from gaussian_splatting.camera import build_camera
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.dataset.colmap import read_colmap_cameras

# Modified from https://github.com/yindaheng98/gaussian-splatting/blob/56576b647d9c5bd05300f5640cd03a8c75a760bc/gaussian_splatting/dataset/colmap/dataset.py#L88


class RescaleColmapCameraDataset(CameraDataset):
    def __init__(self, colmap_folder, load_depth=False, rescale_factor=1.0):
        super().__init__()
        self.raw_cameras = [raw_camera._replace(
            image_height=round(raw_camera.image_height*rescale_factor),
            image_width=round(raw_camera.image_width*rescale_factor),
        ) for raw_camera in read_colmap_cameras(colmap_folder, load_depth=load_depth)]
        self.cameras = [build_camera(**cam._asdict()) for cam in self.raw_cameras]

    def to(self, device):
        self.cameras = [build_camera(**cam._asdict(), device=device) for cam in self.raw_cameras]
        return self

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        return self.cameras[idx]


def RescaleColmapTrainableCameraDataset(colmap_folder, load_depth=False, rescale_factor=1.0):
    return TrainableCameraDataset(RescaleColmapCameraDataset(colmap_folder, load_depth=load_depth, rescale_factor=rescale_factor))


def RescaleColmapCameraTrainableGaussianModel(colmap_folder, load_depth=False, rescale_factor=1.0, *args, **kwargs):
    return CameraTrainableGaussianModel(dataset=RescaleColmapTrainableCameraDataset(colmap_folder, load_depth=load_depth, rescale_factor=rescale_factor), *args, **kwargs)
