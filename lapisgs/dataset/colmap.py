from gaussian_splatting.camera import build_camera
from gaussian_splatting.dataset.colmap import read_colmap_cameras, ColmapCameraDataset
from .dataset import RescaleCameraDatasetIface, RescaleTrainableCameraDataset


class RescaleColmapCameraDataset(RescaleCameraDatasetIface, ColmapCameraDataset):
    def __init__(self, colmap_folder, load_depth=False, rescale_factor=1.0):
        # https://github.com/yindaheng98/gaussian-splatting/blob/56576b647d9c5bd05300f5640cd03a8c75a760bc/gaussian_splatting/dataset/colmap/dataset.py#L89
        self.raw_cameras = read_colmap_cameras(colmap_folder, load_depth=load_depth)
        self.raw_image_hw = [(raw_camera.image_height, raw_camera.image_width) for raw_camera in self.raw_cameras]
        self.raw_cameras = [
            raw_camera._replace(
                image_height=round(h*rescale_factor), image_width=round(w*rescale_factor),
            ) for (h, w), raw_camera in zip(self.raw_image_hw, self.raw_cameras)]
        self.cameras = [build_camera(**cam._asdict(), custom_data=dict(fullimage_width=w, fullimage_height=h)) for (h, w), cam in zip(self.raw_image_hw, self.raw_cameras)]

    def to(self, device):
        # https://github.com/yindaheng98/gaussian-splatting/blob/56576b647d9c5bd05300f5640cd03a8c75a760bc/gaussian_splatting/dataset/colmap/dataset.py#L94
        self.cameras = [build_camera(**cam._asdict(), custom_data=dict(fullimage_width=w, fullimage_height=h), device=device) for (h, w), cam in zip(self.raw_image_hw, self.raw_cameras)]
        return self


def RescaleColmapTrainableCameraDataset(colmap_folder, load_depth=False, rescale_factor=1.0):
    # https://github.com/yindaheng98/gaussian-splatting/blob/56576b647d9c5bd05300f5640cd03a8c75a760bc/gaussian_splatting/dataset/colmap/dataset.py#L105
    return RescaleTrainableCameraDataset(RescaleColmapCameraDataset(colmap_folder, load_depth=load_depth, rescale_factor=rescale_factor))
