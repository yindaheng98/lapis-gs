import torch
import json

from gaussian_splatting.dataset import JSONCameraDataset, TrainableCameraDataset, FixedTrainableCameraDataset
from gaussian_splatting.camera import camera2dict, dict2camera, Camera
from gaussian_splatting.utils import quaternion_to_matrix


class RescaleCameraDatasetIface:
    # Override save_cameras in CameraDataset for all Rescale datasets
    def save_cameras(self, path):
        # https://github.com/yindaheng98/gaussian-splatting/blob/56576b647d9c5bd05300f5640cd03a8c75a760bc/gaussian_splatting/dataset/dataset.py#L24
        cameras = []
        for id, camera in enumerate(self):
            cameradict = camera2dict(camera, id)
            fullimage_width, fullimage_height = camera.image_width, camera.image_height
            if "fullimage_width" in camera.custom_data and "fullimage_height" in camera.custom_data:
                fullimage_width = camera.custom_data["fullimage_width"]
                fullimage_height = camera.custom_data["fullimage_height"]
            cameradict["fullimage_width"] = fullimage_width
            cameradict["fullimage_height"] = fullimage_height
            cameras.append(cameradict)
        with open(path, 'w') as f:
            json.dump(cameras, f, indent=2)


class RescaleJSONCameraDataset(RescaleCameraDatasetIface, JSONCameraDataset):
    def __init__(self, path, load_depth=False, rescale_factor=1.0):
        # https://github.com/yindaheng98/gaussian-splatting/blob/ae0e1d03349e906f0e7ad08b9b5506feb81cd57d/gaussian_splatting/dataset/dataset.py#L38
        with open(path, 'r') as f:
            self.json_cameras = json.load(f)
        self.load_depth = load_depth
        self.rescale_factor = rescale_factor
        self.load_cameras()

    def load_cameras(self, device=None):
        # https://github.com/yindaheng98/gaussian-splatting/blob/ae0e1d03349e906f0e7ad08b9b5506feb81cd57d/gaussian_splatting/dataset/dataset.py#L53
        for camera in self.json_cameras:
            if 'fullimage_width' not in camera:
                camera['fullimage_width'] = camera['width']
            if 'fullimage_height' not in camera:
                camera['fullimage_height'] = camera['height']
            camera['fx'] = camera['fx'] / camera['width'] * camera['fullimage_width'] * self.rescale_factor
            camera['fy'] = camera['fy'] / camera['height'] * camera['fullimage_height'] * self.rescale_factor
            camera['width'] = round(camera['fullimage_width'] * self.rescale_factor)
            camera['height'] = round(camera['fullimage_height'] * self.rescale_factor)
        self.cameras = [dict2camera(
            camera, load_depth=self.load_depth, device=device,
            custom_data=dict(fullimage_width=camera['fullimage_width'], fullimage_height=camera['fullimage_height'])
        ) for camera in self.json_cameras]
        return self


class RescaleTrainableCameraDataset(TrainableCameraDataset):
    def save_cameras(self, path):
        # https://github.com/yindaheng98/gaussian-splatting/blob/56576b647d9c5bd05300f5640cd03a8c75a760bc/gaussian_splatting/dataset/camera_trainable.py#L68
        cameras = []
        for idx, camera in enumerate(self):
            cameras.append({
                **camera2dict(Camera(**{
                    **camera._asdict(),
                    'R': quaternion_to_matrix(self.quaternions[idx, ...]),
                    'T': self.Ts[idx, ...],
                }), idx),
                "exposure": self.exposures[idx, ...].detach().tolist(),
                "fullimage_width": camera.custom_data["fullimage_width"],
                "fullimage_height": camera.custom_data["fullimage_height"],
            })
        with open(path, 'w') as f:
            json.dump(cameras, f, indent=2)

    @classmethod
    def from_json(cls, path, load_depth=False, rescale_factor=1.0):
        # https://github.com/yindaheng98/gaussian-splatting/blob/56576b647d9c5bd05300f5640cd03a8c75a760bc/gaussian_splatting/dataset/camera_trainable.py#L83
        cameras = RescaleJSONCameraDataset(path, load_depth=load_depth, rescale_factor=rescale_factor)
        exposures = [(torch.tensor(camera['exposure'], dtype=torch.float) if 'exposure' in camera else torch.eye(3, 4)) for camera in cameras.json_cameras]
        return cls(cameras, exposures)


class RescaleFixedTrainableCameraDataset(RescaleJSONCameraDataset):
    # Same as TrainableCameraDataset, but is fixed
    # Used for loading cameras saved by TrainableCameraDataset

    def __init__(self, path, load_depth=False, rescale_factor=1.0):
        # https://github.com/yindaheng98/gaussian-splatting/blob/ae0e1d03349e906f0e7ad08b9b5506feb81cd57d/gaussian_splatting/dataset/camera_trainable.py#L93
        super().__init__(path, load_depth=load_depth, rescale_factor=rescale_factor)
        self.load_exposures()

    def to(self, device):
        # https://github.com/yindaheng98/gaussian-splatting/blob/ae0e1d03349e906f0e7ad08b9b5506feb81cd57d/gaussian_splatting/dataset/camera_trainable.py#L97
        self = super().to(device)
        return self.load_exposures(device=device)

    def load_exposures(self, device=None):
        # https://github.com/yindaheng98/gaussian-splatting/blob/ae0e1d03349e906f0e7ad08b9b5506feb81cd57d/gaussian_splatting/dataset/camera_trainable.py#L101
        return FixedTrainableCameraDataset.load_exposures(self, device=device)

    def save_cameras(self, path):
        # https://github.com/yindaheng98/gaussian-splatting/blob/ae0e1d03349e906f0e7ad08b9b5506feb81cd57d/gaussian_splatting/dataset/camera_trainable.py#L111
        cameras = []
        for idx, camera in enumerate(self):
            cameras.append({
                **camera2dict(camera, idx),
                "exposure": camera.custom_data['exposures'].detach().tolist(),
                "fullimage_width": camera.custom_data["fullimage_width"],
                "fullimage_height": camera.custom_data["fullimage_height"],
            })
        with open(path, 'w') as f:
            json.dump(cameras, f, indent=2)
