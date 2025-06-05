import json

from gaussian_splatting.dataset import CameraDataset, JSONCameraDataset
from gaussian_splatting.camera import camera2dict, dict2camera


class RescaleCameraDataset(CameraDataset):
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


class RescaleJSONCameraDataset(JSONCameraDataset):
    def __init__(self, path, load_depth=False, rescale_factor=1.0):
        # https://github.com/yindaheng98/gaussian-splatting/blob/56576b647d9c5bd05300f5640cd03a8c75a760bc/gaussian_splatting/dataset/dataset.py#L38
        with open(path, 'r') as f:
            self.json_cameras = json.load(f)
        self.load_depth = load_depth
        for camera in self.json_cameras:
            camera['width'] = round(camera['fullimage_width'] * rescale_factor)
            camera['height'] = round(camera['fullimage_height'] * rescale_factor)
        self.cameras = [dict2camera(
            camera, load_depth=self.load_depth,
            custom_data=dict(fullimage_width=camera['fullimage_width'], fullimage_height=camera['fullimage_height'])
        ) for camera in self.json_cameras]
