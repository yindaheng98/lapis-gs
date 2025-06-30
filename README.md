# LapisGS: Layered Progressive 3D Gaussian Splatting for Adaptive Streaming (Packaged Python Version)

This repository contains the **refactored Python code for [LapisGS](https://github.com/nus-vv-streams/lapis-gs)**. It is forked from commit [12dcda37ed43838d7407b28675bc26b7364ae431](https://github.com/nus-vv-streams/lapis-gs/tree/12dcda37ed43838d7407b28675bc26b7364ae431). The original code has been **refactored to follow the standard Python package structure**, while **maintaining the same algorithms as the original version**.

## Features

* [x] Code organized as a standard Python package
* [x] Layered progressive 3D Gaussian Splatting
* [x] Multi-resolution training pipeline
* [x] Integration with [reduced-3dgs](https://github.com/yindaheng98/reduced-3dgs)

## Prerequisites

* [Pytorch](https://pytorch.org/) (v2.4 or higher recommended)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-12-4-0-download-archive) (12.4 recommended, should match with PyTorch version)

## Install

### PyPI Install

```shell
pip install --upgrade lapisgs
```

## Install (Development)

Install [`gaussian-splatting`](https://github.com/yindaheng98/gaussian-splatting).
You can download the wheel from [PyPI](https://pypi.org/project/gaussian-splatting/):
```shell
pip install --upgrade gaussian-splatting
```
Alternatively, install the latest version from the source:
```sh
pip install --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master
```

Install [`reduced-3dgs`](https://github.com/yindaheng98/reduced-3dgs).
You can download the wheel from [PyPI](https://pypi.org/project/reduced-3dgs/):
```shell
pip install --upgrade reduced-3dgs
```
Alternatively, install the latest version from the source:
```sh
pip install --upgrade git+https://github.com/yindaheng98/reduced-3dgs.git@main
```

```shell
git clone --recursive https://github.com/yindaheng98/lapis-gs
cd lapis-gs
pip install tqdm plyfile tifffile
pip install --target . --upgrade --no-deps .
```

(Optional) If you prefer not to install `gaussian-splatting` and `reduced-3dgs` in your environment, you can install them in your `lapis-gs` directory:
```sh
pip install --target . --no-deps --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master
pip install --target . --no-deps --upgrade git+https://github.com/yindaheng98/reduced-3dgs.git@main
```

## Quick Start

1. Download dataset (T&T+DB COLMAP dataset, size 650MB):

```shell
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip -P ./data
unzip data/tandt_db.zip -d data/
```

2. Train LapisGS with full pipeline (8x → 4x → 2x → 1x), in this way each layer shares the same training parameters except `rescale_factor`:
```shell
python -m lapisgs.train_full_pipeline_reduced -s data/truck -d output/truck -i 30000 --mode base -olambda_dssim=0.8
```

3. (Optional) Train progressive layers (8x → 4x → 2x → 1x), in this way you can modify the training parameters for each layer:
```shell
# Train 8x (lowest resolution)
python -m lapisgs.train_reduced -s data/truck -d output/truck/8x --rescale_factor 0.125 -i 10000 --mode shculling -olambda_dssim=0.8

# Train 4x (load from 8x)
python -m lapisgs.train_reduced -s data/truck -d output/truck/4x --rescale_factor 0.25 -l output/truck/8x/point_cloud/iteration_10000/point_cloud.ply --load_camera output/truck/8x/cameras.json -i 10000 --mode camera-shculling -olambda_dssim=0.8

# Train 2x (load from 4x)
python -m lapisgs.train_reduced -s data/truck -d output/truck/2x --rescale_factor 0.5 -l output/truck/4x/point_cloud/iteration_10000/point_cloud.ply --load_camera output/truck/4x/cameras.json -i 10000 --mode camera-shculling -olambda_dssim=0.8

# Train 1x (full resolution, load from 2x)
python -m lapisgs.train_reduced -s data/truck -d output/truck/1x --rescale_factor 1.0 -l output/truck/2x/point_cloud/iteration_10000/point_cloud.ply --load_camera output/truck/2x/cameras.json -i 10000 --mode camera-shculling -olambda_dssim=0.8
```

4. Render LapisGS at different resolutions:
```shell
# Render 8x
python -m lapisgs.render -s data/truck -d output/truck/8x -i 10000 --mode base --load_camera output/truck/8x/cameras.json --rescale_factor 0.125

# Render 4x
python -m lapisgs.render -s data/truck -d output/truck/4x -i 10000 --mode camera --load_camera output/truck/4x/cameras.json --rescale_factor 0.25

# Render 2x
python -m lapisgs.render -s data/truck -d output/truck/2x -i 10000 --mode camera --load_camera output/truck/2x/cameras.json --rescale_factor 0.5

# Render 1x (full resolution)
python -m lapisgs.render -s data/truck -d output/truck/1x -i 10000 --mode camera --load_camera output/truck/1x/cameras.json --rescale_factor 1.0
```

> 💡 This repo does not contain code for creating dataset.
> If you want to create your own dataset, please refer to [InstantSplat](https://github.com/yindaheng98/InstantSplat) or use [convert.py](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/convert.py).

> 💡 See [.vscode/launch.json](.vscode/launch.json) for advanced examples. See `lapisgs.train_full_pipeline_reduced` and `lapisgs.train_reduced` for full options.

## API Usage

This project is built on top of [`gaussian-splatting`](https://github.com/yindaheng98/gaussian-splatting) and [`reduced-3dgs`](https://github.com/yindaheng98/reduced-3dgs). Please refer to their documentation for basic usage of Gaussian models, datasets, and trainers.

### Gaussian Models

LapisGS uses the standard Gaussian models from `gaussian-splatting`:

```python
from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel

# Standard Gaussian model
gaussians = GaussianModel(sh_degree).to(device)

# For camera-trainable scenarios
gaussians = CameraTrainableGaussianModel(sh_degree).to(device)
```

### Multi-Resolution Datasets

LapisGS provides rescale-aware dataset classes for multi-resolution training:

```python
from lapisgs.dataset import RescaleColmapCameraDataset, RescaleTrainableCameraDataset

# For standard training
dataset = RescaleColmapCameraDataset(source_path, rescale_factor=0.125, load_depth=True) # 8x
dataset = RescaleColmapCameraDataset(source_path, rescale_factor=0.25, load_depth=True) # 4x
dataset = RescaleColmapCameraDataset(source_path, rescale_factor=0.5, load_depth=True) # 2x
dataset = RescaleColmapCameraDataset(source_path, rescale_factor=1.0, load_depth=True) # 1x
# ... you can use any rescale_factor as you want

# For camera-trainable scenarios
dataset = RescaleTrainableCameraDataset.from_colmap(source_path, rescale_factor=0.125, load_depth=True) # 8x
dataset = RescaleTrainableCameraDataset.from_colmap(source_path, rescale_factor=0.25, load_depth=True) # 4x
dataset = RescaleTrainableCameraDataset.from_colmap(source_path, rescale_factor=0.5, load_depth=True) # 2x
dataset = RescaleTrainableCameraDataset.from_colmap(source_path, rescale_factor=1.0, load_depth=True) # 1x
# ... you can use any rescale_factor as you want

# Load from saved JSON
dataset = RescaleTrainableCameraDataset.from_json(camera_json_path, rescale_factor=0.125, load_depth=True) # 8x
dataset = RescaleTrainableCameraDataset.from_json(camera_json_path, rescale_factor=0.25, load_depth=True) # 4x
dataset = RescaleTrainableCameraDataset.from_json(camera_json_path, rescale_factor=0.5, load_depth=True) # 2x
dataset = RescaleTrainableCameraDataset.from_json(camera_json_path, rescale_factor=1.0, load_depth=True) # 1x
# ... you can use any rescale_factor as you want
```

### LapisGS Trainers

LapisGS provides specialized trainers with partial densification and opacity reset:

```python
from lapisgs.trainer import LapisTrainer, DepthLapisTrainer, LapisCameraTrainer, DepthLapisCameraTrainer

# Basic LapisGS trainer
trainer = LapisTrainer(
    gaussians,
    scene_extent=dataset.scene_extent(),
    # ... other parameters
)

# LapisGS trainer with depth regularization
trainer = DepthLapisTrainer(
    gaussians,
    scene_extent=dataset.scene_extent(),
    # ... other parameters
)

# LapisGS trainer with camera optimization
trainer = LapisCameraTrainer(
    gaussians,
    scene_extent=dataset.scene_extent(),
    dataset=dataset,
    # ... other parameters
)

# LapisGS trainer with both depth and camera optimization
trainer = DepthLapisCameraTrainer(
    gaussians,
    scene_extent=dataset.scene_extent(),
    dataset=dataset,
    # ... other parameters
)
```

### Training Pipeline

```python
from lapisgs.prepare import prepare_dataset, prepare_trainer
from reduced_3dgs.prepare import prepare_gaussians

# Prepare components for training
dataset = prepare_dataset(
    source=source_path,
    device=device,
    trainable_camera=True,
    load_camera=camera_json_path,
    rescale_factor=0.5
)

gaussians = prepare_gaussians(
    sh_degree=3,
    source=source_path,
    device=device,
    trainable_camera=True,
    load_ply=foundation_ply_path
)

trainer = prepare_trainer(
    gaussians=gaussians,
    dataset=dataset,
    mode="camera",  # "base", "camera", "nodepth-base", "nodepth-camera"
    trainable_camera=True,
    load_ply=foundation_ply_path
)

# Training loop
for camera in dataset:
    loss, out = trainer.step(camera)
```

### How to extract the enhanced layer

Note that \<scene\>_res1 is the highest resolution, and \<scene\>_res8 is the lowest resolution. The model is trained from the lowest resolution to the highest resolution. The model stored in the higher resolution folder contains not only the higher layer but also the lower layer(s).

We construct the merged GS with a specially designed order: the lower layers come first as the foundation base, and the enhanced layer is stiched behind the foundation base, as shown in the figure below. As the foundation base is frozen to optimization and adaptive control, one can easily extract the enhanced layer by performing the operation like GS[size_of_foundation_layers:].

<p align="center">
    <a href="">
        <img src="images/model_structure.png" alt="model_structure" width="70%">
    </a>
</p>

### CUDA out-of-memory error

Through experiments, we found that the default loss function is not sensitive to low-resolution images, making optimization and desification failed. It is because in the default loss function, L1 loss is attached much more importance (0.8), but L1 loss is not sensitive to finer details, blurriness, or low-resolution artifacts. Therefore, the loss, computed from the default loss function, would be small at low layers, disabling the parameter update and adaptive control for the low-layer Gaussian splats. Therefore, we set lambda_dssim to 0.8 to emphasize the structural similarity loss, which is more sensitive to low-resolution artifacts and then causes much heavier desification, finally producing bigger 3DGS model. 

To reduce the model size, you may try to 1) lower down the lambda_dssim, or 2) increase the densification threshold. Also, generally speaking, it is not necessary to make it SSIM-sensitive for complex scenes. For example, we note that training LapisGS for complex scene *playroom* with default lambda_dssim 0.2 can still produce reasonable layered structure, while it fails for simple object *lego*.

<div align="center">
    <h1>
        <img src="images/title.png" alt="icon" style="height: 1em; vertical-align: middle; margin-right: 0.1em;">
        <strong>LapisGS: </strong>Layered Progressive 3D Gaussian Splatting for Adaptive Streaming
    </h1>
</div>

<div align="center">
    <a href="https://yuang-ian.github.io" target='_blank'>Yuang Shi</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=PbKu-PsAAAAJ&hl=en" target='_blank'>Simone Gasparini</a><sup>2</sup>,
    <a href="https://scholar.google.de/citations?user=H8QDhhAAAAAJ&hl=en" target='_blank'>Géraldine Morin</a><sup>2</sup>,
    <a href="https://www.comp.nus.edu.sg/~ooiwt/" target='_blank'>Wei Tsang Ooi</a><sup>1</sup>,
    <p>
        <sup>1</sup>National University of Singapore,
        <sup>2</sup>IRIT - Université de Toulouse
    </p>
    <p>
    International Conference on 3D Vision (3DV), 2025
    </p>
</div>


<div align="center">
    <a href="http://arxiv.org/abs/2408.14823" target='_blank'>
        <img src="https://img.shields.io/badge/Paper-%F0%9F%93%83-blue">
    </a>
    <a href="https://yuang-ian.github.io/lapisgs/" target='_blank'>
        <img src="https://img.shields.io/badge/Project-%F0%9F%94%97-yellow">
    </a>
</div> <br> <br>



<p align="center">
  <a href="">
    <img src="images/teaser.png" alt="teaser" width="80%">
  </a>
</p>

<p align="center">
    We introduce <strong><i>LapisGS</i></strong>*, a layered progressive 3DGS, for adaptive streaming and view-adaptive rendering. 
</p>

<p align="center">
    <span class="small">
        *<i>Lapis</i> means "layer" in Malay, the national language of Singapore --- the host of 3DV'25. The logo in the title depicts <a href="https://en.wikipedia.org/wiki/Kue_lapis">kuih lapis</a>, or "layered cake", a local delight in Singapore and neighboring countries. The authors are glad to serve kuih lapis to our friends at the conference to share the joy of the layered approach 🥳.
    </span>
</p>
<br>

If you find our code or paper useful, please cite

```latex
@inproceedings{shi2024lapisgs,
  author    = {Shi, Yuang and Gasparini, Simone and Morin, Géraldine and Ooi, Wei Tsang},
  title     = {{LapisGS}: Layered Progressive {3D Gaussian} Splatting for Adaptive Streaming},
  publisher = {{IEEE}},
  booktitle = {International Conference on 3D Vision, 3DV 2025, Singapore, March 25-28, 2025},
  year      = {2025},
  }
```

Based on our LapisGS, we built the first ever dynamic 3DGS streaming system, which achieves superior performance in both live streaming and on-demand streaming. Our work is to be appeared in the MMSys'25 in March 2025. Access to the [Preprint Paper](https://drive.google.com/file/d/1iDz1ExOd1LrPhA7fv4DbLUbzn-Jioihn/view?usp=share_link).

```latex
@inproceedings{sun2025lts,
  author    = {Sun, Yuan-Chun and Shi, Yuang and Lee, Cheng-Tse and Zhu, Mufeng and Ooi, Wei Tsang and Liu, Yao and Huang, Chun-Ying and Hsu, Cheng-Hsin},
  title     = {{LTS}: A {DASH} Streaming System for Dynamic Multi-Layer {3D Gaussian} Splatting Scenes},
  publisher = {{ACM}},
  booktitle = {The 16th ACM Multimedia Systems Conference, MMSys 2025, 2025},
  year      = {2025},
  }
```