"""Mitochondria segmentation of refractive-index (RI) images with the 3-fold nnU-Net ensemble."""
import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image

# nnU-Net warns at import time if these are unset; inference does not use them
for _key in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'):
    os.environ.setdefault(_key, '/tmp/nnunet_unused')

import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402

MODEL_REPO = os.environ.get('MODEL_REPO', 'YOUR_HF_USERNAME/mito-nnunet-521')
FOLDS = (0, 1, 2)


def download_model(repo_id: str = MODEL_REPO) -> Path:
    """Download (or reuse the cached copy of) the model folder from the Hugging Face Hub.
    A local folder path is used as is (handy for testing before uploading)."""
    if Path(repo_id).is_dir():
        return Path(repo_id)
    return Path(snapshot_download(repo_id=repo_id, repo_type='model'))


def load_predictor(model_dir: Path, folds=FOLDS, use_mirroring: bool = True) -> nnUNetPredictor:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=use_mirroring,
                                perform_everything_on_device=device.type == 'cuda', device=device,
                                verbose=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=tuple(folds),
                                                   checkpoint_name='checkpoint_final.pth')
    return predictor


def to_gray(image: Image.Image) -> np.ndarray:
    """8-bit grayscale array, the format the model was trained on (nnU-Net NaturalImage2DIO)."""
    return np.asarray(image.convert('L'))


_lock = threading.Lock()


def segment(predictor: nnUNetPredictor, gray: np.ndarray, fast: bool = False) -> np.ndarray:
    """Binary mask (0 background, 1 mitochondria) for an (H, W) uint8 image.
    fast=True uses only the first fold model without mirroring, reusing the loaded ensemble
    (no second copy of the weights in memory)."""
    # NaturalImage2DIO layout: (channels, 1, H, W) with a dummy spacing of 999 on the first axis
    data = gray.astype(np.float32)[None, None]
    with _lock:  # the predictor is shared between app sessions; settings are swapped per call
        params, mirroring = predictor.list_of_parameters, predictor.use_mirroring
        if fast:
            predictor.list_of_parameters, predictor.use_mirroring = params[:1], False
        try:
            seg = predictor.predict_single_npy_array(data, {'spacing': (999, 1, 1)})
        finally:
            predictor.list_of_parameters, predictor.use_mirroring = params, mirroring
    return seg[0].astype(np.uint8)


def overlay(gray: np.ndarray, mask: np.ndarray, color=(255, 0, 255), alpha=0.45) -> np.ndarray:
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    m = mask == 1
    rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(color, dtype=np.float32)
    return rgb.astype(np.uint8)
