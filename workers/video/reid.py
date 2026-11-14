"""Appearance embeddings for person crops, used to link track fragments offline.

The default encoder reuses the football detector's own backbone, so no extra weights
are needed. It was trained to find players, not to tell them apart, so treat it as a
baseline; a ReID model trained on sports identities can replace it behind `embed`.
"""
from __future__ import annotations

import cv2
import numpy as np


class BackboneEmbedder:
    link_settings = dict(centre_embeddings=True, max_embedding_distance=0.9, embedding_weight=0.5)

    def __init__(self, weights, device="cpu", layers=(2, 4), size=(256, 128), batch=64):
        import torch
        from ultralytics import YOLO

        self.torch = torch
        self.net = YOLO(str(weights)).model.to(device if device == "cpu" else f"cuda:{device}").eval()
        self.device = next(self.net.parameters()).device
        self.layers, self.size, self.batch = list(layers), size, batch
        self.name = f"detector-backbone-layers-{'-'.join(map(str, layers))}"

    def embed(self, crops):
        """L2-normalised embedding per BGR crop."""
        if not crops:
            return np.zeros((0, 0), np.float32)
        height, width = self.size
        out = []
        for start in range(0, len(crops), self.batch):
            images = np.stack([cv2.cvtColor(cv2.resize(c, (width, height)), cv2.COLOR_BGR2RGB) for c in crops[start:start + self.batch]])
            tensor = self.torch.from_numpy(images).to(self.device).permute(0, 3, 1, 2).float() / 255
            with self.torch.no_grad():
                features = self.torch.stack(list(self.net.predict(tensor, embed=self.layers)))
            out.append(self.torch.nn.functional.normalize(features, dim=1).cpu().numpy())
        return np.concatenate(out)


class OSNetEmbedder:
    """OSNet x1.0 person ReID (512-d). `source` is torchreid's osnet.py; `weights` a torchreid checkpoint."""

    # Measured on dfl_test_10 with the SportsMOT checkpoint: same-player pairs sit below about 0.16.
    link_settings = dict(centre_embeddings=False, max_embedding_distance=0.25, embedding_weight=2.0)

    def __init__(self, weights, source, device="cpu", size=(256, 128), batch=64):
        import importlib.util
        import torch

        spec = importlib.util.spec_from_file_location("osnet_definition", str(source))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.torch = torch
        # torchreid checkpoints store metrics as numpy scalars. Allow only those numpy
        # constructors; never fall back to unrestricted unpickling of downloaded weights.
        import numpy.dtypes
        from numpy.core.multiarray import scalar
        allowed = [(scalar, "numpy.core.multiarray.scalar"), scalar, np.dtype] + [getattr(numpy.dtypes, name) for name in dir(numpy.dtypes) if name.endswith("DType")]
        with torch.serialization.safe_globals(allowed):
            checkpoint = torch.load(str(weights), map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
        self.net = module.osnet_x1_0(num_classes=1, pretrained=False, loss="softmax")
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith("classifier.") for k in missing):
            raise ValueError(f"Checkpoint does not match OSNet x1.0: missing {missing}, unexpected {unexpected}")
        self.device = torch.device("cpu" if device == "cpu" else f"cuda:{device}")
        self.net = self.net.to(self.device).eval()  # eval mode returns the 512-d feature
        self.size, self.batch = size, batch
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.name = f"osnet_x1_0:{getattr(weights, 'name', str(weights))}"

    def embed(self, crops):
        """L2-normalised embedding per BGR crop."""
        if not crops:
            return np.zeros((0, 512), np.float32)
        height, width = self.size
        out = []
        for start in range(0, len(crops), self.batch):
            images = np.stack([cv2.cvtColor(cv2.resize(c, (width, height)), cv2.COLOR_BGR2RGB) for c in crops[start:start + self.batch]])
            tensor = (self.torch.from_numpy(images).to(self.device).permute(0, 3, 1, 2).float() / 255 - self.mean) / self.std
            with self.torch.no_grad():
                out.append(self.torch.nn.functional.normalize(self.net(tensor), dim=1).cpu().numpy())
        return np.concatenate(out)
