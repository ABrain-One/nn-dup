# ab/chatprep/consts.py
# -*- coding: utf-8 -*-

"""
Constants for chat-prep and prompt construction.

Goals
- Strict, deterministic system policy (prevents format drift during SFT).
- Channels-first shapes for easy dummy-tensor validation.
- Dataset registry with modality, tasks, shapes, and default class counts.
- Conservative tricks pool with explicit caps.
- Resource buckets spanning micro-nets to mid-scale backbones.
- Explicit style families for steerable prompting.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

# ---------------------------
# System / formatting policy
# ---------------------------

SYSTEM_POLICY = (
    "You write clean, runnable PyTorch CV architectures. "
    "Output ONLY one Python code block that defines exactly one nn.Module "
    "(use class Net unless a different name is given). "
    "Begin the answer with ```python and end the block; no text before or after. "
    "Use only torch/torch.nn/torch.nn.functional; no external deps, no torchvision.models, "
    "no file/network I/O, no prints. "
    "Respect user resource limits (parameters/FLOPs/latency) and the dataset input shape (channels-first). "
    "Do NOT implement training tricks (mixup/cutmix/EMA/warmup/label smoothing/etc.) inside the module; "
    "training scripts handle them."
)

CODE_FENCE_BEGIN = "```python"
CODE_FENCE_END = "```"

ALLOWED_IMPORTS = ("torch", "torch.nn", "torch.nn.functional")
BANNED_IMPORTS = ("torchvision.models",)

# ---------------------------
# Dataset registry
# ---------------------------
# Each entry:
#   name: {
#       "modality": "image" | "text",
#       "tasks": ["classification", "detection", "segmentation", "captioning", "generative"],
#       "input_shape": "CxHxW" (image) or None (text),
#       "default_num_classes": {task: int}  # when sensible (e.g., classification, detection)
#       "notes": short tip for preprocessing/training size
#   }
#
# Shapes here are *typical training sizes* for CV; raw assets may vary.

DATASETS: Dict[str, Dict] = {
    # 1) MNIST — 28x28 grayscale, 10 classes
    "MNIST": {
        "modality": "image",
        "tasks": ["classification"],
        "input_shape": "1x28x28",
        "default_num_classes": {"classification": 10},
        "notes": "Native 28x28 grayscale.",
    },
    # 2) CIFAR-10 — 32x32 RGB, 10 classes
    "CIFAR-10": {
        "modality": "image",
        "tasks": ["classification"],
        "input_shape": "3x32x32",
        "default_num_classes": {"classification": 10},
        "notes": "Native 32x32 color images.",
    },
    # 3) CIFAR-100 — 32x32 RGB, 100 classes
    "CIFAR-100": {
        "modality": "image",
        "tasks": ["classification"],
        "input_shape": "3x32x32",
        "default_num_classes": {"classification": 100},
        "notes": "100 fine-grained classes (20 coarse groups).",
    },
    # 4) COCO — multi-task (detection/segmentation/captioning; sometimes used for text-to-image pairs)
    "COCO": {
        "modality": "image",
        "tasks": ["detection", "segmentation", "captioning", "generative"],
        "input_shape": "3x640x640",
        "default_num_classes": {
            "detection": 80,        # 80 object categories
            "segmentation": 80      # instance seg; semantic 'stuff' differs by variant
        },
        "notes": "Images vary; 640x640 is a common training resolution. Captioning vocab is dataset-dependent.",
    },
    # 5) Imagenette — 10 classes; standard ImageNet-sized training
    "Imagenette": {
        "modality": "image",
        "tasks": ["classification"],
        "input_shape": "3x224x224",
        "default_num_classes": {"classification": 10},
        "notes": "FastAI subset; 160px/320px variants exist; 224x224 is common.",
    },
    # 6) Places365 — 365 scene categories
    "Places365": {
        "modality": "image",
        "tasks": ["classification"],
        "input_shape": "3x256x256",
        "default_num_classes": {"classification": 365},
        "notes": "Scene recognition; 224–256 crop pipelines are standard.",
    },
    # 7) SVHN — 32x32 RGB, digits (10 classes)
    "SVHN": {
        "modality": "image",
        "tasks": ["classification"],
        "input_shape": "3x32x32",
        "default_num_classes": {"classification": 10},
        "notes": "Street View house numbers; more challenging than MNIST.",
    },
    # 8) CelebA-Gender — binary classification derived from CelebA attributes
    "CelebA-Gender": {
        "modality": "image",
        "tasks": ["classification"],
        "input_shape": "3x224x224",
        "default_num_classes": {"classification": 2},
        "notes": "CelebA faces usually 178x218; center-crop/resize to 224x224.",
    },
    # 9) WikiText-2-raw — language modeling (text)
    "WikiText-2-raw": {
        "modality": "text",
        "tasks": ["language-modeling"],
        "input_shape": None,
        "default_num_classes": {},
        "notes": "Text corpus; out-of-scope for CV nn.Module generation.",
    },
}

# A compact, CV-only default list (channels-first shapes) for simple prompts.
DEFAULT_DATASETS: List[Tuple[str, str]] = [
    ("MNIST", "1x28x28"),
    ("CIFAR-10", "3x32x32"),
    ("CIFAR-100", "3x32x32"),
    ("SVHN", "3x32x32"),
    ("Imagenette", "3x224x224"),
    ("Places365", "3x256x256"),
    # COCO is multi-task; shape here is a *typical* training size
    ("COCO", "3x640x640"),
    # CelebA-Gender is derived from CelebA; we normalize to 224^2
    ("CelebA-Gender", "3x224x224"),
]

# Legacy (human-readable) spec for any back-compat code that still expects it.
DEFAULT_DATASETS_LEGACY = [
    ("MNIST", "28x28 grayscale"),
    ("CIFAR-10", "32x32 RGB"),
    ("CIFAR-100", "32x32 RGB"),
    ("SVHN", "32x32 RGB"),
    ("Imagenette", "224x224 RGB"),
    ("Places365", "256x256 RGB"),
    ("COCO", "variable (commonly 640x640 RGB)"),
    ("CelebA-Gender", "224x224 RGB"),
]

# ---------------------------
# Training tricks policy
# ---------------------------

ALLOWED_TRICKS_POOL = [
    "label_smoothing",
    "cosine_lr",
    "mixup<=0.2",
    "cutmix<=0.2",
    "ema",
    "grad_clip<=3.0",
    "warmup<=500_iters",
    "dropout<=0.5",
    # Architecture-level regularizer that can live inside the module if desired:
    "stochastic_depth<=0.2",
]

# ---------------------------
# Resource buckets
# ---------------------------

PARAM_BUCKETS = [0.1e6, 0.3e6, 0.8e6, 1.5e6, 3e6, 6e6, 12e6, 25e6, 50e6]
FLOP_BUCKETS = [50e6, 150e6, 300e6, 600e6, 1_200e6]

# ---------------------------
# Style / family steering
# ---------------------------

FAMILIES: List[str] = [
    "resnet",
    "densenet",
    "vgg",
    "mobilenet",
    "efficientnet",
    "convnext",
    "vit",
    "mlp-mixer",
    "fractal",
    "generic",
]
