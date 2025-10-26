# ab/chatprep/consts.py
SYSTEM_POLICY = (
    "You write clean, runnable PyTorch CV architectures. "
    "Output only a single Python code block containing a complete nn.Module. "
    "Respect user-provided resource limits (parameters/FLOPs/latency) and allowed training tricks. "
    "No extra prose outside the code block."
)

DEFAULT_DATASETS = [
    # (name, input_spec)
    ("CIFAR-10", "32x32 RGB"),
    ("Tiny-ImageNet", "64x64 RGB"),
    ("ImageNet-1k", "224x224 RGB"),
    ("MNIST", "28x28 grayscale"),
]

ALLOWED_TRICKS_POOL = [
    "label_smoothing", "cosine_lr", "mixup<=0.2",
    "cutmix<=0.2", "ema", "grad_clip<=3.0",
    "warmup<=500_iters", "dropout<=0.5"
]

# Buckets ~ rough ceilings chosen to be above our static estimates
PARAM_BUCKETS = [0.3e6, 0.8e6, 1.5e6, 3e6, 6e6, 12e6, 25e6]

FAMILIES = ["transformer", "mobile", "resnet", "densenet", "vgg", "fractal", "generic"]
