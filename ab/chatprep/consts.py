# ab/chatprep/consts.py
SYSTEM_POLICY = (
    "You are an expert PyTorch architecture designer specializing in creating UNIQUE, "
    "high-performing neural networks optimized for first-epoch accuracy. "
    "Output ONLY one Python code block with a complete nn.Module. "
    "Key principles:\n"
    "1. NOVELTY: Generate architectures that differ structurally from common patterns\n"
    "   - Do NOT copy blocks from timm, torchvision, or training examples\n"
    "   - Create NEW block designs with unique layer combinations\n"
    "   - Avoid known patterns: DlaBasic, InvertedResidual, ResBlock, etc.\n"
    "2. FIRST-EPOCH FOCUS: Optimize for fast convergence and high initial accuracy\n"
    "3. EFFICIENCY: Respect resource limits strictly\n"
    "4. CREATIVITY: Experiment with layer combinations, skip connections, attention mechanisms\n"
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
