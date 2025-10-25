# setup.py
from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent.resolve()
README = ROOT / "README.md"
REQUIREMENTS = ROOT / "requirements.txt"

def load_requirements():
    """Load requirements from requirements.txt file."""
    if REQUIREMENTS.exists():
        with open(REQUIREMENTS, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

long_description = README.read_text(encoding="utf-8") if README.exists() else (
    "LEMUR NN curation & dedup pipeline (prefix-aware exact/near/AST dedup, "
    "diversity top-up, and accepted-code export)."
)

setup(
    name="lemur-preprocessing",
    version="0.1.1",
    description="Prefix-aware curation & near-dedup for NN code via MinHash/LSH and AST fingerprints.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Team",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(include=["ab", "ab.*"]),
    include_package_data=True,
    install_requires=load_requirements(),
    extras_require={
        "dev": ["black>=23.0", "isort>=5.0", "flake8>=6.0"],
    },
    entry_points={
        "console_scripts": [
            "lemur-preprocess=ab.dup.preprocessing:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Code Generators",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "LLM", "deduplication", "MinHash", "LSH", "AST", "neural networks",
        "data curation", "pretraining", "fine-tuning",
    ],
    project_urls={
        "Homepage": "https://github.com/your-org/lemur-preprocessing",
        "Bug Tracker": "https://github.com/your-org/lemur-preprocessing/issues",
    },
)
