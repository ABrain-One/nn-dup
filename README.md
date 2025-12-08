# Neural Network Deduplication Pipeline (NN Dup)
<sub><a href='https://pypi.python.org/pypi/nn-dup'><img src='https://img.shields.io/pypi/v/nn-dup.svg'/></a> <a href="https://pepy.tech/project/nn-dup"><img alt="GitHub release" src="https://static.pepy.tech/badge/nn-dup"></a><br/>
short alias <a href='https://pypi.python.org/pypi/ldup'>ldup</a></sub>

The original version of the NN Dup project was created by <strong>Waleed Khalid</strong> at the Computer Vision Laboratory, University of Würzburg, Germany.

## 📖 Overview

A sophisticated data curation and near-deduplication pipeline for neural network code from the <a href='https://github.com/ABrain-One/nn-dataset'>LEMUR dataset</a>. This project implements prefix-aware exact/near/AST deduplication with diversity top-up capabilities, followed by conversational chat data preparation for language model training. Outputs include train/dev/test JSON files ready for supervised fine-tuning.

This comprehensive pipeline processes neural network implementations from the LEMUR dataset through two main stages:

### Stage 1: Deduplication Pipeline
- **Exact deduplication** with prefix-aware canonicalization
- **Lexical near-deduplication** using MinHash and LSH
- **Structural deduplication** using AST fingerprints
- **Diversity top-up** for underrepresented model families
- **Family-aware train/dev/test splits**

### Stage 2: ChatPrep Pipeline
- **Conversational conversion** of deduplicated code into chat format
- **Template-based generation** of user-assistant interactions
- **Quality validation** following SFT standards
- **JSONL export** for language model training

## Features

### Deduplication Pipeline
- **Multi-level Deduplication**: Exact, lexical (MinHash+LSH), and structural (AST) deduplication
- **Prefix-aware Processing**: Maintains representation across different model families
- **Family-aware Splits**: Ensures proper train/dev/test separation by model families
- **Diversity Top-up**: Intelligently adds diverse samples for underrepresented prefixes
- **Comprehensive Reporting**: Detailed statistics and curation reports
- **Code Export**: Exports deduplicated code files for further use

### ChatPrep Pipeline
- **Conversational Format**: Converts code into chat-style interactions for LLM training
- **Template-based Generation**: Uses customizable templates for consistent formatting
- **Infill Support**: Generates partial code examples for completion tasks
- **Validation & Filtering**: Ensures high-quality, parseable examples following SFT standards
- **Family-aware Splitting**: Prevents data leakage across model families
- **JSONL Export**: Generates training-ready chat data in standard format

## Installation

### Prerequisites
- Python 3.9+
- CUDA 12.6 (for PyTorch compatibility)

### Setup

1. **Create and activate virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

### Install directly from GitHub

Install the latest version directly from the GitHub repository:

```bash
pip install git+https://github.com/ABrain-One/nn-dup.git
```

## Usage

### Basic Usage

Run the deduplication pipeline with default settings:

```bash
python -m ab.dup.main --out ./curation_output
```

### Advanced Usage

Filter for specific model families and configure deduplication:

```bash
python -m ab.dup.main \
    --out ./curation_output \
    --include FractalNet \
    --include ResNet \
    --min-per-prefix 10 \
    --keep-per-family 5 \
    --lex-thresh-fractal 0.97 \
    --verbose
```

### Command Line Options

- `--out`: Output directory (default: `./curation_output`)
- `--include`: Prefix filters for model names (repeatable)
- `--prefer-prefix-order`: Priority order for canonicalization
- `--min-per-prefix`: Minimum records per prefix after dedup
- `--keep-per-family`: Maximum exemplars per family in clusters
- `--lex-thresh-fractal`: Jaccard threshold for FractalNet family
- `--topup-prefix`: Enable diversity top-up for specific prefixes
- `--topup-per-prefix`: Maximum top-up records per prefix
- `--topup-lex-max`: Maximum lexical similarity for top-up
- `--topup-struct-max`: Maximum structural similarity for top-up
- `--dump-accepted-code-dir`: Subdirectory for exported code files
- `--upweight`: Sampling weight rules (PREFIX:FACTOR)
- `--verbose`: Enable verbose logging

## ChatPrep: Converting Code to Chat Data

The ChatPrep module converts deduplicated neural network code into structured chat data suitable for training language models. It generates conversational examples with system prompts, user requests, and model responses.

### ChatPrep Usage

#### Python API

```python
from ab.chatprep import ChatPrepConfig

# Basic usage with defaults
config = ChatPrepConfig()
result = config.run()

# Custom configuration
config = ChatPrepConfig(
    accepted_dir="../curation_output/accepted_code",
    out_dir="../curation_output/chat_data",
    seed=123,
    fix_fences=True,
    drop_unparseable=True,
    group_by_source=True
)
result = config.run()
```

#### Command Line Interface

```bash
python -m ab.chatprep.cli.main \
    --accepted-dir ../curation_output/accepted_code \
    --out ../curation_output/chat_data \
    --seed 123 \
    --fix-fences \
    --drop-unparseable \
    --group-by-source
```

### ChatPrep Configuration Parameters

- `accepted_dir`: Directory with accepted code files (default: `"curation_output/accepted_code"`)
- `out_dir`: Output directory for chat data (default: `"curation_output/chat_data"`)
- `no_infill`: Disable infill generation (default: `False`)
- `seed`: Random seed for reproducibility (default: `42`)
- `fix_fences`: Fix code fences in generated examples (default: `True`)
- `drop_unparseable`: Drop unparseable examples (default: `True`)
- `require_module_subclass`: Require module subclass structure (default: `True`)
- `write_drop_report`: Write report of dropped examples (default: `True`)
- `group_by_source`: Group by source for splitting (default: `True`)
- `split_ratios`: Train/dev/test ratios (default: `(0.8, 0.1, 0.1)`)
- `model_name`: Optional model name for rendering (default: `None`)
- `train_name`: Training file name (default: `"train.jsonl"`)
- `dev_name`: Dev file name (default: `"dev.jsonl"`)
- `test_name`: Test file name (default: `"test.jsonl"`)

### ChatPrep Output Files

The ChatPrep pipeline generates:

- **`train.jsonl`**: Training examples in JSONL format
- **`dev.jsonl`**: Development/validation examples
- **`test.jsonl`**: Test examples
- **`manifest.json`**: Metadata about the generated dataset
- **`drop_report.json`**: Report of examples that were filtered out

### Example Chat Data Format

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant that generates PyTorch code for computer vision tasks."
    },
    {
      "role": "user", 
      "content": "Task: Design a PyTorch CV model for image classification.\nDataset: CIFAR-10 (32x32x3).\nParameters: ≤1M.\nTricks: batch normalization, dropout, data augmentation."
    },
    {
      "role": "assistant",
      "content": "```python\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nclass CIFAR10Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)\n        self.bn1 = nn.BatchNorm2d(32)\n        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)\n        self.bn2 = nn.BatchNorm2d(64)\n        self.dropout = nn.Dropout(0.25)\n        self.fc = nn.Linear(64 * 8 * 8, 10)\n    \n    def forward(self, x):\n        x = F.relu(self.bn1(self.conv1(x)))\n        x = F.max_pool2d(x, 2)\n        x = F.relu(self.bn2(self.conv2(x)))\n        x = F.max_pool2d(x, 2)\n        x = self.dropout(x)\n        x = x.view(x.size(0), -1)\n        x = self.fc(x)\n        return x\n```"
    }
  ]
}
```

## Configuration

Key parameters in `ab/dup/consts/settings.py`:

- `SHINGLE_K = 10`: Token shingle length
- `NUM_PERM = 256`: MinHash permutations
- `LSH_THRESH = 0.85`: LSH retrieval threshold
- `JACCARD_THRESH_LEX = 0.90`: Lexical similarity threshold
- `JACCARD_THRESH_STRUCT = 0.90`: Structural similarity threshold
- `SPLIT_RATIOS = (0.80, 0.10, 0.10)`: Train/dev/test ratios

## Output Files

The pipeline generates several output files:

- **`kept_records.json`**: Metadata for kept records
- **`tombstones.json`**: Metadata for removed records
- **`splits.json`**: Train/dev/test assignments
- **`dedup_report.md`**: Comprehensive curation report
- **`accepted_code/`**: Directory with deduplicated Python files
- **`sampling_weights.csv`**: Optional sampling weights

## Example Report

```
# Curation Report (LEMUR API)

## Summary
- Total rows fetched from LEMUR: **115,127**
- Exact duplicates removed: **104,804**
- Lexical near-duplicates removed: **8,939**
- Structural duplicates removed: **320**
- **Kept for training/eval:** **1,064** records

## Parameters
- Shingle length (k): `10`, MinHash permutations: `256`
- Lexical Jaccard verify (generic): `0.9`, (Fractal): `0.97`
- Keep per family (K): `5`, Min per prefix: `1`
- Train/dev/test ratios: `(0.8, 0.1, 0.1)`
```

## Development

### Running Tests

```bash
python -m ab.dup.main --help
```

### Code Quality

```bash
pip install -e ".[dev]"
black ab/
isort ab/
flake8 ab/
```

## Dependencies

- `nn-dataset>=2.1.0`: LEMUR dataset access
- `datasketch`: MinHash and LSH implementations
- `pandas>=1.3,<3.0`: Data manipulation
- `scipy`: Scientific computing
- `scikit-learn`: Machine learning utilities

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{nn_dup_2025,
  title={Neural Network Deduplication Pipeline},
  author={Waleed Khalid},
  year={2025},
  url={https://github.com/ABrain-One/nn-dup}
}
```

## Acknowledgments

- Built for the <a href='https://github.com/ABrain-One/nn-dataset'>LEMUR dataset</a> and <a href='https://github.com/ABrain-One/nn-gpt'>NNGPT</a> projects
- Developed at the Computer Vision Laboratory, University of Würzburg
- Part of the ABrain One research initiative
