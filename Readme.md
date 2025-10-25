# Neural Network Deduplication Pipeline (nn-dup)

A sophisticated data curation and near-deduplication pipeline for neural network code from the LEMUR dataset. This project implements prefix-aware exact/near/AST deduplication with diversity top-up capabilities.

## Overview

This pipeline processes neural network implementations from the LEMUR dataset, performing:
- **Exact deduplication** with prefix-aware canonicalization
- **Lexical near-deduplication** using MinHash and LSH
- **Structural deduplication** using AST fingerprints
- **Diversity top-up** for underrepresented model families
- **Family-aware train/dev/test splits**

## Features

- **Multi-level Deduplication**: Exact, lexical (MinHash+LSH), and structural (AST) deduplication
- **Prefix-aware Processing**: Maintains representation across different model families
- **Family-aware Splits**: Ensures proper train/dev/test separation by model families
- **Diversity Top-up**: Intelligently adds diverse samples for underrepresented prefixes
- **Comprehensive Reporting**: Detailed statistics and curation reports
- **Code Export**: Exports deduplicated code files for further use

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

## Usage

### Basic Usage

Run the deduplication pipeline with default settings:

```bash
python -m ab.dup.preprocessing --out ./curation_output
```

### Advanced Usage

Filter for specific model families and configure deduplication:

```bash
python -m ab.dup.preprocessing \
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
python -m ab.dup.preprocessing --help
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
  url={https://github.com/your-org/nn-dup}
}
```

## Acknowledgments

- Built for the LEMUR dataset and NNGPT projects
- Developed at the Computer Vision Laboratory, University of Würzburg
- Part of the ABrain One research initiative