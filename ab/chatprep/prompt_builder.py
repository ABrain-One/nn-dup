# ab/chatprep/prompt_builder.py
from __future__ import annotations

import re
import ast
import random
import hashlib
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm

from .consts import SYSTEM_POLICY, DEFAULT_DATASETS, ALLOWED_TRICKS_POOL, PARAM_BUCKETS
from .utils.code_io import load_py_files
from .example_builder import build_examples_from_code
from .schema import write_jsonl
from .renderer import render_with_template
from .utils.split_utils import stratified_split_by_family
from .utils.manifest import write_manifest

# Prefer a source-aware split if present; fallback is defined below.
try:
    from .utils.split_utils import stratified_family_split_by_source  # type: ignore
    _HAS_SOURCE_SPLIT = True
except Exception:
    _HAS_SOURCE_SPLIT = False


# ----------------------------
# Message construction helpers
# ----------------------------
def _pick_dataset(family: str) -> Tuple[str, str]:
    if family in {"mobile", "vgg"}:
        return random.choice(DEFAULT_DATASETS[:3])
    return random.choice(DEFAULT_DATASETS)

def _bucket_cap(n_params: int) -> int:
    for cap in PARAM_BUCKETS:
        if n_params <= cap:
            return int(cap)
    return int(PARAM_BUCKETS[-1])

def build_messages(example_id: str, code_text: str, det: Dict[str, Any]) -> Dict[str, Any]:
    """Build training messages with deterministic seeding to prevent duplicate prompts."""
    # Use deterministic seeding based on example_id to prevent duplicates
    seed = int(hashlib.md5(example_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    
    # Deterministic dataset selection
    ds_name, ds_shape = _pick_dataset_deterministic(det["family"], rng)
    
    est = max(det["param_estimate"], 50_000)
    cap = _bucket_cap(int(est * 1.3))
    
    # Deterministic tricks selection
    tricks = ", ".join(sorted(rng.sample(ALLOWED_TRICKS_POOL, k=min(3, len(ALLOWED_TRICKS_POOL)))))

    user_text = (
        f"Task: Design a PyTorch CV model for image classification.\n"
        f"Dataset: {ds_name} ({ds_shape}).\n"
        f"Resource limits: params ≤ {cap:.0f}; latency budget: tight (edge-friendly).\n"
        f"Constraints: use standard layers only; no pretrained weights.\n"
        f"Allowed training tricks (handled by trainer): {tricks}.\n"
        f"**Goal**: Generate a UNIQUE architecture NOT seen in training data. "
        f"Target MAXIMUM accuracy after the FIRST epoch of training.\n"
        f"Output contract: one Python code block defining a complete nn.Module (e.g., class Net(nn.Module))."
    )
    return {
        "system": SYSTEM_POLICY,
        "user": user_text,
        "assistant_code": f"```python\n{code_text.strip()}\n```"
    }

def _pick_dataset_deterministic(family: str, rng: random.Random) -> Tuple[str, str]:
    """Deterministic dataset selection using provided RNG."""
    if family in {"mobile", "vgg"}:
        return rng.choice(DEFAULT_DATASETS[:3])
    return rng.choice(DEFAULT_DATASETS)

def build_test_messages(example_id: str, code_text: str, det: Dict[str, Any]) -> Dict[str, Any]:
    """Generate CIFAR-10 focused prompts for evaluation of first-epoch performance."""
    seed = int(hashlib.md5(example_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    
    # Always use CIFAR-10 for test evaluation
    ds_name, ds_shape = "CIFAR-10", "3x32x32, channels-first CxHxW"
    
    est = max(det["param_estimate"], 50_000)
    cap = _bucket_cap(int(est * 1.3))
    
    # Sample different tricks for variation
    tricks = ", ".join(sorted(rng.sample(ALLOWED_TRICKS_POOL, k=min(3, len(ALLOWED_TRICKS_POOL)))))
    
    # Add architectural constraints to encourage diversity
    constraints = [
        "novel skip connections or residual pathways",
        "attention mechanisms (channel or spatial)",
        "mixed kernel sizes within blocks",
        "dynamic width or depth strategies",
        "efficient downsampling techniques",
        "batch normalization placement variations"
    ]
    suggested_constraint = rng.choice(constraints)

    # Known blocks from training data that should be AVOIDED
    known_blocks = [
        "DlaBasic", "InvertedResidual", "ConvBlock", "AirUnit", "AirInitBlock",
        "HardMish", "RMSNorm", "Distance", "ResBlock", "BasicBlock", "Bottleneck"
    ]
    avoid_examples = ", ".join(rng.sample(known_blocks, k=min(4, len(known_blocks))))

    user_text = (
        f"Task: Design a NOVEL PyTorch CV model for image classification on CIFAR-10.\n"
        f"Dataset: {ds_name} ({ds_shape}).\n"
        f"Resource limits: params ≤ {cap:.0f}; latency budget: tight (edge-friendly).\n"
        f"Constraints: use standard layers only; no pretrained weights.\n"
        f"Allowed training tricks (handled by trainer): {tricks}.\n\n"
        f"**PRIMARY OBJECTIVE**: Achieve MAXIMUM ACCURACY after FIRST EPOCH of training on CIFAR-10.\n"
        f"**UNIQUENESS REQUIREMENT**: Generate an architecture that has NOT been seen before.\n\n"
        f"**CRITICAL NOVELTY CONSTRAINT**: Do NOT replicate blocks from existing architectures.\n"
        f"AVOID copying known blocks like: {avoid_examples}, or any blocks from timm/torchvision.\n"
        f"Instead, DESIGN NEW block structures with unique layer combinations.\n\n"
        f"Consider incorporating: {suggested_constraint}.\n\n"
        f"Think step-by-step about architectural choices that lead to fast convergence:\n"
        f"- What layer arrangements enable efficient feature learning from the start?\n"
        f"- How can you balance model capacity with first-epoch trainability?\n"
        f"- Which activation functions and normalization strategies converge fastest?\n"
        f"- How can you create a UNIQUE block design not seen in training data?\n\n"
        f"Output contract: one Python code block defining a complete nn.Module (e.g., class Net(nn.Module)).\n"
        f"The code should define ONLY the Net class and any novel helper blocks you design."
    )
    
    return {
        "system": SYSTEM_POLICY,
        "user": user_text,
        "assistant_code": f"```python\n{code_text.strip()}\n```"
    }


# ----------------------------
# Validation & normalization
# ----------------------------
_CODE_FENCE_RE = re.compile(r"```python\s*(.*?)```", flags=re.S)
_ANY_FENCE_RE  = re.compile(r"```\s*(.*?)```", flags=re.S)

def _sanitize_assistant_to_single_python_block(text: str) -> str:
    """Coerce assistant content to exactly one fenced python block."""
    import re, ast

    # 1) collect all python-fenced blocks; if none, collect any fenced blocks
    py_blocks = re.findall(r"```python\s*(.*?)```", text, flags=re.S)
    if not py_blocks:
        any_blocks = re.findall(r"```\s*(.*?)```", text, flags=re.S)
        py_blocks = any_blocks

    # nothing fenced: just wrap the whole thing
    if not py_blocks:
        code = text.strip()
        return f"```python\n{code}\n```"

    def is_parseable(code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except Exception:
            return False

    # 2) choose the longest parseable block if possible
    blocks_sorted = sorted(py_blocks, key=lambda s: len(s), reverse=True)
    for blk in blocks_sorted:
        if is_parseable(blk):
            return f"```python\n{blk.strip()}\n```"

    # 3) try concatenation of all blocks if single blocks aren't parseable
    joined = "\n\n".join(b.strip() for b in blocks_sorted)
    if is_parseable(joined):
        return f"```python\n{joined}\n```"

    # 4) fallback: longest block
    return f"```python\n{blocks_sorted[0].strip()}\n```"

def _extract_code(text: str) -> Optional[str]:
    m = _CODE_FENCE_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()

def _is_parseable_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

def _has_nn_module_subclass(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for b in node.bases:
                if getattr(b, "id", "") == "Module":     # class X(Module)
                    return True
                if getattr(b, "attr", "") == "Module":    # class X(nn.Module) / torch.nn.Module
                    return True
    return False


# ---------------------------------
# Source-grouped splitting with family stratification
# ---------------------------------
def _source_grouped_split_with_family_stratification(
    items: List[dict], seed: int = 42, ratios: Tuple[float, float, float] = (0.80, 0.10, 0.10)
) -> Dict[str, List[dict]]:
    """
    Group examples by source_path, compute family label per group (majority or first),
    then stratify groups into train/dev/test. Ensures zero overlap between splits.
    """
    from collections import Counter
    
    # Group examples by source_path
    source_groups = defaultdict(list)
    for item in items:
        meta = item.get("meta", {})
        source_path = meta.get("source_path") or f"NA::{item.get('id')}"
        source_groups[source_path].append(item)
    
    # Compute family label for each source group (majority vote, fallback to first)
    source_to_family = {}
    for source_path, group_items in source_groups.items():
        families = [item.get("meta", {}).get("family", "unknown") for item in group_items]
        # Use majority vote, fallback to first if tie
        family_counts = Counter(families)
        majority_family = family_counts.most_common(1)[0][0]
        source_to_family[source_path] = majority_family
    
    # Group source paths by family for stratification
    family_to_sources = defaultdict(list)
    for source_path, family in source_to_family.items():
        family_to_sources[family].append(source_path)
    
    # Stratify source groups by family
    rnd = random.Random(seed)
    train_sources = set()
    dev_sources = set()
    test_sources = set()
    
    for family, sources in family_to_sources.items():
        rnd.shuffle(sources)
        n_sources = len(sources)
        n_train = max(1, int(round(n_sources * ratios[0])))
        n_dev = max(1, int(round(n_sources * ratios[1])))
        
        # Ensure we don't exceed available sources
        n_train = min(n_train, n_sources - 2)  # Leave at least 2 for dev/test
        n_dev = min(n_dev, n_sources - n_train - 1)  # Leave at least 1 for test
        
        train_sources.update(sources[:n_train])
        dev_sources.update(sources[n_train:n_train + n_dev])
        test_sources.update(sources[n_train + n_dev:])
    
    # Assign examples to splits based on their source group
    train_items = []
    dev_items = []
    test_items = []
    
    for source_path, group_items in source_groups.items():
        if source_path in train_sources:
            train_items.extend(group_items)
        elif source_path in dev_sources:
            dev_items.extend(group_items)
        elif source_path in test_sources:
            test_items.extend(group_items)
        else:
            # Fallback: assign to train if not explicitly assigned
            train_items.extend(group_items)
    
    return {
        "train": train_items,
        "dev": dev_items,
        "test": test_items
    }

def _verify_no_overlap(train: List[dict], dev: List[dict], test: List[dict]) -> Dict[str, int]:
    """Verify zero overlap between splits by checking source_path intersections."""
    # Filter out None values to avoid false positives
    train_sources = {item.get("meta", {}).get("source_path") for item in train 
                     if item.get("meta", {}).get("source_path") is not None}
    dev_sources = {item.get("meta", {}).get("source_path") for item in dev 
                   if item.get("meta", {}).get("source_path") is not None}
    test_sources = {item.get("meta", {}).get("source_path") for item in test 
                    if item.get("meta", {}).get("source_path") is not None}
    
    # Count items with missing source_path
    train_missing = sum(1 for item in train if item.get("meta", {}).get("source_path") is None)
    dev_missing = sum(1 for item in dev if item.get("meta", {}).get("source_path") is None)
    test_missing = sum(1 for item in test if item.get("meta", {}).get("source_path") is None)
    
    if train_missing + dev_missing + test_missing > 0:
        print(f"Warning: Items with missing source_path - train: {train_missing}, dev: {dev_missing}, test: {test_missing}")
    
    train_dev_overlap = len(train_sources & dev_sources)
    train_test_overlap = len(train_sources & test_sources)
    dev_test_overlap = len(dev_sources & test_sources)
    
    return {
        "train_dev_overlap": train_dev_overlap,
        "train_test_overlap": train_test_overlap,
        "dev_test_overlap": dev_test_overlap
    }


# ----------------------------
# Main, reusable API
# ----------------------------
@dataclass
class ChatPrepConfig:
    """
    Reusable pipeline. Instantiate and call `run()` programmatically, or wire via CLI.
    """
    # Inputs/outputs
    accepted_dir: str = "curation_output/accepted_code"
    out_dir: str = "curation_output/chat_data"

    # Data generation
    no_infill: bool = False
    seed: int = 42

    # Post-processing / validation (SFT standards on by default)
    fix_fences: bool = True
    drop_unparseable: bool = True
    require_module_subclass: bool = True
    write_drop_report: bool = True

    # Splitting (leakage prevention)
    group_by_source: bool = True
    split_ratios: Tuple[float, float, float] = (0.80, 0.10, 0.10)

    # Optional rendering
    model_name: Optional[str] = None

    # NEW: Enable CIFAR-10 focused test prompts
    cifar10_test_focus: bool = True

    # Filenames
    train_name: str = "train.jsonl"
    dev_name: str = "dev.jsonl"
    test_name: str = "test.jsonl"

    def __post_init__(self):
        """Set default values for None parameters."""
        # Set defaults for None values
        if self.accepted_dir is None:
            self.accepted_dir = "curation_output/accepted_code"
        if self.out_dir is None:
            self.out_dir = "curation_output/chat_data"
        if self.train_name is None:
            self.train_name = "train.jsonl"
        if self.dev_name is None:
            self.dev_name = "dev.jsonl"
        if self.test_name is None:
            self.test_name = "test.jsonl"
        if self.split_ratios is None:
            self.split_ratios = (0.80, 0.10, 0.10)

    # ------------- public API -------------
    def run(self) -> Dict[str, Any]:
        random.seed(self.seed)

        in_dir = Path(self.accepted_dir)
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) curated code → examples
        print("Loading Python files...")
        pairs = load_py_files(str(in_dir))  # List[Tuple[path, code]]
        print(f"Loaded {len(pairs)} files. Building examples...")
        examples = self._build_examples(pairs, add_infill=(not self.no_infill))

        # 2) normalize & validate for SFT
        print("Normalizing examples...")
        examples = self._normalize_examples_to_dict(examples)
        print("Sanitizing and filtering examples...")
        examples, dropped = self._sanitize_and_filter(examples)

        # 3) leak-safe split with source grouping and family stratification
        if self.group_by_source:
            if _HAS_SOURCE_SPLIT:
                splits = stratified_family_split_by_source(
                    examples, ratios=self.split_ratios, seed=self.seed
                )
                train, dev, test = splits["train"], splits["dev"], splits["test"]
            else:
                splits = _source_grouped_split_with_family_stratification(
                    examples, seed=self.seed, ratios=self.split_ratios
                )
                train, dev, test = splits["train"], splits["dev"], splits["test"]
            
            # Verify zero overlap between splits
            overlap_check = _verify_no_overlap(train, dev, test)
            if any(overlap_check.values()):
                print(f"WARNING: Found overlaps in splits: {overlap_check}")
        else:
            train, dev, test = stratified_split_by_family(
                examples, ratios=self.split_ratios, seed=self.seed
            )
        
        # 3.5) Regenerate test prompts with CIFAR-10 focus if enabled
        if self.cifar10_test_focus:
            print("Regenerating test prompts with CIFAR-10 first-epoch focus...")
            test_regenerated = []
            for item in tqdm(test, desc="Regenerating test prompts", leave=False):
                meta = item.get("meta", {})
                source_path = meta.get("source_path", "")
                code = self._extract_code_from_messages(item)
                if code and source_path:
                    # Regenerate with test-focused prompts
                    test_examples = build_examples_from_code(source_path, code, add_infill=False, is_test=True)
                    test_regenerated.extend([self._ex_to_dict(ex) for ex in test_examples])
            
            if test_regenerated:
                # Replace test split with regenerated CIFAR-10 focused prompts
                test = test_regenerated
                print(f"Regenerated {len(test)} test examples with CIFAR-10 focus")

        # 4) write jsonl
        print("Writing JSONL files...")
        train_path = out_dir / self.train_name
        dev_path   = out_dir / self.dev_name
        test_path  = out_dir / self.test_name
        write_jsonl(str(train_path), train)
        write_jsonl(str(dev_path), dev)
        write_jsonl(str(test_path), test)
        
        # 4.5) generate fresh manifest after writing splits and verify no leakage
        manifest_path = write_manifest(str(out_dir))
        
        # Assert zero overlap - fail fast if leakage detected
        overlap_check = _verify_no_overlap(train, dev, test)
        if any(overlap_check.values()):
            raise RuntimeError(f"LEAKAGE DETECTED! Overlaps found: {overlap_check}. "
                             f"Manifest saved to {manifest_path} for investigation.")

        # 5) optional: render with model chat template
        rendered_paths = {}
        if self.model_name:
            print(f"Rendering with {self.model_name} template...")
            rend_dir = out_dir / "rendered"
            rend_dir.mkdir(exist_ok=True)
            for split_name, split_data in tqdm([("train", train), ("dev", dev), ("test", test)], 
                                               desc="Rendering templates", leave=False):
                rendered = render_with_template(split_data, self.model_name)
                rp = rend_dir / f"{split_name}.jsonl"
                with open(rp, "w", encoding="utf-8") as f:
                    for item in rendered:
                        f.write(self._json_dumps(item) + "\n")
                rendered_paths[split_name] = str(rp)

        # 6) drop report
        if self.write_drop_report and dropped:
            with open(out_dir / "_drop_report.jsonl", "w", encoding="utf-8") as f:
                for r in dropped:
                    f.write(self._json_dumps(r) + "\n")

        # 7) quick sanity: unique sources per split
        def _uniq_sources(rows: List[dict]) -> int:
            return len({r.get("meta", {}).get("source_path") for r in rows if r.get("meta", {}).get("source_path")})

        # Prepare return data
        result = {
            "counts": {"train": len(train), "dev": len(dev), "test": len(test),
                       "total": len(train) + len(dev) + len(test)},
            "paths": {"train": str(train_path), "dev": str(dev_path), "test": str(test_path),
                      "rendered": rendered_paths or None, "manifest": manifest_path},
            "dropped": len(dropped),
            "unique_sources": {"train": _uniq_sources(train), "dev": _uniq_sources(dev), "test": _uniq_sources(test)},
            "config": {
                "fix_fences": self.fix_fences,
                "drop_unparseable": self.drop_unparseable,
                "require_module_subclass": self.require_module_subclass,
                "group_by_source": self.group_by_source,
            }
        }
        
        # Add overlap verification results if source grouping was used
        if self.group_by_source:
            overlap_check = _verify_no_overlap(train, dev, test)
            result["overlap_verification"] = overlap_check
        
        return result

    # ------------- internals -------------
    def _build_examples(self, file_pairs: List[Tuple[str, str]], add_infill: bool, is_test: bool = False) -> List[Any]:
        exs: List[Any] = []
        for path, code in tqdm(file_pairs, desc="Building examples", leave=False):
            exs.extend(build_examples_from_code(path, code, add_infill=add_infill, is_test=is_test))
        return exs
    
    def _extract_code_from_messages(self, item: Dict[str, Any]) -> str:
        """Extract code from assistant message."""
        messages = item.get("messages", [])
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Extract code from markdown fence
                match = re.search(r"```python\s*(.*?)```", content, re.DOTALL)
                if match:
                    return match.group(1).strip()
        return ""

    @staticmethod
    def _ex_to_dict(ex: Any) -> Dict[str, Any]:
        # Accept dict, dataclass (ChatExample), or objects with to_dict().
        if isinstance(ex, dict):
            return ex
        if hasattr(ex, "to_dict") and callable(getattr(ex, "to_dict")):
            return ex.to_dict()
        if is_dataclass(ex):
            return asdict(ex)
        # Last resort: try attribute projection for known fields
        try:
            return {
                "id": getattr(ex, "id", None),
                "messages": getattr(ex, "messages", None),
                "meta": getattr(ex, "meta", None),
            }
        except Exception:
            raise TypeError(f"Unsupported example type: {type(ex)!r}")

    def _normalize_examples_to_dict(self, examples: List[Any]) -> List[Dict[str, Any]]:
        return [self._ex_to_dict(ex) for ex in tqdm(examples, desc="Normalizing", leave=False)]

    def _sanitize_and_filter(self, examples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        kept: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []

        for ex in tqdm(examples, desc="Sanitizing & filtering", leave=False):
            msgs = ex.get("messages", [])
            # Require canonical 3-turn structure (system/user/assistant)
            if not (isinstance(msgs, list) and len(msgs) == 3):
                ex["_drop_reason"] = "bad_schema_len"
                dropped.append(ex); continue
            roles = [m.get("role") for m in msgs]
            if roles != ["system", "user", "assistant"]:
                ex["_drop_reason"] = "bad_roles"
                dropped.append(ex); continue

            # Enforce single fenced python block in assistant
            if self.fix_fences:
                msgs[-1]["content"] = _sanitize_assistant_to_single_python_block(msgs[-1]["content"])

            code = _extract_code(msgs[-1].get("content", "") or "")
            if code is None:
                ex["_drop_reason"] = "no_code_fence"
                dropped.append(ex); continue

            if self.drop_unparseable and not _is_parseable_python(code):
                ex["_drop_reason"] = "parse_fail"
                dropped.append(ex); continue

            if self.require_module_subclass and not _has_nn_module_subclass(code):
                ex["_drop_reason"] = "no_nn_module"
                dropped.append(ex); continue

            ex["messages"] = msgs
            kept.append(ex)

        return kept, dropped

    def _generate_manifest(self, train: List[dict], dev: List[dict], test: List[dict]) -> Dict[str, Any]:
        """Generate a manifest with source paths, families, and overlap verification."""
        from collections import defaultdict, Counter
        
        def _extract_sources_and_families(split_data: List[dict], split_name: str) -> Dict[str, Any]:
            sources = set()
            families = []
            family_counts = Counter()
            
            for item in split_data:
                meta = item.get("meta", {})
                source_path = meta.get("source_path")
                family = meta.get("family", "unknown")
                
                if source_path:
                    sources.add(source_path)
                families.append(family)
                family_counts[family] += 1
            
            return {
                "split": split_name,
                "count": len(split_data),
                "unique_sources": len(sources),
                "source_paths": sorted(list(sources)),
                "family_distribution": dict(family_counts.most_common()),
                "families": families
            }
        
        # Generate split manifests
        train_manifest = _extract_sources_and_families(train, "train")
        dev_manifest = _extract_sources_and_families(dev, "dev")
        test_manifest = _extract_sources_and_families(test, "test")
        
        # Compute overlaps
        train_sources = set(train_manifest["source_paths"])
        dev_sources = set(dev_manifest["source_paths"])
        test_sources = set(test_manifest["source_paths"])
        
        overlaps = {
            "train_dev": sorted(list(train_sources & dev_sources)),
            "train_test": sorted(list(train_sources & test_sources)),
            "dev_test": sorted(list(dev_sources & test_sources))
        }
        
        # Overall statistics
        total_sources = len(train_sources | dev_sources | test_sources)
        total_examples = len(train) + len(dev) + len(test)
        
        return {
            "metadata": {
                "generated_at": str(Path().cwd()),
                "config": {
                    "group_by_source": self.group_by_source,
                    "split_ratios": self.split_ratios,
                    "seed": self.seed
                }
            },
            "summary": {
                "total_examples": total_examples,
                "total_unique_sources": total_sources,
                "overlap_detected": any(overlaps.values())
            },
            "splits": {
                "train": train_manifest,
                "dev": dev_manifest,
                "test": test_manifest
            },
            "overlaps": overlaps,
            "verification": {
                "train_dev_overlap_count": len(overlaps["train_dev"]),
                "train_test_overlap_count": len(overlaps["train_test"]),
                "dev_test_overlap_count": len(overlaps["dev_test"]),
                "total_overlaps": sum(len(overlap) for overlap in overlaps.values())
            }
        }

    @staticmethod
    def _json_dumps(o: Any) -> str:
        import json
        return json.dumps(o, ensure_ascii=False)
