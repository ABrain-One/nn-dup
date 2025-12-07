# cli/chatprep_cli.py
import argparse
import sys
from pathlib import Path

# Handle both module and direct execution
try:
    from ..prompt_builder import ChatPrepConfig
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from ab.chatprep.prompt_builder import ChatPrepConfig

def main():
    ap = argparse.ArgumentParser("LEMUR ChatPrep – arguments → ChatPrepConfig")
    ap.add_argument("--accepted-dir", type=str, default="curation_output/accepted_code")
    ap.add_argument("--out", type=str, default="curation_output/chat_data")
    ap.add_argument("--no-infill", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fix-fences", action="store_true")
    ap.add_argument("--drop-unparseable", action="store_true")
    ap.add_argument("--no-group-by-source", action="store_true", help="Disable source-level grouping (default: enabled)")
    ap.add_argument("--model", type=str, default=None)

    args = ap.parse_args()

    cfg = ChatPrepConfig(
        accepted_dir=args.accepted_dir,
        out_dir=args.out,
        no_infill=args.no_infill,
        seed=args.seed,
        fix_fences=args.fix_fences,
        drop_unparseable=args.drop_unparseable,
        group_by_source=not args.no_group_by_source,
        model_name=args.model,
    )

    result = cfg.run()
    print("Chat data written:", result["paths"])
    print("Counts:", result["counts"])

if __name__ == "__main__":
    main()
