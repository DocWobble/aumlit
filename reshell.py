"""Command-line entry point for Reshell probing.

This tiny module parses CLI arguments and hands off to the probing
pipeline.  The actual probing logic is not implemented yet; instead we
emit placeholder paths for the contact trace and obligations files.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple


def run_pipeline(artifact: Path, out_dir: Path) -> Tuple[Path, Path]:
    """Stub probing pipeline.

    In the full implementation this function will run the probing
    workflow and return the paths to the generated ``contact_trace``
    and ``obligations`` artifacts.  For now it simply constructs the
    paths and leaves a placeholder for future work.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    contact_trace = out_dir / "contact_trace.json"
    obligations = out_dir / "obligations.json"
    return contact_trace, obligations


def main(artifact: str, out_dir: str = "out") -> None:
    """Parse arguments and delegate to the probing pipeline.

    Parameters
    ----------
    artifact:
        Path to the model artifact to analyse.
    out_dir:
        Directory where probe results should be written.  The directory
        is created if it does not exist.
    """
    contact_trace, obligations = run_pipeline(Path(artifact), Path(out_dir))
    print(f"contact trace written to: {contact_trace}")
    print(f"obligations written to: {obligations}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe a model artifact")
    parser.add_argument("--artifact", required=True, help="path to model artifact")
    parser.add_argument("--out", default="out", help="output directory for results")
    args = parser.parse_args()

    main(args.artifact, args.out)
