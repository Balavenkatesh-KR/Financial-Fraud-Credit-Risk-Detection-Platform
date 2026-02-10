from __future__ import annotations

import argparse

import pandas as pd

from src.drift_detection.psi import population_stability_index


def run(reference_path: str, production_path: str, column: str) -> None:
    ref = pd.read_csv(reference_path)[column]
    prod = pd.read_csv(production_path)[column]
    psi = population_stability_index(ref, prod)
    print(f"PSI({column}) = {psi:.4f}")
    if psi > 0.25:
        print("ALERT: Significant drift detected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--production", required=True)
    parser.add_argument("--column", required=True)
    args = parser.parse_args()
    run(args.reference, args.production, args.column)
