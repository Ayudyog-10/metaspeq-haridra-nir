"""Train the Haridra starch screening model and write the artefact.

    python src/train.py --components 8

Writes models/starch_pls.joblib and models/model_card.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import joblib

from chemometrics import (
    DEFAULT_BAND, DEFAULT_N_COMPONENTS, component_scan, interpret_rpd,
    load_starch_dataset, band_mask, preprocess, train_starch_model,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", type=int, default=DEFAULT_N_COMPONENTS)
    ap.add_argument("--preprocessing", default="snv")
    ap.add_argument("--scan", action="store_true",
                    help="also write the RMSECV-vs-components table")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out = root / "models"
    out.mkdir(exist_ok=True)

    print(f"Training PLS with {args.components} components, "
          f"preprocessing={args.preprocessing} ...")
    model = train_starch_model(args.components, args.preprocessing, DEFAULT_BAND)

    cv = model.cv_metrics
    print("\nLeave-one-sample-out validation")
    print(f"  samples        {cv['n_samples']}   spectra {cv['n']}")
    print(f"  R2             {cv['r2']:.4f}")
    print(f"  RMSECV         {cv['rmse']:.3f} % w/w")
    print(f"  bias           {cv['bias']:+.3f} % w/w")
    print(f"  slope          {cv['slope']:.4f}")
    print(f"  RPD            {cv['rpd']:.2f}  -> {interpret_rpd(cv['rpd'])}")

    joblib.dump(model, out / "starch_pls.joblib")

    card = {
        "name": "Haridra dry extract — starch screening (PLS)",
        "trained": date.today().isoformat(),
        "analyte": "starch, % w/w",
        "matrix": "Haridra (Curcuma longa) dry extract, binary blend",
        "instrument": "METASPEQ MQNIR-DR26A, diffuse reflectance",
        "spectral_band_nm": list(DEFAULT_BAND),
        "preprocessing": args.preprocessing,
        "n_components": args.components,
        "calibration_range_pct": [cv["ref_min"], cv["ref_max"]],
        "n_samples": cv["n_samples"],
        "n_spectra": cv["n"],
        "validation": "leave-one-sample-out (all 16 replicate scans withheld together)",
        "cv_metrics": cv,
        "calibration_metrics": model.calibration_metrics,
        "rpd_interpretation": interpret_rpd(cv["rpd"]),
        "intended_use": "Goods-in screening for gross starch adulteration. "
                        "Not validated for release testing or for reporting "
                        "against a specification limit.",
        "known_limitations": [
            "Reference values are gravimetric, not anchored to an independent assay.",
            "Single extract batch (DCL/2404001/24-25) from one supplier.",
            "Binary blend with soluble starch; not a commercial formulation.",
            "Sample moisture was neither measured nor controlled.",
            "Accuracy, repeatability and robustness were not formally assessed.",
        ],
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2))
    print(f"\nWrote {out/'starch_pls.joblib'} and {out/'model_card.json'}")

    if args.scan:
        wl, X, y, groups = load_starch_dataset()
        Xp = preprocess(X[:, band_mask(wl, DEFAULT_BAND)], args.preprocessing)
        table = component_scan(Xp, y, groups, 14)
        table.to_csv(out / "component_scan.csv", index=False)
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
