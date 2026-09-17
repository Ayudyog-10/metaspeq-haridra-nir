# METASPEQ Haridra NIR

Near-infrared chemometric models for **Haridra (*Curcuma longa*) dry extract**, with an
interactive Streamlit application for exploring the spectra, running predictions and
inspecting model diagnostics.

Developed by **Ayudyog Private Limited (METASPEQ)** in association with
**Mendine Pharmaceuticals Pvt. Ltd.**, on a METASPEQ MQNIR-DR26A desktop diffuse
reflectance spectrometer.
https://metaspeq-haridra-nir-gugaxbmecwogcbcmxjgvfz.streamlit.app/
---

## What's here

| Method | Analyte | Range | Cross-validated performance |
|---|---|---|---|
| Starch adulteration screening | Starch, % w/w | 0.99 – 30.07 % | R² 0.956, RMSECV 1.79 % w/w, RPD 4.78 |
| Curcuminoid assay | Curcuminoid, % w/w | 0.5 – 15 % | R² 0.964, SEP 0.84 % w/w, RPD 5.25 |

All starch figures come from **leave-one-sample-out** cross-validation. The 16
replicate scans of each physical sample are withheld together, so the model is
always scored on a sample it has never encountered. See
[Validation methodology](#validation-methodology) for why this matters.

---

## Quick start

```bash
git clone https://github.com/Ayudyog-10/metaspeq-haridra-nir.git
cd metaspeq-haridra-nir

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`.

To rebuild the model artefact from the raw spectra:

```bash
python src/train.py --components 8 --scan
```

---

## The application

Five tabs:

- **Predict** — select any measured spectrum from the study, or upload your own
  256-point absorbance CSV, and get a starch prediction with a green/amber/red
  confidence zone based on proximity to the calibration range. Ready-made test
  files are in `examples/`.
- **Spectra** — overlay spectra by concentration level, before and after
  preprocessing, to see what SNV actually removes.
- **Model Diagnostics** — predicted-vs-reference, the RMSECV-versus-complexity
  curve used to select the number of latent variables, and a residual plot.
- **Curcuminoid Assay** — linearity, error-by-level, and the working-range
  restriction that follows from the absolute error floor.
- **Model Card** — full provenance, intended use and limitations.

The sidebar lets you change the number of PLS latent variables and the
preprocessing, and everything recomputes. Dropping below 6 components is
instructive: the model visibly underfits.

---

## Repository layout

```
├── streamlit_app.py              # the application
├── src/
│   ├── chemometrics.py           # preprocessing, PLS, sample-wise validation
│   └── train.py                  # builds models/starch_pls.joblib
├── models/
│   ├── starch_pls.joblib         # trained 8-component model
│   ├── model_card.json           # provenance, metrics, limitations
│   └── component_scan.csv        # RMSECV vs number of latent variables
├── data/
│   ├── haridra_starch_absorbance.csv   # 688 spectra, 43 samples, 256 points
│   ├── wavelengths_256.csv             # 892–1710 nm grid
│   └── curcuminoid_linearity.csv       # 30-level assay data
└── docs/
    └── VALIDATION.md             # methodology notes
```

---

## The data

688 absorbance spectra from 43 samples. Each sample is a fixed 10 g of Haridra
dry extract with soluble starch added in 0.1 g increments, giving 0.99 % to
30.07 % w/w starch. Sixteen scans per sample.

| Setting | Value |
|---|---|
| Instrument | METASPEQ MQNIR-DR26A, diffuse reflectance |
| Spectral range | 892 – 1710 nm, 256 detector points |
| Model band | 1074 – 1574 nm (156 points) |
| Exposure | 25,000 µs |
| Scans averaged | 4 |
| Working A/D count | 50,000 – 52,000 |
| Cuvette | Circular quartz, 32 mm visible diameter |
| White reference | Barium sulphate |
| Extract batch | DCL/2404001/24-25, Konark Herbals and Health Care Pvt. Ltd. |
| Starch | Merck Emparta ACS soluble starch |

---

## Validation methodology

Replicate scans of the same packed cuvette are near-duplicates. If a train/test
split is made across individual *spectra*, replicates of the same physical sample
land on both sides, and the model is scored on data it has effectively already
seen. The result measures instrument repeatability, not predictive ability, and
it is optimistic by a wide margin.

Every routine in `src/chemometrics.py` therefore groups by `sample_id` and uses
`GroupKFold`, so all 16 scans of a sample move together. This is the honest
analogue of routine use, where the instrument meets a sample for the first time.

The difference is not academic. On this dataset, the same 4-component model gives
R² 0.94 under a spectrum-level split and R² 0.85 under a sample-level one.

### Choosing the number of latent variables

`models/component_scan.csv` holds honest RMSECV against model complexity:

| Latent variables | R² | RMSECV (% w/w) | RPD |
|---|---|---|---|
| 4 | 0.853 | 3.27 | 2.61 |
| 6 | 0.923 | 2.37 | 3.61 |
| **8** | **0.956** | **1.79** | **4.78** |
| 10 | 0.955 | 1.82 | 4.69 |
| 12 | 0.953 | 1.84 | 4.63 |

Error falls steeply to 8 and then flattens. Eight is the elbow, and the default.

### Interpreting RPD

| RPD | Interpretation |
|---|---|
| < 2.0 | Not usable for quantification |
| 2.0 – 3.0 | Rough screening only |
| 3.0 – 5.0 | Adequate for QC screening and in-process control |
| 5.0 – 8.0 | Adequate for QC and, with validation, release quantification |
| > 8.0 | Suitable for any application |

---

## Scope and limitations

These are **method-development and feasibility studies**, not GMP analytical
method validations. They do not qualify either method for batch release.

- Reference values are **gravimetric**, not anchored to an independent validated
  assay such as HPLC.
- Both use **binary model blends**, not commercial formulations.
- All work used a **single extract batch from one supplier**. Botanical raw
  material varies substantially between batches, cultivars and seasons.
- **Sample moisture was not controlled**, and the strongest band in these spectra
  (≈1450 nm) belongs to water.
- Accuracy, repeatability, intermediate precision and robustness were **not
  formally assessed**.
- The curcuminoid method has an absolute error floor of about ±0.84 % w/w, so
  quantitative reporting should be restricted to roughly **3–15 % w/w**.

Anchoring both calibrations to HPLC values on real product, across several
suppliers and batches, is the single largest available improvement.

---

## Contact

**Ayudyog Private Limited** · info@metaspeq.com · +91 96742 37301
15 Nandalal Mitra Lane, Tollygunge Regent Park, Kolkata, West Bengal 700040, India

---

## Licence

Code is released under the MIT Licence (see `LICENSE`).

Spectral data and model artefacts are © 2026 Ayudyog Private Limited, released for
evaluation and research use. Please contact Ayudyog before any commercial use or
redistribution.
