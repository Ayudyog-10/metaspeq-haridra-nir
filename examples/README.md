# Example spectra

Real measured spectra from the Mendine study, extracted for testing the
**Predict** tab of the Streamlit app and for checking any integration you build
against the model.

## Single-spectrum files

`spectrum_<NN.NN>pct_starch.csv` — one absorbance value per line, 256 lines, no
header. This is the format the app's uploader expects. Drop any of these into
the Predict tab and it will return a starch estimate.

| File | Reference starch | Model prediction | Residual |
|---|---|---|---|
| `spectrum_00.99pct_starch.csv` | 0.99 % | −0.11 % | −1.10 |
| `spectrum_04.76pct_starch.csv` | 4.76 % | 6.41 % | +1.64 |
| `spectrum_09.09pct_starch.csv` | 9.09 % | 7.75 % | −1.34 |
| `spectrum_15.25pct_starch.csv` | 15.25 % | 18.41 % | +3.16 |
| `spectrum_20.00pct_starch.csv` | 20.00 % | 21.16 % | +1.16 |
| `spectrum_24.81pct_starch.csv` | 24.81 % | 24.74 % | −0.07 |
| `spectrum_30.07pct_starch.csv` | 30.07 % | 29.10 % | −0.97 |

`MANIFEST.csv` holds the same table in machine-readable form, with the source
`sample_id` for each file.

### Read the residuals correctly

These are **single scans**, not the 16-scan average a routine measurement would
use. Scan-to-scan spread is therefore fully visible in the residual column, and
individual values run wider than the model's RMSECV of 1.79 % w/w — the 15.25 %
file is +3.16, well outside it.

That is expected and is the point of including them. Averaging 4 scans (the
configured acquisition setting) removes most of this spread. If you want to see
the averaged behaviour, select a sample in the app's Predict tab and step
through its replicates, or use the batch file below.

The 0.99 % file predicting slightly negative is also expected: it sits at the
very bottom of the calibration range, where the model's absolute error is
comparable to the analyte level. A negative prediction should be read as "at or
near zero", and the app flags it amber for exactly this reason.

## Batch file

`batch_7_spectra.csv` — the same seven spectra as rows, with a header row of
wavelengths in nm and a `label` column. Use this for scripted prediction:

```python
import pandas as pd, joblib
df = pd.read_csv("examples/batch_7_spectra.csv")
model = joblib.load("models/starch_pls.joblib")
df["predicted_starch_pct"] = model.predict(df.iloc[:, 1:].to_numpy())
print(df[["label", "predicted_starch_pct"]])
```

## Template

`TEMPLATE_blank_spectrum.csv` — the 256-point METASPEQ wavelength grid
(892.434 – 1709.864 nm) with an empty absorbance column. Paste your own measured
absorbance values into the second column, then save the absorbance column alone
as a headerless single-column CSV to upload.

Your spectrum must be on this exact grid. The model was built on these 256
detector positions and cannot interpolate from a different one.

## Provenance

All spectra were acquired on the METASPEQ MQNIR-DR26A at Mendine
Pharmaceuticals in August 2025: diffuse reflectance, circular quartz cuvette,
25,000 µs exposure, barium sulphate white reference, A/D count 50,000–52,000.
Material was Haridra dry extract batch DCL/2404001/24-25 blended with Merck
Emparta soluble starch.
