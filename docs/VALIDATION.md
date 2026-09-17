# Validation methodology

## Why the split has to be at sample level

Sixteen scans were acquired from each of the 43 physical samples without
repacking the cuvette between scans. Those 16 spectra are near-duplicates:
they differ only by detector noise and short-term instrument drift.

A random train/test split across the 688 individual spectra therefore places
replicates of the same packed cuvette on both sides. The model is then scored
on spectra it has effectively already seen. What that measures is the
repeatability of the instrument on a fixed packing, not the ability of the
model to predict a sample it has never met.

On this dataset the gap is large:

| Split | R² | RMSE (% w/w) |
|---|---|---|
| Spectrum-level (all replicates mixed) | 0.94 | 2.10 |
| Sample-level, leave-one-sample-out | 0.85 | 3.27 |

Both rows use the same 4-component model and the same data. Only the
validation design differs.

`src/chemometrics.py` uses `GroupKFold` grouped on `sample_id` throughout, so
this mistake cannot be made accidentally.

## Metrics reported

- **R²** — coefficient of determination. Proportion of variance in the
  reference values explained by the model. Not the correlation coefficient.
- **RMSECV** — root mean square error of cross-validation, in % w/w. The
  headline error figure.
- **Bias** — mean of (predicted − reference). A systematic offset.
- **SEP** — standard deviation of the residuals after removing bias.
- **Slope** — gradient of predicted on reference. Below 1 indicates range
  compression: the model under-reads at the top and over-reads at the bottom.
- **RPD** — standard deviation of the reference set divided by SEP. How much
  better the model is than simply guessing the mean.

## Choosing model complexity

Run:

```bash
python src/train.py --components 8 --scan
```

This writes `models/component_scan.csv`. Choose the elbow of the RMSECV curve,
not the global minimum. Adding latent variables for a fraction of a percent of
improvement fits noise and degrades field performance.

A large gap between calibration error (RMSEC) and cross-validated error
(RMSECV) is the signature of overfitting. The reverse pattern — RMSECV falling
steeply as components are added — indicates underfitting.

## Preprocessing must match

Whatever preprocessing is used to build a model must be applied identically at
prediction time. Applying SNV to a model calibrated on Savitzky-Golay data
produces confident nonsense with no error message. `StarchModel.predict()`
stores and reapplies the correct pipeline so this cannot drift.

## What has not been validated

- Accuracy as percent recovery
- Repeatability (within-day replicates)
- Intermediate precision (day-to-day)
- Robustness against operator, packing, temperature, humidity, warm-up
- Instrument-to-instrument transfer for the starch model
- Performance on commercial formulations rather than binary blends
- Performance across extract batches, suppliers, cultivars and seasons

These are required before any regulated use.
