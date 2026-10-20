# Known Technical Concerns & Research Bottlenecks

## 1. Spatial Generalization & Mandal Heterogeneity
- `Sub-District` spatial grouping does not completely eliminate spatial autocorrelation because adjacent mandals can share microclimates or agro-ecological boundaries.
- Severe class imbalances in certain mandals: in Phase 6D, Folds 3 and 4 exhibited a steep performance collapse on Maize (F1=0.35-0.38 vs. 0.55-0.71 in other folds) due to regional phenological variation.

## 2. Cloud Obscuration in Kharif Season
- Kharif crops (monsoon season: June - October) suffer from prolonged persistent cloud cover, resulting in higher missingness in early and mid phenological windows compared to Rabi crops.
- Preserving NaNs is crucial to avoid artificial zero distortion, but requires robust imputation strategies that do not leak spatial data across folds.

## 3. Earth Engine Quotas & Transient Timeouts
- High-concurrency requests or large geometry operations on GEE occasionally face computation timeouts (error 429 or deadline exceeded) and transient socket drops.
- Extraction campaigns must use bounded concurrency and exponential backoff retry.

## 4. Retrospective vs. Operational Prediction Distinction
- Models trained on late-season or full-season spectral composite windows cannot be deployed for early in-season management decisions.
- A clear distinction between Track A (retrospective) and Track B (in-season) is scientifically mandatory.
