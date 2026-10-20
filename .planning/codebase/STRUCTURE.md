# Codebase Structure

```
major_proj_final/
├── config/
│   └── project_config.yaml         # Centralized configuration (crops, bands, CRS, ML hyperparameters)
├── datasss/                        # Raw survey data (READ ONLY)
│   ├── Train.csv                   # 7,888 ground truth survey records
│   ├── Test.csv                    # Evaluation set
│   └── Starter_Notebook.ipynb      # Reference notebook & CRS specifications
├── outputs/
│   ├── data/                       # Manifests, raw spectral chunks, feature files, OOF predictions
│   │   ├── s2_full_expansion_manifest_7164.csv
│   │   ├── s2_phase6e_chunk_*.csv (chunks 00 to 14)
│   │   ├── s2_phase6e_7164_raw_bands.csv
│   │   ├── phase6c_oof_predictions.csv
│   │   └── phase6d_oof_predictions.csv
│   ├── reports/                    # Audit and benchmark reports (JSON & Markdown)
│   ├── figures/                    # Spatial and error visualization plots
│   └── models/                     # Serialized model pipelines (.joblib)
├── scripts/
│   ├── extract_s2_full_campaign_phase6e.py # Phase 6E chunk extraction runner
│   ├── run_phase6d_pilot.py                # Pilot comparison runner
│   ├── audit_mgrs_mosaic.py                # MGRS duplicate overlap auditor
│   ├── reconcile_source_labels.py          # Ground truth label reconciliation
│   └── visualize_spatial.py                # Geospatial visualization tool
├── src/
│   ├── data/
│   │   ├── load_data.py                    # Data ingestion & WKT parsing
│   │   └── validate_data.py                # Quality assertions & anomaly checks
│   ├── features/
│   │   └── indices.py                      # Spectral & vegetation index algorithms
│   ├── geospatial/
│   │   ├── geometry_utils.py               # Polygon validation, metric projection, area calculation
│   │   └── spatial_viz.py                  # Spatial plotting utilities
│   ├── models/
│   │   ├── baseline_rf.py                  # Simple Random Forest baseline
│   │   └── train_pilot_models.py           # Multi-track benchmark runner with spatial CV
│   └── satellite/
│       └── s2_extractor.py                 # Earth Engine extraction engine
└── tests/
    ├── test_baseline.py                    # Tabular baseline tests
    ├── test_phase6c.py                     # Phase 6C regression tests
    ├── test_phase6d_audit.py               # Phase 6D OOF & reproducibility assertions
    ├── test_pre_extraction_gate.py         # MGRS overlap, cloud masking & NaN assertions
    └── test_s2_mosaic.py                   # Daily mosaic correctness tests
```
