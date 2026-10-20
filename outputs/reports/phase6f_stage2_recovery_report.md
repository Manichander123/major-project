# Phase 6F Stage 2: Failure Recovery & Schema Validation Report

**Execution Date:** 2026-10-18T08:58:32.490731+00:00  
**Target Farms Attempted:** 5  
**Fully Recovered:** 5  
**Partially Recovered:** 0  
**Still Failed:** 0  

---

## 1. Recovered Farm Summary

| FarmID | Crop | Sub-District | Final Status | Note |
| :---: | :--- | :--- | :---: | :--- |
| 1330053 | Bengal Gram | Sonala | **`success`** |  |
| 1330055 | Bengal Gram | Sonala | **`success`** |  |
| 1330056 | Bengal Gram | Boath | **`success`** |  |
| 1330059 | Bengal Gram | Boath | **`success`** |  |
| 1330230 | Bengal Gram | Jainad | **`success`** |  |

## 2. Operation Ledger

| FarmID | Window | Requested | Attempts | Outcome | Valid Scenes |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1330053 | early | False | 0 | `preserved_existing` | 10.0 |
| 1330053 | mid | False | 0 | `preserved_existing` | 55.0 |
| 1330053 | late | False | 0 | `preserved_existing` | 34.0 |
| 1330053 | full | True | 1 | `success` | 74.0 |
| 1330055 | early | False | 0 | `preserved_existing` | 8.0 |
| 1330055 | mid | True | 1 | `success` | 48.0 |
| 1330055 | late | True | 1 | `success` | 43.0 |
| 1330055 | full | True | 1 | `success` | 97.0 |
| 1330056 | early | False | 0 | `preserved_existing` | 23.0 |
| 1330056 | mid | True | 1 | `success` | 41.0 |
| 1330056 | late | True | 1 | `success` | 26.0 |
| 1330056 | full | True | 1 | `success` | 86.0 |
| 1330059 | early | True | 1 | `success` | 14.0 |
| 1330059 | mid | True | 1 | `success` | 26.0 |
| 1330059 | late | True | 1 | `success` | 15.0 |
| 1330059 | full | True | 1 | `success` | 55.0 |
| 1330230 | early | True | 1 | `success` | 33.0 |
| 1330230 | mid | True | 1 | `success` | 35.0 |
| 1330230 | late | True | 1 | `success` | 46.0 |
| 1330230 | full | True | 1 | `success` | 114.0 |

