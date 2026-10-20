"""Reconcile Train.csv source-label counts with Phase 6D report, manifest, features, and OOF predictions."""
import pandas as pd, json

train = pd.read_csv("datasss/Train.csv")
manifest = pd.read_csv("outputs/data/s2_phase6c_manifest_350.csv")
features = pd.read_csv("outputs/data/s2_phase6c_350_features.csv")
oof = pd.read_csv("outputs/data/phase6c_oof_predictions.csv")

print("=" * 80)
print("TASK 1: Source-Label Reconciliation")
print("=" * 80)

# 1. Train.csv actual counts
print("\n1. Train.csv actual Crop counts (datasss/Train.csv, 7888 rows):")
actual = train["Crop"].value_counts().sort_index()
for crop, count in actual.items():
    print(f"   {crop:<15} {count:>5}")

# 2. Phase 6D report stated counts (from memory/prior summary)
reported = {"Bengal Gram": 244, "Chillies": 948, "Cotton": 1624, "Groundnut": 201,
            "Maize": 1475, "Paddy": 2130, "Red Gram": 1266}
print("\n2. Phase 6D Report 'Available in Train.csv' (PRIOR CLAIM):")
for crop in sorted(reported):
    act = int(actual.get(crop, 0))
    rep = reported[crop]
    flag = " <<< WRONG" if act != rep else ""
    print(f"   {crop:<15} Actual={act:>5}  Reported={rep:>5}{flag}")

# 3. Manifest alignment
print("\n3. Manifest (s2_phase6c_manifest_350.csv):")
print(f"   Total rows: {len(manifest)}")
print(f"   Unique FarmIDs: {manifest['FarmID'].nunique()}")
print(f"   Crop counts:\n{manifest['Crop'].value_counts().sort_index().to_string()}")

# 4. Features alignment
print("\n4. Features CSV (s2_phase6c_350_features.csv):")
print(f"   Total rows: {len(features)}")
clean = features[features["status"] == "success"]
print(f"   Successful: {len(clean)}")
print(f"   Failed: {len(features) - len(clean)}")
print(f"   Clean Crop counts:\n{clean['Crop'].value_counts().sort_index().to_string()}")

# 5. OOF alignment
print("\n5. OOF predictions (phase6c_oof_predictions.csv):")
print(f"   Total rows: {len(oof)}")
print(f"   Crop counts:\n{oof['Crop'].value_counts().sort_index().to_string()}")

# 6. Cross-check: all FarmIDs in OOF must be in features (successful)
oof_ids = set(oof["FarmID"])
clean_ids = set(clean["FarmID"])
print(f"\n6. FarmID alignment: OOF IDs == Clean Feature IDs? {oof_ids == clean_ids}")
print(f"   In OOF but not features: {oof_ids - clean_ids}")
print(f"   In features but not OOF: {clean_ids - oof_ids}")

# 7. Cross-check: all manifest FarmIDs exist in Train.csv
manifest_ids = set(manifest["FarmID"])
train_ids = set(train["FarmID"])
missing = manifest_ids - train_ids
print(f"\n7. Manifest FarmIDs not in Train.csv: {len(missing)}")
if missing:
    print(f"   Missing IDs: {missing}")

# 8. Crop label consistency: same label in manifest/features/oof for each FarmID
feat_labels = features.set_index("FarmID")["Crop"].to_dict()
man_labels = manifest.set_index("FarmID")["Crop"].to_dict()
oof_labels = oof.set_index("FarmID")["Crop"].to_dict()
mismatches = []
for fid in oof_ids:
    m = man_labels.get(fid)
    f = feat_labels.get(fid)
    o = oof_labels.get(fid)
    if not (m == f == o):
        mismatches.append((fid, m, f, o))
print(f"\n8. Crop label mismatches across manifest/features/oof: {len(mismatches)}")
if mismatches:
    for mm in mismatches[:5]:
        print(f"   FarmID {mm[0]}: manifest={mm[1]}, features={mm[2]}, oof={mm[3]}")
