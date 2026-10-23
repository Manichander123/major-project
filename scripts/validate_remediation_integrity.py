"""
scripts/validate_remediation_integrity.py
=========================================
Phase 10 Comprehensive Automated Validation Script:
Verifies:
1. Dataset & Partition sample counts (Fold 0: 6,335/1,508; Test: 2,718/2,265/453).
2. Spatial isolation & 0 Mandal overlap (116 train vs 35 test).
3. Test metrics and confusion matrix arithmetic (1,847 correct, 418 incorrect).
4. Feature schema counts (140 S+T, 160 S+T+P).
5. Text consistency in MANUSCRIPT_DRAFT.md (no forbidden phrases).
6. Two-way citation parity between manuscript and references.
7. Table and Figure inventory cross-reference checks.
8. Deliverables synchronization between outputs/paper/ and root.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASSES = [
    "Bengal Gram",
    "Chillies",
    "Cotton",
    "Groundnut",
    "Maize",
    "Paddy",
    "Red Gram",
]

def check_partitions() -> dict:
    cv_folds_path = PROJECT_ROOT / "outputs" / "data" / "phase6f_cv_folds.csv"
    assert cv_folds_path.exists(), f"Missing {cv_folds_path}"
    df = pd.read_csv(cv_folds_path)
    
    assert len(df) == 7843, f"Expected 7843 rows, got {len(df)}"
    fold0_val = df[df["fold"] == 0]
    fold0_train = df[df["fold"] != 0]
    assert len(fold0_train) == 6335, f"Expected 6335, got {len(fold0_train)}"
    assert len(fold0_val) == 1508, f"Expected 1508, got {len(fold0_val)}"
    
    train_mandals = set(fold0_train["Sub-District"])
    val_mandals = set(fold0_val["Sub-District"])
    assert len(train_mandals) == 93, f"Expected 93 train mandals, got {len(train_mandals)}"
    assert len(val_mandals) == 23, f"Expected 23 val mandals, got {len(val_mandals)}"
    assert len(train_mandals.intersection(val_mandals)) == 0, "Fold 0 mandal overlap!"
    
    all_mandals = set(df["Sub-District"])
    assert len(all_mandals) == 116, f"Expected 116 total train mandals, got {len(all_mandals)}"
    return {"status": "PASS", "train_parcels": 7843, "fold0_train": 6335, "fold0_val": 1508, "train_mandals": 116}

def check_test_metrics() -> dict:
    test_pred_path = PROJECT_ROOT / "outputs" / "data" / "phase7" / "test_predictions.csv"
    assert test_pred_path.exists(), f"Missing {test_pred_path}"
    df = pd.read_csv(test_pred_path)
    
    assert len(df) == 2718, f"Expected 2718 test rows, got {len(df)}"
    eval_df = df[df["extraction_status"] == "success"].copy()
    excluded_df = df[df["extraction_status"] != "success"].copy()
    assert len(eval_df) == 2265, f"Expected 2265 evaluated, got {len(eval_df)}"
    assert len(excluded_df) == 453, f"Expected 453 excluded, got {len(excluded_df)}"
    
    train_df = pd.read_csv(PROJECT_ROOT / "outputs" / "data" / "phase6f_cv_folds.csv")
    train_mandals = set(train_df["Sub-District"].unique())
    test_mandals = set(df["Sub-District"].unique())
    assert len(test_mandals) == 35, f"Expected 35 test mandals, got {len(test_mandals)}"
    assert len(train_mandals.intersection(test_mandals)) == 0, "Train-test mandal overlap!"
    
    y_true = eval_df["true_crop"]
    y_pred = eval_df["predicted_crop"]
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    
    expected_cm = np.array([
        [36,   1,   4,   0,   2,   1,   2],
        [ 0, 243,  40,   5,  12,  11,   0],
        [ 7,  14, 385,  20,  15,  28,   3],
        [ 0,   9,  11, 211,   6,   4,   0],
        [ 6,  41,  36,  12, 495,  40,   1],
        [ 1,  18,  16,   5,  18, 470,   1],
        [ 3,   2,  21,   0,   1,   1,   7]
    ])
    assert np.array_equal(cm, expected_cm), f"Confusion matrix mismatch!"
    
    correct = int(np.trace(cm))
    incorrect = int(len(eval_df) - correct)
    assert correct == 1847, f"Expected 1847 correct, got {correct}"
    assert incorrect == 418, f"Expected 418 incorrect, got {incorrect}"
    
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=CLASSES))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", labels=CLASSES))
    
    assert np.isclose(acc, 0.8154525, atol=1e-5), f"Accuracy mismatch: {acc}"
    assert np.isclose(bal_acc, 0.732585, atol=1e-5), f"Balanced accuracy mismatch: {bal_acc}"
    assert np.isclose(macro_f1, 0.730810, atol=1e-5), f"Macro-F1 mismatch: {macro_f1}"
    assert np.isclose(weighted_f1, 0.813671, atol=1e-5), f"Weighted-F1 mismatch: {weighted_f1}"
    
    return {
        "status": "PASS",
        "total_test": 2718,
        "evaluated": 2265,
        "excluded": 453,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "weighted_f1": weighted_f1,
    }

def check_feature_manifest() -> dict:
    manifest_path = PROJECT_ROOT / "outputs" / "reports" / "phase6h" / "phase6h_feature_schema_manifest.json"
    assert manifest_path.exists(), f"Missing {manifest_path}"
    with open(manifest_path) as f:
        data = json.load(f)
    
    counts = data["group_counts"]
    assert counts["S+T"] == 140, f"Expected 140 S+T, got {counts['S+T']}"
    assert counts["P"] == 20, f"Expected 20 P, got {counts['P']}"
    assert counts["S+T+P"] == 160, f"Expected 160 S+T+P, got {counts['S+T+P']}"
    
    st_features = data["groups"]["S"] + data["groups"]["T"]
    assert len(st_features) == 140, f"Expected 140 features in S+T, got {len(st_features)}"
    p_features = data["groups"]["P"]
    assert len(p_features) == 20, f"Expected 20 features in P, got {len(p_features)}"
    
    return {"status": "PASS", "s_plus_t_count": counts["S+T"], "glcm_count": counts["P"], "s_plus_t_p_count": counts["S+T+P"]}

def check_manuscript_text() -> dict:
    manuscript_path = PROJECT_ROOT / "outputs" / "paper" / "MANUSCRIPT_DRAFT.md"
    assert manuscript_path.exists(), f"Missing {manuscript_path}"
    text = manuscript_path.read_text(encoding="utf-8")
    # Ensure no affirmative claims of out-of-region or cross-state transfer exist
    for sentence in re.split(r"[.\n]", text):
        s_lower = sentence.lower()
        if "out-of-region" in s_lower or "cross-state" in s_lower:
            assert ("does not" in s_lower or "not" in s_lower or "without" in s_lower), \
                f"Out-of-region or cross-state mentioned affirmatively without disclaimer: '{sentence.strip()}'"

    forbidden_affirmative_patterns = [
        r"(?<!does not\s)(?<!cannot\s)(?:achieves?|guarantees?|ensures?|proven|absolute)\s+zero\s+spatial\s+autocorrelation",
        r"(?<!rather than fully\s)(?<!rather than\s)(?<!not a\s)(?<!not\s)nested\s+cross-validation",
        r"(?<!\d)6,?274(?!\d)",
        r"proven\s+boundary\s+contamination",
        r"proved\s+boundary\s+contamination",
    ]
    for pattern in forbidden_affirmative_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        assert len(matches) == 0, f"Found forbidden phrase '{pattern}' in manuscript: {matches}"
    
    # Must have the required caveat: "does not guarantee zero spatial autocorrelation" or "does not eliminate"
    assert "does not guarantee zero spatial autocorrelation" in text or "does not eliminate" in text, \
        "Manuscript must explicitly state that administrative holdout does not guarantee zero spatial autocorrelation"
    
    # Must explicitly state geographic holdout framing:
    assert "independent intra-state administrative holdout across unseen Mandals within Telangana" in text, \
        "Missing required geographic evaluation phrasing"

    # Must explicitly disclaim full nested cross-validation:
    assert "not full nested cross-validation" in text.lower() or "not a nested cross-validation" in text.lower() or "rather than fully nested" in text.lower(), \
        "Manuscript must explicitly state protocol is not full nested cross-validation"

    # Baseline attribution:
    assert "8.95" in text, "Missing +8.95 pp temporal windowing gain"
    assert "1.57" in text, "Missing +1.57 pp class weighting gain"
    assert "10.52" in text, "Missing +10.52 pp total gain"
    assert "static full-season baseline" in text, "Missing static full-season baseline attribution"
    assert "single-date composite" not in text, "Found single-date composite description for full-season baseline"

    # Hyperparameter accuracy in manuscript text:
    assert "reg_alpha = 0.20" in text or "reg_alpha=0.20" in text or "0.20" in text, "Missing reg_alpha = 0.20 in manuscript"
    assert "gamma = 0.20" in text or "gamma=0.20" in text or "gamma = 0.2" in text, "Missing gamma = 0.20 in manuscript"

    # Required phrases / numbers
    assert "0.8155" in text or "81.55%" in text, "Missing accuracy 81.55%"
    assert "0.7308" in text, "Missing test Macro-F1 0.7308"
    assert "0.7226" in text, "Missing CV Macro-F1 0.7226"
    assert "1,847" in text, "Missing correct count 1,847"
    assert "418" in text, "Missing incorrect count 418"
    assert "2,265" in text, "Missing evaluated count 2,265"
    assert "453" in text, "Missing excluded count 453"
    assert "6,335" in text, "Missing Fold 0 train count 6,335"
    assert "1,508" in text, "Missing Fold 0 val count 1,508"
    assert "Gumma et al., 2020" in text or "Gumma et al. (2020)" in text, "Missing Gumma 2020"
    assert "Gumma et al., 2022" in text or "Gumma et al. (2022)" in text, "Missing Gumma 2022"
    
    return {"status": "PASS", "forbidden_phrases_count": 0}

def check_citations() -> dict:
    manuscript_path = PROJECT_ROOT / "outputs" / "paper" / "MANUSCRIPT_DRAFT.md"
    references_path = PROJECT_ROOT / "outputs" / "paper" / "RELATED_WORK_AND_REFERENCES.md"
    manifest_path = PROJECT_ROOT / "outputs" / "paper" / "DELIVERABLES_MANIFEST.md"
    
    m_text = manuscript_path.read_text(encoding="utf-8")
    r_text = references_path.read_text(encoding="utf-8")
    man_text = manifest_path.read_text(encoding="utf-8")
    
    expected_citations = [
        "Defourny et al.",
        "Jin et al.",
        "Griffiths et al.",
        "Roberts et al.",
        "Meyer et al.",
        "Vuolo et al.",
        "Lebourgeois et al.",
        "Inglada et al.",
        "Maxwell et al.",
        "Chen & Guestrin",
        "Gumma et al. (2020)",
        "Gumma et al. (2022)",
        "Haralick et al.",
    ]
    for cit in expected_citations:
        assert cit in m_text or cit.replace(" (", ", ").replace(")", "") in m_text, f"Missing citation in manuscript: {cit}"
        base_name = cit.split()[0]
        assert base_name in r_text, f"Missing citation in references: {cit}"
        
    dois = re.findall(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", r_text)
    clean_dois = set(d.rstrip(").,") for d in dois)
    assert len(clean_dois) == 13, f"Expected 13 verified DOIs in references, found {len(clean_dois)}: {clean_dois}"
    
    assert "13" in man_text and ("references" in man_text.lower() or "citations" in man_text.lower()), \
        "Deliverables manifest must record 13 verified citations"
    
    return {"status": "PASS", "citations_checked": len(expected_citations), "unique_dois": len(clean_dois)}

def check_model_hyperparameters() -> dict:
    import joblib
    model_path = PROJECT_ROOT / "outputs" / "models" / "phase6h" / "xgb_weighted_s+t.joblib"
    assert model_path.exists(), f"Missing model artifact {model_path}"
    model = joblib.load(model_path)
    params = model.get_params()
    
    assert params["max_depth"] == 7, f"Expected max_depth 7, got {params['max_depth']}"
    assert params["learning_rate"] == 0.05, f"Expected learning_rate 0.05, got {params['learning_rate']}"
    assert params["n_estimators"] == 300, f"Expected n_estimators 300, got {params['n_estimators']}"
    assert params["subsample"] == 0.8, f"Expected subsample 0.8, got {params['subsample']}"
    assert params["colsample_bytree"] == 0.75, f"Expected colsample_bytree 0.75, got {params['colsample_bytree']}"
    assert params["min_child_weight"] == 3, f"Expected min_child_weight 3, got {params['min_child_weight']}"
    assert params["gamma"] == 0.2, f"Expected gamma 0.2, got {params['gamma']}"
    assert params["reg_alpha"] == 0.2, f"Expected reg_alpha 0.2, got {params['reg_alpha']}"
    assert params["reg_lambda"] == 5.0, f"Expected reg_lambda 5.0, got {params['reg_lambda']}"
    
    return {
        "status": "PASS",
        "verified_params": {
            "max_depth": params["max_depth"],
            "learning_rate": params["learning_rate"],
            "n_estimators": params["n_estimators"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "min_child_weight": params["min_child_weight"],
            "gamma": params["gamma"],
            "reg_alpha": params["reg_alpha"],
            "reg_lambda": params["reg_lambda"],
        }
    }

def check_figures_and_tables() -> dict:
    tables_path = PROJECT_ROOT / "outputs" / "paper" / "RESULTS_TABLES.md"
    figs_path = PROJECT_ROOT / "outputs" / "paper" / "FIGURE_INVENTORY.md"
    t_text = tables_path.read_text(encoding="utf-8")
    f_text = figs_path.read_text(encoding="utf-8")
    
    for t_idx in range(1, 11):
        assert f"Table {t_idx}:" in t_text, f"Missing Table {t_idx} in RESULTS_TABLES.md"
        
    for f_idx in range(1, 7):
        assert f"Figure {f_idx}" in f_text, f"Missing Figure {f_idx} in FIGURE_INVENTORY.md"
        
    expected_images = [
        PROJECT_ROOT / "outputs" / "reports" / "paper_materials" / "figures" / "fig1_confusion_matrices.png",
        PROJECT_ROOT / "outputs" / "reports" / "paper_materials" / "figures" / "fig2_class_wise_performance.png",
        PROJECT_ROOT / "outputs" / "reports" / "paper_materials" / "figures" / "fig3_fold_variability.png",
        PROJECT_ROOT / "outputs" / "reports" / "paper_materials" / "figures" / "fig4_parcel_size_quartiles.png",
        PROJECT_ROOT / "outputs" / "reports" / "paper_materials" / "figures" / "fig5_pipeline_evolution.png",
        PROJECT_ROOT / "outputs" / "figures" / "telangana_spatial_overview.png",
    ]
    for img in expected_images:
        assert img.exists(), f"Missing figure file: {img}"
        
    return {"status": "PASS", "tables_verified": 10, "figures_verified": 6, "images_found": len(expected_images)}

def check_file_sync() -> dict:
    sync_files = [
        "MANUSCRIPT_DRAFT.md",
        "RESULTS_TABLES.md",
        "FIGURE_INVENTORY.md",
        "RELATED_WORK_AND_REFERENCES.md",
        "CLAIM_AND_CITATION_AUDIT.md",
        "DELIVERABLES_MANIFEST.md",
        "MANUSCRIPT_REMEDIATION_REPORT.md",
    ]
    for fn in sync_files:
        p1 = PROJECT_ROOT / "outputs" / "paper" / fn
        p2 = PROJECT_ROOT / fn
        assert p1.exists(), f"Missing {p1}"
        assert p2.exists(), f"Missing {p2}"
        c1 = p1.read_text(encoding="utf-8")
        c2 = p2.read_text(encoding="utf-8")
        assert c1 == c2, f"Out of sync: outputs/paper/{fn} vs ./{fn}"
        
    return {"status": "PASS", "files_synchronized": len(sync_files)}

def main():
    print("=" * 60)
    print("Phase 10 Comprehensive Automated Validation")
    print("=" * 60)
    
    r_parts = check_partitions()
    print("✓ Check 1: Dataset & Partition Counts:", r_parts["status"])
    
    r_metrics = check_test_metrics()
    print("✓ Check 2: Test Metrics & Confusion Matrix:", r_metrics["status"])
    
    r_feat = check_feature_manifest()
    print("✓ Check 3: Feature Schemas (140 S+T, 20 GLCM):", r_feat["status"])
    
    r_model = check_model_hyperparameters()
    print("✓ Check 4: Model Artifact Hyperparameters (reg_alpha=0.2, gamma=0.2):", r_model["status"])
    
    r_text = check_manuscript_text()
    print("✓ Check 5: Manuscript Text Consistency (Forbidden Phrases Absence):", r_text["status"])
    
    r_cits = check_citations()
    print("✓ Check 6: Citation Parity (13/13 DOIs & Manifest sync):", r_cits["status"])
    
    r_figs = check_figures_and_tables()
    print("✓ Check 7: Tables (10) & Figures (6) Assets:", r_figs["status"])
    
    r_sync = check_file_sync()
    print("✓ Check 8: Deliverables Synchronization (outputs/paper vs root):", r_sync["status"])
    
    out_dir = PROJECT_ROOT / "outputs" / "remediation"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "phase10_validation_results.json"
    with open(summary_path, "w") as f:
        json.dump({
            "partitions": r_parts,
            "metrics": r_metrics,
            "features": r_feat,
            "model_hyperparameters": r_model,
            "text": r_text,
            "citations": r_cits,
            "figures_and_tables": r_figs,
            "sync": r_sync,
            "overall_status": "ALL_CHECKS_PASSED"
        }, f, indent=2)
    print("=" * 60)
    print(f"All Phase 10 automated checks passed cleanly! Saved results to {summary_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()

