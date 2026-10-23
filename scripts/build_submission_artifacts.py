"""
scripts/build_submission_artifacts.py
=====================================
Builds authoritative submission artifacts for ICIDA 2026:
1. outputs/submission/MANUSCRIPT_ICIDA_2026.docx (Springer Proceedings Word template)
2. outputs/submission/MANUSCRIPT_ICIDA_2026.pdf (Rendered via Word COM)
3. outputs/submission/MANUSCRIPT_ICIDA_2026_LNCS.tex (Springer LNCS LaTeX template)

Conforms strictly to:
- Springer Proceedings guidelines (A4, 122mm x 193mm text area, Times New Roman).
- Double-blind peer review rules (anonymized authors, institutions, acknowledgments).
- 100% preservation of verified evidence, tables, equations, figures, and 13 references.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "submission"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOCX_PATH = OUTPUT_DIR / "MANUSCRIPT_ICIDA_2026.docx"
PDF_PATH = OUTPUT_DIR / "MANUSCRIPT_ICIDA_2026.pdf"
TEX_PATH = OUTPUT_DIR / "MANUSCRIPT_ICIDA_2026_LNCS.tex"

# Table XML styling helpers
def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}>'
                      f'<w:top w:w="{top}" w:type="dxa"/>'
                      f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
                      f'<w:left w:w="{left}" w:type="dxa"/>'
                      f'<w:right w:w="{right}" w:type="dxa"/>'
                      f'</w:tcMar>')
    tcPr.append(tcMar)

def set_cell_shading(cell, color_hex="F2F2F2"):
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)

def set_table_borders(table):
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'<w:tblBorders {nsdecls("w")}>'
                        f'<w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
                        f'<w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
                        f'<w:left w:val="none"/>'
                        f'<w:right w:val="none"/>'
                        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>'
                        f'<w:insideV w:val="none"/>'
                        f'</w:tblBorders>')
    tblPr.append(borders)

def build_docx():
    print(f"Building Word document: {DOCX_PATH}...")
    doc = Document()
    
    # 1. Page setup: A4, Springer LNCS 122mm x 193mm text frame
    # A4 = 210mm x 297mm (8.268" x 11.693")
    # Margins: Left/Right = 44mm (1.732"), Top/Bottom = 52mm (2.047")
    section = doc.sections[0]
    section.page_width = Inches(8.268)
    section.page_height = Inches(11.693)
    section.left_margin = Inches(1.732)
    section.right_margin = Inches(1.732)
    section.top_margin = Inches(2.047)
    section.bottom_margin = Inches(2.047)
    
    # Normal style
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Times New Roman'
    style_normal.font.size = Pt(10)
    style_normal.font.color.rgb = RGBColor(0, 0, 0)
    
    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(8)
    run_title = p_title.add_run(
        "Multi-Temporal Sentinel-2 Spectral-Temporal Features for Seven-Class Crop Classification in Telangana: "
        "Spatial Cross-Validation and Independent Test Evaluation"
    )
    run_title.font.name = "Times New Roman"
    run_title.font.size = Pt(14)
    run_title.font.bold = True
    
    # Framework subtitle note
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(12)
    run_sub = p_sub.add_run(
        "Working Title for Framework Context: AI-Based Remote Sensing Framework for Crop Monitoring and "
        "Irrigation Decision Support Using Multi-Temporal Sentinel-2 Satellite Imagery"
    )
    run_sub.font.name = "Times New Roman"
    run_sub.font.size = Pt(9)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(80, 80, 80)
    
    # Authors (Double-blind anonymized)
    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.space_before = Pt(0)
    p_auth.paragraph_format.space_after = Pt(4)
    run_auth = p_auth.add_run("[Author Names Anonymized for Double-Blind Peer Review]")
    run_auth.font.name = "Times New Roman"
    run_auth.font.size = Pt(10)
    run_auth.font.bold = True
    
    # Affiliation (Anonymized)
    p_aff = doc.add_paragraph()
    p_aff.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_aff.paragraph_format.space_before = Pt(0)
    p_aff.paragraph_format.space_after = Pt(14)
    run_aff = p_aff.add_run("[Department and Institutional Affiliations Anonymized for Double-Blind Peer Review]\n[Author Correspondence Email Placeholder]")
    run_aff.font.name = "Times New Roman"
    run_aff.font.size = Pt(9)
    run_aff.font.italic = True
    run_aff.font.color.rgb = RGBColor(100, 100, 100)
    
    # Abstract
    p_abs = doc.add_paragraph()
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_abs.paragraph_format.left_indent = Inches(0.4)
    p_abs.paragraph_format.right_indent = Inches(0.4)
    p_abs.paragraph_format.space_before = Pt(6)
    p_abs.paragraph_format.space_after = Pt(6)
    p_abs.paragraph_format.line_spacing = 1.15
    
    run_absh = p_abs.add_run("Abstract. ")
    run_absh.font.name = "Times New Roman"
    run_absh.font.size = Pt(9)
    run_absh.font.bold = True
    
    abs_text = (
        "Accurate, parcel-level crop-type classification in smallholder agricultural landscapes is fundamental for regional "
        "food security, yield forecasting, and future agricultural water management. However, in semi-arid tropical regions "
        "such as Telangana, India, smallholder agricultural mapping is severely challenged by parcel fragmentation (median area "
        "~0.28 ha), optical cloud contamination during the monsoon season, extreme class imbalance, and performance over-optimism "
        "induced by spatial autocorrelation under conventional random train/test splits. In this study, we present an end-to-end, "
        "reproducible remote sensing and machine learning methodology that classifies seven major agricultural crops (Paddy, Cotton, "
        "Maize, Bengal Gram, Groundnut, Chillies, and Red Gram) using multi-temporal Sentinel-2 surface reflectance imagery. "
        "We engineer a 140-dimensional Spectral-Temporal (S+T) feature space combining full-season aggregations, phenological window "
        "compositing across early-season vegetative establishment, mid-season reproductive peak, and late-season senescence, alongside "
        "four dynamic transition trajectory deltas. To ensure authentic validation, we enforce strict 5-fold spatial cross-validation "
        "grouped by Sub-District (116 Mandals, zero Mandal group overlap) on N = 7,843 validated farm parcels, coupled with fold-internal "
        "inverse-frequency sample weighting. Our primary regularized XGBoost model achieves a pooled out-of-fold Macro-F1 score of 0.7226, "
        "Balanced Accuracy of 0.7180, and Overall Accuracy of 0.7881 (78.81%) across cross-validation. Fold-internal weighting rescues "
        "minority Red Gram F1-score from 0.3683 to 0.4420 (+7.37 percentage points). Furthermore, incorporating 20 Gray-Level Co-occurrence "
        "Matrix (GLCM) spatial texture features provided no statistically supported global improvement (p = 0.8600, 95% CI [-0.0066, +0.0092]), "
        "while parcel-area stratification revealed a performance reduction on the smallest parcel quartile (<1,257 m2, delta = -0.0133), "
        "consistent with the physical hypothesis of boundary pixel contamination. We further evaluate the frozen champion model on an "
        "independent intra-state administrative holdout of N = 2,718 surveyed farm parcels spanning 35 unseen Mandals within Telangana "
        "(zero Mandal overlap with training). Across the N = 2,265 parcels (83.33% of the surveyed cohort) where satellite features were "
        "successfully extracted across all four temporal windows, the frozen model achieves a Macro-F1 of 0.7308, Balanced Accuracy of "
        "0.7326, and Overall Accuracy of 0.8155 (81.55%), with 1,847 correct and 418 incorrect predictions. We document that persistent cloud "
        "cover caused the exclusion of 453 test parcels (16.67%), with exclusion rates varying from 4.17% on dry-season pulses to 25.91% "
        "on monsoon paddy. The metric parity between cross-validation and independent test evaluation confirms the stability of the 140-feature "
        "representation across unseen administrative units within the state, while highlighting the necessity of accounting for cloud-driven "
        "attrition in operational agricultural monitoring."
    )
    run_abst = p_abs.add_run(abs_text)
    run_abst.font.name = "Times New Roman"
    run_abst.font.size = Pt(9)
    
    # Keywords
    p_kw = doc.add_paragraph()
    p_kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_kw.paragraph_format.left_indent = Inches(0.4)
    p_kw.paragraph_format.right_indent = Inches(0.4)
    p_kw.paragraph_format.space_before = Pt(2)
    p_kw.paragraph_format.space_after = Pt(14)
    
    run_kwh = p_kw.add_run("Keywords: ")
    run_kwh.font.name = "Times New Roman"
    run_kwh.font.size = Pt(9)
    run_kwh.font.bold = True
    
    run_kwt = p_kw.add_run("Crop Classification · Sentinel-2 · Multi-Temporal Remote Sensing · Spatial Cross-Validation · Smallholder Agriculture · XGBoost · Class Imbalance · Telangana.")
    run_kwt.font.name = "Times New Roman"
    run_kwt.font.size = Pt(9)
    
    # Helper to add standard Section 1 heading
    def add_h1(title):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(title)
        run.font.name = "Times New Roman"
        run.font.size = Pt(12)
        run.font.bold = True
        return p

    def add_h2(title):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(title)
        run.font.name = "Times New Roman"
        run.font.size = Pt(10)
        run.font.bold = True
        return p

    def add_h3(title):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(title)
        run.font.name = "Times New Roman"
        run.font.size = Pt(10)
        run.font.italic = True
        return p

    def add_body(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(10)
        return p

    def add_caption(text, is_table=True):
        p = doc.add_paragraph()
        p.paragraph_format.keep_with_next = is_table
        p.paragraph_format.space_before = Pt(8 if is_table else 4)
        p.paragraph_format.space_after = Pt(3 if is_table else 10)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if is_table else WD_ALIGN_PARAGRAPH.CENTER
        
        # Split on the first period for bold label
        parts = text.split(".", 1)
        run_lbl = p.add_run(parts[0] + ".")
        run_lbl.font.name = "Times New Roman"
        run_lbl.font.size = Pt(9)
        run_lbl.font.bold = True
        
        if len(parts) > 1:
            run_txt = p.add_run(parts[1])
            run_txt.font.name = "Times New Roman"
            run_txt.font.size = Pt(9)
        return p

    def add_figure_img(img_path, caption_text, width_in=4.6):
        if Path(img_path).exists():
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(6)
            p_img.paragraph_format.space_after = Pt(2)
            p_img.paragraph_format.keep_with_next = True
            run_img = p_img.add_run()
            run_img.add_picture(str(img_path), width=Inches(width_in))
            add_caption(caption_text, is_table=False)
        else:
            print(f"Warning: Figure not found at {img_path}")

    # Build document contents
    # SECTION 1
    add_h1("1 Introduction")
    add_body(
        "Accurate, timely, and spatially explicit crop-type maps are essential for agricultural monitoring, subsidy allocation, "
        "agricultural insurance verification, and regional water resource stewardship [1, 3]. In semi-arid agrarian economies such as "
        "the state of Telangana in southern India, crop production is highly dynamic, relying on a complex mosaic of rainfed cultivation, "
        "minor tank storage, and expanding lift-irrigation networks [11]. Farmers cultivate a diverse portfolio of summer and winter crops "
        "across distinct seasons (Kharif and Rabi), including major cereals (Paddy, Maize), industrial fibers (Cotton), high-value spices "
        "(Chillies), and protein-rich pulses and oilseeds (Groundnut, Bengal Gram, Red Gram). However, conventional agricultural statistics "
        "in the region rely heavily on manual field-by-field revenue inspections (patwari surveys), which are labor-intensive, logistically "
        "demanding, prone to reporting latency, and aggregated to coarse administrative levels."
    )
    add_body(
        "Satellite remote sensing provides an objective, scalable alternative for parcel-level crop mapping. The European Space Agency’s (ESA) "
        "Copernicus Sentinel-2 constellation offers multispectral imagery at 10 m and 20 m spatial resolutions with a 5-day nominal revisit "
        "frequency, establishing an unprecedented capability for agricultural monitoring [1, 8]. Nevertheless, operational crop classification "
        "in smallholder environments faces four major scientific and practical hurdles:"
    )
    add_body(
        "1. Spectral Ambiguity in Fragmented Landscapes: Smallholder agricultural holdings in Telangana are intensely fragmented, with a median "
        "parcel area of approximately 0.28 ha (2,779 m2). At this scale, individual farm plots span only a few 10-meter pixels. Multiple co-occurring "
        "crops exhibit near-identical spectral reflectance during peak vegetative growth, making single-date optical acquisitions insufficient "
        "for reliable discrimination [2, 6]."
    )
    add_body(
        "2. Spatial Autocorrelation and Evaluation Bias: A critical flaw in much of the applied remote-sensing machine learning literature is the "
        "reliance on naive random k-fold cross-validation or unstratified random train/test splits [4, 5]. Because neighboring farm fields share "
        "identical soil types, micro-climates, irrigation canal access, and planting dates, spatial autocorrelation induces severe data leakage "
        "between training and evaluation subsets. Random splits place adjacent fields from the same village into both sets, artificially inflating "
        "out-of-sample accuracy and masking performance degradation when models are deployed to unmapped administrative jurisdictions [4, 5]."
    )
    add_body(
        "3. Severe Class Imbalance and Minority Erasure: Agrarian land use is heavily skewed toward dominant cash crops such as Paddy and Cotton. "
        "High-value or subsistence pulses, such as Red Gram (Cajanus cajan), frequently represent less than 3% of regional farm parcels. Standard "
        "unweighted classifiers maximize global accuracy by predicting majority classes, resulting in the misclassification and erasure of minority pulse crops [9]."
    )
    add_body(
        "4. Cloud-Driven Data Missingness: The Kharif cropping cycle coincides with the Southwest monsoon, generating dense, persistent cloud and "
        "cloud-shadow cover across optical acquisitions. In operational settings, cloud contamination results in incomplete time-series stacks, "
        "creating parcel-level extraction failures that are rarely accounted for transparently in published classification benchmarks."
    )
    
    add_h2("1.1 Scope Demarcation")
    add_body(
        "To maintain strict scientific integrity, we explicitly define the implemented scope of this study. While the broader conceptual framework "
        "envisions an integrated pipeline coupling crop identification with disease diagnosis, crop water stress retrieval, and FAO-56 Penman-Monteith "
        "irrigation decision support, the experimentally evaluated and validated contribution of this paper is seven-class crop-type classification. "
        "Downstream secondary tasks—specifically crop health status diagnosis using survey category labels, meteorological reanalysis integration "
        "(ERA5-Land), and evapotranspiration-based irrigation advisory generation—represent planned future extensions and are not presented as validated operational systems."
    )

    add_h2("1.2 Primary Contributions")
    add_body(
        "In this study, we address the aforementioned challenges through a rigorously audited, leakage-free empirical investigation:\n"
        "1. Multi-Temporal Phenological Feature Engineering: We design a 140-dimensional Spectral-Temporal (S+T) feature space combining full-season "
        "statistics, three distinct phenological window aggregations (Early, Mid, Late season), and four dynamic transition trajectory deltas, "
        "demonstrating an +8.95 percentage point Macro-F1 gain over the static full-season baseline (0.6174 to 0.7069) from multi-temporal windowing, "
        "rising to 0.7226 (+10.52 percentage points total) with fold-internal class weighting.\n"
        "2. Strict Spatial Group Cross-Validation: We enforce 5-fold StratifiedGroupKFold validation partitioned strictly by Sub-District (116 Mandals, "
        "zero Mandal overlap), proving that regularized XGBoost achieves a robust pooled Macro-F1 of 0.7226 and Overall Accuracy of 78.81% across 7,843 "
        "parcels without village co-location leakage.\n"
        "3. Evaluation of Spatial Texture in Smallholder Parcels: We demonstrate that incorporating 20 GLCM spatial texture features yields no statistically "
        "supported overall gain under spatial cross-validation (p = 0.8600, 95% CI [-0.0066, +0.0092]) and is associated with reduced performance in the "
        "smallest parcel quartile (<1,257 m2, delta = -0.0133), providing an empirical basis for selecting the more parsimonious 140-feature representation.\n"
        "4. Leak-Free Minority Pulse Recovery: We implement fold-internal inverse-frequency sample weighting, boosting Red Gram F1-score from 0.3683 to "
        "0.4420 (+7.37 percentage points) while preserving strict spatial validation boundaries.\n"
        "5. Independent Intra-State Administrative Holdout with Cloud Accounting: We execute frozen-model inference on an independent intra-state "
        "administrative holdout of 2,718 surveyed parcels across 35 unseen Mandals in Telangana (zero Mandal overlap with training), establishing "
        "that the model achieves 0.7308 Macro-F1 and 81.55% accuracy on the 2,265 successfully extracted parcels while transparently documenting a 16.67% cloud-exclusion rate."
    )

    # SECTION 2
    add_h1("2 Related Work")
    add_h2("2.1 Multi-Temporal Satellite Optical Mapping")
    add_body(
        "Early automated crop mapping relied heavily on Landsat or MODIS imagery, which offered either high temporal frequency with coarse spatial "
        "resolution (250–500 m) or moderate spatial resolution (30 m) with a 16-day revisit cycle that frequently missed cloud-free windows [8, 12]. "
        "The advent of Sentinel-2 MSI transformed smallholder mapping by providing 10-meter visible and near-infrared bands alongside red-edge "
        "spectral bands specifically sensitive to canopy chlorophyll content and nitrogen status [1]. Vuolo et al. [6] demonstrated that multi-temporal "
        "Sentinel-2 acquisitions increased classification accuracy by 5% to 17% compared to single-date imagery. Similarly, Griffiths et al. [3] "
        "proved that intra-annual phenological compositing metrics effectively separate spectrally overlapping crops across heterogeneous climatic "
        "gradients. In India, Gumma et al. [11] demonstrated the utility of Sentinel-2 NDVI time series for mapping field-level pulse and cereal "
        "crops in semi-arid environments, while Gumma et al. [12] established cloud-based machine learning workflows on Google Earth Engine for "
        "multi-cropland products across South Asia."
    )

    add_h2("2.2 Spatial Cross-Validation and Spatial Autocorrelation")
    add_body(
        "The necessity of spatial blocking in geospatial machine learning has been established by Roberts et al. [4] and Meyer et al. [5]. "
        "Spatial autocorrelation in environmental covariates means that standard random splits violate the fundamental assumption that training "
        "and validation instances are independently and identically distributed. When random cross-validation is applied to agricultural parcel "
        "datasets, models memorize local spatial context rather than generalizable spectral-temporal relationships, producing over-optimistic "
        "performance estimates that fail to transfer to unmapped regions [5]. Roberts et al. [4] documented that non-spatial cross-validation can "
        "underestimate true prediction error by 20% to 60% in spatially structured datasets. Spatially clustered or administrative-group "
        "cross-validation provides an honest estimate of out-of-Mandal predictive power, although it does not eliminate residual spatial autocorrelation across administrative borders."
    )

    add_h2("2.3 Class Imbalance and Cost-Sensitive Learning")
    add_body(
        "Agricultural classification tasks frequently suffer from severe class imbalance. As noted by Maxwell et al. [9], standard loss functions "
        "in gradient-boosted decision trees optimize global error rates at the expense of minority classes. Synthetic oversampling methods, such as "
        "SMOTE, can create unrealistic spectral signatures that do not conform to physical radiative transfer principles. Inverse-frequency sample "
        "weighting adjusts the gradient contribution of each class according to its empirical rarity, providing a principled mechanism to recover "
        "minority decision boundaries without distorting physical feature distributions [9, 10]."
    )

    add_h2("2.4 Smallholder Parcel Fragmentation and Spatial Texture")
    add_body(
        "In smallholder systems, parcel boundaries are irregular and individual field extents are often smaller than 0.5 ha [2]. While spatial "
        "texture metrics, such as Haralick’s Gray-Level Co-occurrence Matrix (GLCM; [13]), have demonstrated value in homogeneous, broad-acre "
        "commercial farms [6], Lebourgeois et al. [7] highlighted that plots under 0.25 ha suffer from severe boundary contamination, where 30% "
        "to over 50% of parcel pixels overlap non-agricultural borders. Despite this, prior studies have rarely quantified the precise parcel-size "
        "threshold where spatial texture transitions from an asset to a liability."
    )

    # SECTION 3
    add_h1("3 Study Area and Dataset")
    add_h2("3.1 Study Area")
    add_body(
        "The study was conducted across the state of Telangana in southern India (Fig. 1), covering an area of approximately 112,077 km2. The region "
        "experiences a semi-arid tropical climate characterized by three distinct seasons: the Southwest monsoon (Kharif, June to October), the dry winter "
        "(Rabi, November to March), and a hot summer (March to May). Annual rainfall ranges between 700 mm and 1,200 mm, with more than 80% concentrated "
        "during the monsoon months. Agricultural production is highly heterogeneous, spanning black cotton soils (vertisols), red sandy loams (alfisols), "
        "and alluvial canal command zones."
    )
    add_figure_img(
        PROJECT_ROOT / "outputs" / "figures" / "telangana_spatial_overview.png",
        "Fig. 1. Spatial distribution of training (116 Mandals) and independent test (35 Mandals) agricultural parcels across Telangana, India."
    )

    add_h2("3.2 Ground-Truth Parcel Cohorts")
    add_body(
        "The empirical ground-truth dataset consists of verified vector polygons derived from extensive field surveys conducted across Telangana's "
        "agricultural landscape:\n"
        "1. Training Cohort (Train.csv): Comprises 7,888 initial surveyed parcel polygons spanning 14 districts and 116 Sub-Districts (Mandals). "
        "Following cloud-screening and topological validation, 45 parcels failed feature extraction across all temporal windows, leaving an "
        "authoritative training cohort of N = 7,843 eligible farm parcels.\n"
        "2. Independent Test Cohort (Test.csv): Comprises N = 2,718 farm parcels distributed across 35 Mandals in 12 districts within Telangana. "
        "A rigorous preflight spatial audit confirmed zero Mandal overlap between the training and test cohorts, establishing an independent "
        "intra-state administrative holdout across unseen Mandals within Telangana (Table 1). We emphasize that while Mandals are strictly disjoint, "
        "ten broader administrative districts overlap between training and test sets, meaning held-out test Mandals may be geographically contiguous "
        "with training Mandals and share regional agro-climatic conditions; this evaluation does not establish cross-state transfer, out-of-region "
        "transfer, or the absence of spatial autocorrelation.\n"
        "3. Target Nomenclature: The surveyed ground-truth crop label (Crop) encompasses seven distinct economic crop classes: Bengal Gram, Chillies, "
        "Cotton, Groundnut, Maize, Paddy, and Red Gram."
    )

    # TABLE 1
    add_caption("Table 1. Cohort Composition and Spatial Partitioning.")
    t1_data = [
        ["Cohort Property", "Training Cohort (Train.csv)", "Independent Held-Out Test Cohort (Test.csv)"],
        ["Initial Surveyed Parcels", "7,888", "2,718"],
        ["Persistent Cloud / Shadow Failures", "45 (0.57%)", "453 (16.67%)"],
        ["Eligible Evaluated Parcels", "7,843 (99.43%)", "2,265 (83.33%)"],
        ["Administrative Units (Mandals)", "116", "35"],
        ["Districts Covered", "14", "12 (10 districts overlap with training cohort)"],
        ["Inter-Cohort Mandal Overlap", "—", "0 Mandals (Disjoint Mandal Holdout within Telangana)"],
        ["Validation Design", "5-Fold StratifiedGroupKFold", "Frozen Model Out-of-Mandal Inference"],
        ["Target Variable", "Crop (7 Ground-Truth Classes)", "Crop (7 Ground-Truth Classes)"],
        ["Secondary Target (category)", "Preserved (Health status; not trained)", "Preserved (Health status; not evaluated)"],
    ]
    t1 = doc.add_table(rows=len(t1_data), cols=3)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t1)
    for r_idx, row in enumerate(t1_data):
        for c_idx, val in enumerate(row):
            cell = t1.cell(r_idx, c_idx)
            cell.text = val
            set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.runs[0]
            run.font.name = "Times New Roman"
            run.font.size = Pt(8.5)
            if r_idx == 0:
                run.font.bold = True
                set_cell_shading(cell, "EAEAEA")
    
    p_t1_note = doc.add_paragraph()
    p_t1_note.paragraph_format.space_before = Pt(2)
    p_t1_note.paragraph_format.space_after = Pt(8)
    run_t1_note = p_t1_note.add_run(
        "Note: Both cohorts reside within Telangana. Disjoint Mandals provide administrative separation but do not guarantee geographic "
        "independence or eliminate spatial autocorrelation. Ten broader districts overlap between cohorts."
    )
    run_t1_note.font.name = "Times New Roman"
    run_t1_note.font.size = Pt(8)
    run_t1_note.font.italic = True

    # TABLE 2
    add_caption("Table 2. Parcel-Level Feature Extraction Coverage & Cloud Exclusion Accounting.")
    t2_data = [
        ["Crop Class", "Total Test Parcels", "Evaluated Parcels", "Cloud-Excluded Parcels", "Extraction Coverage (%)", "Exclusion Rate (%)"],
        ["Bengal Gram", "48", "46", "2", "95.83%", "4.17%"],
        ["Groundnut", "266", "241", "25", "90.60%", "9.40%"],
        ["Maize", "699", "631", "68", "90.27%", "9.73%"],
        ["Red Gram", "41", "35", "6", "85.37%", "14.63%"],
        ["Cotton", "569", "472", "97", "82.95%", "17.05%"],
        ["Chillies", "381", "311", "70", "81.63%", "18.37%"],
        ["Paddy", "714", "529", "185", "74.09%", "25.91%"],
        ["Total Cohort", "2,718", "2,265", "453", "83.33%", "16.67%"],
    ]
    t2 = doc.add_table(rows=len(t2_data), cols=6)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t2)
    for r_idx, row in enumerate(t2_data):
        for c_idx, val in enumerate(row):
            cell = t2.cell(r_idx, c_idx)
            cell.text = val
            set_cell_margins(cell, top=50, bottom=50, left=80, right=80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.runs[0]
            run.font.name = "Times New Roman"
            run.font.size = Pt(8.5)
            if r_idx == 0 or r_idx == len(t2_data) - 1:
                run.font.bold = True
                set_cell_shading(cell, "EAEAEA" if r_idx == 0 else "F8F8F8")

    add_h2("3.3 Vector Topology and Geometric Validation")
    add_body(
        "All parcel boundary shapefiles were projected from WGS 84 geographic coordinates (EPSG:4326) to the local Universal Transverse "
        "Mercator projection (UTM Zone 44N, EPSG:32644) to enable accurate metric parcel area calculations. Geometric validity was verified "
        "using GEOS; invalid self-intersecting polygon rings were repaired using shapely.validation.make_valid(). Parcels range in size from "
        "54 m2 to 69,491 m2, with a median parcel area of 2,779 m2 (0.278 ha), reflecting typical smallholder fragmentation."
    )

    # SECTION 4
    add_h1("4 Proposed Methodology")
    add_body(
        "The end-to-end classification framework comprises four sequential stages: satellite ingestion and cloud masking, phenological "
        "feature engineering, regularized spatial learning with class weighting, and out-of-Mandal frozen inference."
    )
    add_h2("4.1 Satellite Preprocessing and Cloud Masking")
    add_body(
        "Multi-temporal surface reflectance imagery was ingested from the Copernicus Sentinel-2 Level-2A archive (COPERNICUS/S2_SR_HARMONIZED) "
        "using Google Earth Engine (GEE). For every acquisition intersecting a farm parcel, non-vegetated and compromised pixels were identified "
        "using the Scene Classification Layer (SCL). Pixels flagged as saturated/defective, cloud shadows, cloud medium probability, cloud high "
        "probability, or thin cirrus were masked out. To avoid temporal swath overweighting caused by overlapping orbit footprints, acquisitions "
        "from the same calendar date were daily-mosaicked before computing parcel zonal statistics."
    )

    add_h2("4.2 Phenological Windowing and Feature Representation")
    add_body(
        "Agricultural parcels exhibit distinct phenological rhythms across their cultivation cycles. We establish four discrete temporal "
        "extraction windows tailored to Telangana’s cropping calendar:\n"
        "• Early Season (Window 1: Jun 01 – Jul 31): Land preparation, sowing, and vegetative emergence.\n"
        "• Mid Season (Window 2: Aug 01 – Sep 30): Peak canopy development, tillering, and flowering.\n"
        "• Late Season (Window 3: Oct 01 – Nov 30): Maturation, grain filling, boll opening, and senescence.\n"
        "• Full Season (Full Window: Jun 01 – Dec 31): Aggregate seasonal cycle.\n"
        "Within each temporal window, clear pixels were aggregated using temporal median compositing. For each parcel polygon, we extracted "
        "zonal mean reflectance across 10 spectral bands: Blue (B2), Green (B3), Red (B4), Red Edge 1 (B5), Red Edge 2 (B6), Red Edge 3 (B7), "
        "NIR (B8), Narrow NIR (B8A), SWIR 1 (B11), and SWIR 2 (B12). Furthermore, 14 biophysical, chlorophyll, and water indices were calculated "
        "per window (Table 3), generating 34 features per window (34 x 3 = 102 windowed features, plus 34 full-season features = 136 features)."
    )
    add_body(
        "To capture dynamic phenological transitions across growth stages, we engineer four explicit transition delta features:\n"
        "1. Vegetative Growth Delta: ndvi_growth_delta = NDVI_Mid - NDVI_Early\n"
        "2. Peak-to-Early Vigor Ratio: peak_to_early_ratio = NDVI_Mid / (NDVI_Early + 1e-6)\n"
        "3. Senescence Drop Delta: ndvi_senescence_delta = NDVI_Late - NDVI_Mid\n"
        "4. Late-Season Moisture Loss: moisture_loss_delta = NDMI_Late - NDMI_Mid\n"
        "The final Spectral-Temporal (S+T) feature space comprises exactly 140 continuous features (34 + 102 + 4 = 140)."
    )

    # TABLE 3
    add_caption("Table 3. 140-Dimensional Spectral-Temporal Feature Representation.")
    t3_data = [
        ["Feature Group", "Temporal Scope / Window", "Feature Description", "Count"],
        ["Spectral & Biophysical Indices (S)", "Full Season (Jun 01 – Dec 31)", "10 S2 L2A surface reflectance bands + 14 spectral indices (NDVI, EVI, NDMI, NDWI, GNDVI, NDRE, MSAVI, CI Green, CI Red Edge, MSI, NDBI, SAVI, SIPI, NMDI). Extracted via median aggregation.", "34"],
        ["Windowed Phenology (W1: Early)", "Early Season (Jun 01 – Jul 31)", "Identical 34 spectral bands and vegetation/water indices extracted during crop establishment / vegetative emergence.", "34"],
        ["Windowed Phenology (W2: Mid)", "Mid Season (Aug 01 – Sep 30)", "Identical 34 spectral bands and vegetation/water indices extracted during peak vegetative vigor / flowering.", "34"],
        ["Windowed Phenology (W3: Late)", "Late Season (Oct 01 – Nov 30)", "Identical 34 spectral bands and vegetation/water indices extracted during maturity / senescence.", "34"],
        ["Dynamic Transition Deltas (T)", "Multi-Window Trajectories", "1. ndvi_growth_delta; 2. peak_to_early_ratio; 3. ndvi_senescence_delta; 4. moisture_loss_delta.", "4"],
        ["Total Dimensionality", "—", "Full Spectral-Temporal Vector Space (S + T)", "140"],
    ]
    t3 = doc.add_table(rows=len(t3_data), cols=4)
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t3)
    for r_idx, row in enumerate(t3_data):
        for c_idx, val in enumerate(row):
            cell = t3.cell(r_idx, c_idx)
            cell.text = val
            set_cell_margins(cell, top=50, bottom=50, left=80, right=80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx < 3 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.runs[0]
            run.font.name = "Times New Roman"
            run.font.size = Pt(8.5)
            if r_idx == 0 or r_idx == len(t3_data) - 1:
                run.font.bold = True
                set_cell_shading(cell, "EAEAEA" if r_idx == 0 else "F8F8F8")

    add_h2("4.3 Spatial Texture Features (Ablation Pipeline)")
    add_body(
        "To rigorously evaluate spatial texture, we constructed an augmented 160-dimensional Spectral-Temporal-Spatial (S+T+P) representation. "
        "We computed Gray-Level Co-occurrence Matrix (GLCM; [13]) texture rasters across the NIR band (B8) at 10-meter resolution using a 3x3 "
        "sliding window. Five standard second-order texture statistics (Angular Second Moment, Contrast, Correlation, Entropy, and Variance) "
        "were computed across each of the four temporal windows (5 x 4 = 20 GLCM features) and aggregated per parcel."
    )

    add_h2("4.4 Model Architecture and Class-Weighted Optimization")
    add_body(
        "We deploy Extreme Gradient Boosting (XGBoost; [10]) as our core classifier family due to its proven capacity to model complex non-linear "
        "feature interactions in tabular remote sensing representations. To counter extreme class imbalance without distorting feature manifolds "
        "through synthetic oversampling, we calculate inverse-frequency sample weights w_c for each crop class c in {1, ..., C}:"
    )
    p_eq1 = doc.add_paragraph()
    p_eq1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq1.paragraph_format.space_before = Pt(4)
    p_eq1.paragraph_format.space_after = Pt(4)
    r_eq1 = p_eq1.add_run("w_c = N_train / (C · N_c)                                                  (1)")
    r_eq1.font.name = "Times New Roman"
    r_eq1.font.size = Pt(9.5)
    r_eq1.font.italic = True
    add_body(
        "where N_train is the total number of training samples in the partition, C = 7 is the number of crop classes, and N_c is the class-specific "
        "sample support. Crucially, sample weights are computed strictly inside each training fold, preventing any validation partition frequency leakage."
    )

    add_h2("4.5 Spatial Validation Design")
    add_body(
        "To prevent spatial data leakage, validation is structured as a 5-fold StratifiedGroupKFold grouped strictly by Mandal (Sub-District). "
        "Parcels belonging to the same Mandal are assigned exclusively to either the training set or the validation set in any given fold. "
        "Across the 116 training Mandals, each validation fold evaluates between 21 and 26 completely unseen Mandals (1,508–1,610 parcels per fold), "
        "preventing parcel co-location leakage and ensuring zero administrative group overlap between training and validation folds."
    )

    add_h2("4.6 Hyperparameter Optimization Protocol")
    add_body(
        "Hyperparameters were evaluated via an exploratory bounded search of ten candidate configurations assessed via 3-fold grouped inner "
        "cross-validation strictly on the training partition of Outer Fold 0 (N = 6,335 parcels across 93 Mandals; Outer Fold 0 validation "
        "partition has N = 1,508 parcels across 23 Mandals). Selection was based on inner Macro-F1 (0.6797). The selected configuration—"
        "max_depth=7, learning_rate=0.05, n_estimators=300, subsample=0.80, colsample_bytree=0.75, min_child_weight=3, gamma=0.20, "
        "reg_alpha=0.20, and reg_lambda=5.0—was frozen and evaluated across all five spatial cross-validation folds and the independent test "
        "inference pipeline (Table 4). This design represents an exploratory parameter search on a single outer-fold training partition rather "
        "than fully nested cross-validation across all five folds. While inner grouped CV ensured zero validation leakage during selection, "
        "freezing parameters chosen on Fold 0 training data entails potential selection bias, though Fold 0 exhibited typical distribution properties."
    )

    # TABLE 4
    add_caption("Table 4. XGBoost Hyperparameter Specification and Tuning Protocol.")
    t4_data = [
        ["Hyperparameter", "Selected Value", "Search Space Evaluated (10 Trials)", "Tuning Protocol & Context"],
        ["n_estimators", "300", "[100, 200, 300, 500]", "Fixed during exploration"],
        ["max_depth", "7", "[5, 6, 7, 8]", "Selected for non-linear canopy interactions"],
        ["learning_rate (eta)", "0.05", "[0.03, 0.05, 0.10]", "Conservative shrinkage for generalization"],
        ["subsample", "0.80", "[0.70, 0.80, 0.90]", "Row sub-sampling per boosting iteration"],
        ["colsample_bytree", "0.75", "[0.60, 0.75, 0.80, 1.00]", "Feature sub-sampling per tree split"],
        ["min_child_weight", "3", "[1, 3, 5]", "Prevents fitting spurious smallholder clusters"],
        ["gamma", "0.20", "[0.0, 0.1, 0.2, 0.3]", "Minimum loss reduction to partition leaf"],
        ["reg_lambda (L2)", "5.0", "[1.0, 2.0, 3.0, 5.0, 10.0]", "Heavy L2 regularization against spatial noise"],
        ["reg_alpha (L1)", "0.20", "[0.0, 0.1, 0.2, 0.5, 1.0]", "Mild sparsity enforcement"],
        ["tree_method", "hist", "['hist']", "Fast exact histogram binning"],
        ["Class Weighting", "Inverse-Frequency", "[None, Balanced, Inverse-Freq]", "w_c = N / (C · N_c); computed fold-internally"],
    ]
    t4 = doc.add_table(rows=len(t4_data), cols=4)
    t4.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t4)
    for r_idx, row in enumerate(t4_data):
        for c_idx, val in enumerate(row):
            cell = t4.cell(r_idx, c_idx)
            cell.text = val
            set_cell_margins(cell, top=50, bottom=50, left=80, right=80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 or c_idx == 3 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.runs[0]
            run.font.name = "Times New Roman"
            run.font.size = Pt(8.5)
            if r_idx == 0:
                run.font.bold = True
                set_cell_shading(cell, "EAEAEA")

    p_t4_note = doc.add_paragraph()
    p_t4_note.paragraph_format.space_before = Pt(2)
    p_t4_note.paragraph_format.space_after = Pt(8)
    run_t4_note = p_t4_note.add_run(
        "Methodological Note: Hyperparameters were tuned via 3-fold grouped inner CV strictly on Outer Fold 0 training partition (6,335 parcels, 93 Mandals). "
        "Selected parameters were frozen and reused across all folds; this is not full nested CV."
    )
    run_t4_note.font.name = "Times New Roman"
    run_t4_note.font.size = Pt(8)
    run_t4_note.font.italic = True

    # SECTION 5
    add_h1("5 Results and Discussion")
    add_h2("5.1 Primary Benchmark Performance under Spatial Cross-Validation")
    add_body(
        "Across the authoritative training cohort of N = 7,843 parcels in 116 Mandals, the primary Weighted S+T XGBoost model achieves a pooled "
        "out-of-fold Macro-F1 score of 0.7226, Balanced Accuracy of 0.7180, and Overall Accuracy of 0.7881 (78.81%) (Table 5). The fold-mean "
        "Macro-F1 across the five spatial partitions is 0.7117 ± 0.0181 (population standard deviation, ddof=0) or ± 0.0202 (sample standard deviation, "
        "ddof=1), demonstrating high stability across diverse sub-districts (Fig. 2)."
    )

    # TABLE 5
    add_caption("Table 5. 5-Fold Spatial Cross-Validation Performance (Weighted S+T, N=7,843).")
    t5_data = [
        ["Spatial Fold ID", "Validation Mandals", "Validation Parcels", "Macro-F1", "Overall Accuracy", "Balanced Accuracy", "Weighted-F1"],
        ["Fold 0", "23", "1,508", "0.7269", "0.7960", "0.7203", "0.7932"],
        ["Fold 1", "21", "1,600", "0.7381", "0.8013", "0.7329", "0.7997"],
        ["Fold 2", "22", "1,595", "0.6919", "0.7717", "0.6865", "0.7681"],
        ["Fold 3", "26", "1,530", "0.6945", "0.7749", "0.6911", "0.7728"],
        ["Fold 4", "24", "1,610", "0.7073", "0.7966", "0.7102", "0.7946"],
        ["Fold Mean ± Pop. SD", "23.2", "1,568.6", "0.7117 ± 0.0181", "0.7881 ± 0.0123", "0.7082 ± 0.0177", "0.7857 ± 0.0128"],
        ["Fold Mean ± Sample SD", "23.2", "1,568.6", "0.7117 ± 0.0202", "0.7881 ± 0.0137", "0.7082 ± 0.0198", "0.7857 ± 0.0143"],
        ["Pooled Out-of-Fold", "116", "7,843", "0.7226", "0.7881 (78.81%)", "0.7180", "0.7857"],
    ]
    t5 = doc.add_table(rows=len(t5_data), cols=7)
    t5.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t5)
    for r_idx, row in enumerate(t5_data):
        for c_idx, val in enumerate(row):
            cell = t5.cell(r_idx, c_idx)
            cell.text = val
            set_cell_margins(cell, top=50, bottom=50, left=70, right=70)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.runs[0]
            run.font.name = "Times New Roman"
            run.font.size = Pt(8.5)
            if r_idx == 0 or r_idx == len(t5_data) - 1:
                run.font.bold = True
                set_cell_shading(cell, "EAEAEA" if r_idx == 0 else "F8F8F8")

    add_figure_img(
        PROJECT_ROOT / "outputs" / "reports" / "paper_materials" / "figures" / "fig3_fold_variability.png",
        "Fig. 2. Performance metric dispersion across 5 outer spatial cross-validation folds.",
        width_in=4.4
    )

    add_h2("5.2 Feature Ablation and Systematic Pipeline Evolution")
    add_body(
        "To isolate the sources of classification performance, we evaluated five distinct pipeline iterations under identical spatial cross-validation protocols (Table 6, Fig. 3):\n"
        "1. Static Single-Window Baselines: The Early-Season single-window model achieved a Macro-F1 of only 0.5058 (Accuracy: 63.58%), reflecting severe vegetative ambiguity during emergence. "
        "Capturing full-season static median reflectance increased Macro-F1 to 0.6174 (Accuracy: 72.08%), indicating that seasonal composites reduce variance but lose temporal trajectories.\n"
        "2. Multi-Temporal Windowing Gain: Introducing the multi-window framework (140 S+T features, unweighted) propelled Macro-F1 to 0.7069 (Accuracy: 79.24%), an improvement of +8.95 percentage points "
        "over the static full-season baseline. This confirms that multi-stage phenological tracking provides vital discriminatory power.\n"
        "3. Sample Weighting Impact: Implementing fold-internal inverse-frequency weighting increased Macro-F1 from 0.7069 to 0.7226 (+1.57 pp). Crucially, weighting dramatically rescued the extreme minority "
        "pulse crop, Red Gram, raising its F1-score from 0.3683 to 0.4420 (+7.37 pp) and precision from 0.6170 to 0.5467 under strict spatial isolation."
    )

    # TABLE 6
    add_caption("Table 6. Model Evolution, Feature Ablations, and Significance Testing.")
    t6_data = [
        ["Model / Pipeline Stage", "Features", "Cohort (N)", "Weighting Strategy", "Spatial Macro-F1", "Overall Accuracy", "Balanced Accuracy", "Statistical Contrast vs Weighted S+T"],
        ["Phase 6D Satellite Pilot", "34", "714", "Unweighted", "0.5511", "0.6653", "0.5312", "Exploratory cohort; baseline"],
        ["Early-Season Single Window", "34", "7,843", "Unweighted", "0.5058", "0.6358", "0.4902", "Delta = -0.2168 (-21.68 pp)"],
        ["Full-Season Single Window", "34", "7,843", "Unweighted", "0.6174", "0.7208", "0.6015", "Delta = -0.1052 (-10.52 pp)"],
        ["Unweighted Multi-Temporal S+T", "140", "7,843", "Unweighted", "0.7069", "0.7924", "0.6982", "Delta = -0.0157 (-1.57 pp)"],
        ["Unweighted S+T+P (+20 GLCM)", "160", "7,843", "Unweighted", "0.7038", "0.7900", "0.6895", "Delta = -0.0188 (-1.88 pp)"],
        ["Final Weighted S+T (Winner)", "140", "7,843", "Inverse-Freq", "0.7226", "0.7881", "0.7180", "Reference Champion Pipeline"],
        ["Weighted S+T+P (+20 GLCM)", "160", "7,843", "Inverse-Freq", "0.7228", "0.7905", "0.7188", "Delta = +0.0002 (p=0.8600, 95% CI [-0.0066, +0.0092])"],
    ]
    t6 = doc.add_table(rows=len(t6_data), cols=8)
    t6.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t6)
    for r_idx, row in enumerate(t6_data):
        for c_idx, val in enumerate(row):
            cell = t6.cell(r_idx, c_idx)
            cell.text = val
            set_cell_margins(cell, top=50, bottom=50, left=60, right=60)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 or c_idx == 7 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.runs[0]
            run.font.name = "Times New Roman"
            run.font.size = Pt(8)
            if r_idx == 0 or r_idx == 6:
                run.font.bold = True
                set_cell_shading(cell, "EAEAEA" if r_idx == 0 else "F8F8F8")

    add_figure_img(
        PROJECT_ROOT / "outputs" / "reports" / "paper_materials" / "figures" / "fig5_pipeline_evolution.png",
        "Fig. 3. Stepwise feature ablation progression from single-window baseline to champion weighted S+T pipeline.",
        width_in=4.4
    )

    add_h2("5.3 Spatial Texture Evaluation and Parcel-Size Stratification")
    add_body(
        "We evaluated whether incorporating GLCM spatial texture features improves smallholder crop classification. The 160-dimensional "
        "Weighted S+T+P model achieved a pooled Macro-F1 of 0.7228, virtually identical to the 140-dimensional Weighted S+T model (0.7226, delta = +0.0002). "
        "A paired Student's t-test across the five spatial folds revealed no statistically significant difference (t = 0.3037, p = 0.7765). Furthermore, "
        "a 1,000-iteration Mandal cluster bootstrap confirmed that the global performance difference is statistically indistinguishable from zero "
        "(p = 0.8600, 95% CI [-0.0066, +0.0092])."
    )
    add_body(
        "To understand why texture failed to yield benefits, we conducted an exploratory parcel-size quartile stratification (N = 7,843; Table 10, Fig. 4):\n"
        "• In Quartile 1 (smallest parcels, 54–1,257 m2, median area ~0.08 ha), the inclusion of spatial texture was associated with a descriptive "
        "drop in Macro-F1 of -0.0133 (-1.33 pp) from 0.6598 to 0.6465.\n"
        "• In Quartile 2 (1,257–2,783 m2), texture showed a minor negative association (delta = -0.0148).\n"
        "• In Quartile 3 (2,784–5,517 m2), texture showed a positive association (delta = +0.0130), while in Quartile 4 (>5,528 m2), performance was virtually identical (delta = -0.0034).\n"
        "This pattern is consistent with the physical hypothesis of boundary pixel contamination: in smallholder parcels comprising only a few 10-meter pixels, "
        "3x3 convolution kernels (900 m2) inevitably overlap non-agricultural bunds, field paths, and adjacent crops, introducing edge-pixel noise. "
        "We emphasize that this boundary contamination is a plausible physical hypothesis rather than directly measured or causally proven ground truth. "
        "Furthermore, the absence of a statistically significant difference does not constitute proof of equivalence or demonstrate that texture features "
        "are universally ineffective across all spatial resolutions. Rather, based on parsimony (Occam’s razor), the 140-feature Spectral-Temporal "
        "representation was selected as the champion architecture, eliminating 20 computationally intensive GEE texture rasters without measurable performance penalty."
    )

    # TABLE 10
    add_caption("Table 10. Parcel Area Quartile Stratification: GLCM Spatial Texture vs Spectral-Temporal.")
    t10_data = [
        ["Quartile", "Area Range (m2)", "Area Description", "Parcels (N)", "Weighted S+T Macro-F1", "Weighted S+T+P Macro-F1", "Delta", "Physical Hypothesis"],
        ["Q1", "54 – 1,257", "Smallest Parcels", "1,961", "0.6598", "0.6465", "-0.0133 (-1.33 pp)", "Consistent with boundary contamination (3x3 kernel covers field edges)"],
        ["Q2", "1,257 – 2,783", "Medium-Low Area", "1,961", "0.7332", "0.7184", "-0.0148 (-1.48 pp)", "Intermediate parcel scale; texture associated with slight drop"],
        ["Q3", "2,784 – 5,517", "Medium-High Area", "1,960", "0.7400", "0.7530", "+0.0130 (+1.30 pp)", "Interior parcel pixels exceed boundary footprint; texture captures canopy"],
        ["Q4", "5,528 – 69,491", "Largest Holdings", "1,961", "0.7253", "0.7219", "-0.0034 (-0.34 pp)", "Homogeneous interior canopy; negligible difference"],
        ["Full", "54 – 69,491", "Entire Cohort", "7,843", "0.7226", "0.7228", "+0.0002 (+0.02 pp)", "Global difference is statistically indistinguishable (p = 0.8600)"],
    ]
    t10 = doc.add_table(rows=len(t10_data), cols=8)
    t10.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t10)
    for r_idx, row in enumerate(t10_data):
        for c_idx, val in enumerate(row):
            cell = t10.cell(r_idx, c_idx)
            cell.text = val
            set_cell_margins(cell, top=50, bottom=50, left=60, right=60)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 or c_idx == 7 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.runs[0]
            run.font.name = "Times New Roman"
            run.font.size = Pt(8)
            if r_idx == 0 or r_idx == len(t10_data) - 1:
                run.font.bold = True
                set_cell_shading(cell, "EAEAEA" if r_idx == 0 else "F8F8F8")

    add_figure_img(
        PROJECT_ROOT / "outputs" / "reports" / "paper_materials" / "figures" / "fig4_parcel_size_quartiles.png",
        "Fig. 4. Macro-F1 delta between S+T+P and S+T across parcel-area quartiles, illustrating boundary contamination vulnerability.",
        width_in=4.4
    )

    add_h2("5.4 Independent Test Set Evaluation across 35 Unseen Mandals")
    add_body(
        "To evaluate model performance across unseen administrative units within Telangana, the frozen champion model was deployed on an "
        "independent intra-state administrative holdout (Test.csv, N = 2,718 parcels) across 35 geographically unseen Mandals in 12 districts (Table 7).\n"
        "Across the N = 2,265 successfully extracted parcels (83.33% of the surveyed cohort), the model achieved:\n"
        "• Macro-F1 Score: 0.7308 (+0.0082 vs CV)\n"
        "• Overall Accuracy: 0.8155 (81.55%) (+0.0274 vs CV)\n"
        "• Balanced Accuracy: 0.7326 (+0.0146 vs CV)\n"
        "• Weighted-F1 Score: 0.8137 (+0.0280 vs CV)\n"
        "• Prediction Counts: 1,847 correct classifications, 418 misclassifications."
    )

    # TABLE 7
    add_caption("Table 7. Independent Test Performance vs Spatial Cross-Validation Parity.")
    t7_data = [
        ["Metric", "5-Fold Spatial CV (Pooled N=7,843)", "Independent Test Set (Evaluated N=2,265)", "Observed Delta", "Interpretation / Context"],
        ["Macro-F1", "0.7226", "0.7308", "+0.0082", "Confirms consistent performance across 35 unseen Mandals"],
        ["Overall Accuracy", "0.7881 (78.81%)", "0.8155 (81.55%)", "+0.0274", "Influenced by higher proportion of Maize in test set"],
        ["Balanced Accuracy", "0.7180", "0.7326", "+0.0146", "Steady unweighted per-class discrimination"],
        ["Weighted-F1", "0.7857", "0.8137", "+0.0280", "Reflects high accuracy on dominant test classes"],
        ["Evaluated Sample Size", "7,843", "2,265", "—", "116 training Mandals vs 35 unseen test Mandals"],
        ["Excluded Sample Size", "45 (0.57%)", "453 (16.67%)", "—", "Higher monsoon cloud impact in test cohort"],
    ]
    t7 = doc.add_table(rows=len(t7_data), cols=5)
    t7.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t7)
    for r_idx, row in enumerate(t7_data):
        for c_idx, val in enumerate(row):
            cell = t7.cell(r_idx, c_idx)
            cell.text = val
            set_cell_margins(cell, top=50, bottom=50, left=80, right=80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 or c_idx == 4 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.runs[0]
            run.font.name = "Times New Roman"
            run.font.size = Pt(8.5)
            if r_idx == 0:
                run.font.bold = True
                set_cell_shading(cell, "EAEAEA")

    add_h2("5.5 Class-Wise Classification Mechanics and Confusion Analysis")
    add_body(
        "Analysis of the independent test confusion matrix (Table 9, Fig. 5) and per-class metrics (Table 8, Fig. 6) reveals distinct agronomic error patterns:\n"
        "1. Major Cereals and Cash Crops: Paddy (F1: 0.8672, Recall: 88.85%), Groundnut (F1: 0.8543, Recall: 87.55%), and Maize (F1: 0.8390, Precision: 90.16%) "
        "achieved high accuracy, benefiting from distinctive flood-moisture signatures and rapid vegetative vigor. Cotton achieved an F1 of 0.7817 (Recall: 81.57%).\n"
        "2. Chillies: Chillies achieved an F1 of 0.7606 (Precision: 74.09%, Recall: 78.14%), though 40 Maize parcels and 28 Cotton parcels were misclassified as Chillies, "
        "likely reflecting overlapping commercial irrigation practices in command areas.\n"
        "3. Bengal Gram: Achieved an F1 of 0.7273 (Recall: 78.26%, 36/46 correct). As a winter Rabi crop, its phenological cycle diverges cleanly from summer monsoon crops.\n"
        "4. Red Gram Confusion Mechanics: Red Gram exhibited the lowest recall across the test cohort at 20.00% (7/35 correct), resulting in an F1-score of 0.2857. "
        "Of the 35 evaluated Red Gram parcels, 21 (60.0%) were misclassified as Cotton, 3 as Bengal Gram, 2 as Chillies, and 1 each as Maize and Paddy. This confusion "
        "is agronomically plausible: in Telangana, Red Gram (Cajanus cajan) and Cotton (Gossypium hirsutum) share overlapping sowing calendars (June/July) and broadleaf, "
        "woody canopy architectures with deep taproot structures. Furthermore, Red Gram is frequently intercropped with Cotton along parcel borders. Given the extremely "
        "limited test support (N = 35 evaluated parcels, representing 1.5% of the test cohort), minority pulse classification remains an open challenge."
    )

    # TABLE 8
    add_caption("Table 8. Verified Class-Wise Performance Metrics (Spatial CV vs Independent Test).")
    t8_data = [
        ["Crop Class", "CV Prec", "CV Rec", "CV F1", "Test Prec", "Test Rec", "Test F1", "Test Eval N", "Test Total N", "Cloud Loss (%)"],
        ["Bengal Gram", "0.7916", "0.8667", "0.8275", "0.6792", "0.7826", "0.7273", "46", "48", "4.17%"],
        ["Chillies", "0.5249", "0.5441", "0.5343", "0.7409", "0.7814", "0.7606", "311", "381", "18.37%"],
        ["Cotton", "0.7939", "0.8261", "0.8097", "0.7505", "0.8157", "0.7817", "472", "569", "17.05%"],
        ["Groundnut", "0.8597", "0.8698", "0.8647", "0.8340", "0.8755", "0.8543", "241", "266", "9.40%"],
        ["Maize", "0.7492", "0.6730", "0.7090", "0.9016", "0.7845", "0.8390", "631", "699", "9.73%"],
        ["Paddy", "0.8664", "0.8752", "0.8708", "0.8468", "0.8885", "0.8672", "529", "714", "25.91%"],
        ["Red Gram", "0.5467", "0.3710", "0.4420", "0.5000", "0.2000", "0.2857", "35", "41", "14.63%"],
        ["Macro Average", "0.7332", "0.7180", "0.7226", "0.7504", "0.7326", "0.7308", "2,265", "2,718", "16.67%"],
    ]
    t8 = doc.add_table(rows=len(t8_data), cols=10)
    t8.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t8)
    for r_idx, row in enumerate(t8_data):
        for c_idx, val in enumerate(row):
            cell = t8.cell(r_idx, c_idx)
            cell.text = val
            set_cell_margins(cell, top=50, bottom=50, left=50, right=50)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.runs[0]
            run.font.name = "Times New Roman"
            run.font.size = Pt(8)
            if r_idx == 0 or r_idx == len(t8_data) - 1:
                run.font.bold = True
                set_cell_shading(cell, "EAEAEA" if r_idx == 0 else "F8F8F8")

    # TABLE 9
    add_caption("Table 9. Independent Test Set Confusion Matrix (N=2,265).")
    t9_data = [
        ["True Crop \\ Pred", "Bengal Gram", "Chillies", "Cotton", "Groundnut", "Maize", "Paddy", "Red Gram", "True Total"],
        ["Bengal Gram", "36", "1", "4", "0", "2", "1", "2", "46"],
        ["Chillies", "0", "243", "40", "5", "12", "11", "0", "311"],
        ["Cotton", "7", "14", "385", "20", "15", "28", "3", "472"],
        ["Groundnut", "0", "9", "11", "211", "6", "4", "0", "241"],
        ["Maize", "6", "41", "36", "12", "495", "40", "1", "631"],
        ["Paddy", "1", "18", "16", "5", "18", "470", "1", "529"],
        ["Red Gram", "3", "2", "21", "0", "1", "1", "7", "35"],
        ["Pred Total", "53", "328", "513", "253", "549", "555", "14", "2,265"],
    ]
    t9 = doc.add_table(rows=len(t9_data), cols=9)
    t9.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t9)
    for r_idx, row in enumerate(t9_data):
        for c_idx, val in enumerate(row):
            cell = t9.cell(r_idx, c_idx)
            cell.text = val
            set_cell_margins(cell, top=50, bottom=50, left=50, right=50)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.runs[0]
            run.font.name = "Times New Roman"
            run.font.size = Pt(8)
            if r_idx == 0 or r_idx == len(t9_data) - 1 or c_idx == len(row) - 1 or (r_idx == c_idx and r_idx > 0):
                run.font.bold = True
                if r_idx == 0 or r_idx == len(t9_data) - 1:
                    set_cell_shading(cell, "EAEAEA" if r_idx == 0 else "F8F8F8")
                elif r_idx == c_idx:
                    set_cell_shading(cell, "E0EAF5")

    add_figure_img(
        PROJECT_ROOT / "outputs" / "reports" / "paper_materials" / "figures" / "fig1_confusion_matrices.png",
        "Fig. 5. Normalized confusion matrices under 5-fold spatial cross-validation (left) and independent out-of-Mandal test evaluation (right).",
        width_in=4.6
    )

    add_figure_img(
        PROJECT_ROOT / "outputs" / "reports" / "paper_materials" / "figures" / "fig2_class_wise_performance.png",
        "Fig. 6. Class-wise precision, recall, and F1-score across seven crop types on the independent test set.",
        width_in=4.4
    )

    add_h2("5.6 Scientific Interpretation of Cross-Validation vs Test Scores")
    add_body(
        "The independent test scores (Macro-F1 0.7308, Accuracy 81.55%) are numerically higher than the 5-fold spatial cross-validation baseline "
        "(Macro-F1 0.7226, Accuracy 78.81%). However, this must not be interpreted as evidence of superior geographic generalization. Rather, "
        "this divergence is explained by two structural factors:\n"
        "1. Class Distribution Shift: The test set has a significantly higher proportion of readily classifiable crops. Maize constitutes 27.86% "
        "of evaluated test parcels compared to 17.54% of training parcels. Conversely, Bengal Gram drops from 11.96% in training to 2.03% in testing. "
        "Because Maize achieves high precision (90.16%), this distribution shift inflates overall accuracy and weighted metrics.\n"
        "2. Cloud-Related Selection Bias: 453 test parcels (16.67%) were excluded due to cloud contamination. Importantly, this exclusion was "
        "non-uniform: Kharif Paddy experienced a 25.91% exclusion rate (185/714 parcels lost), whereas Bengal Gram experienced only a 4.17% loss "
        "(Table 2). Consequently, the evaluated test subset represents a partially filtered sub-population rather than a uniform random sample of Telangana agriculture."
    )

    # SECTION 6
    add_h1("6 Limitations and Operational Constraints")
    add_body(
        "To preserve scientific rigor, we document five primary limitations:\n"
        "1. Cloud-Driven Attrition: The operational reliance on optical Sentinel-2 imagery resulted in the complete exclusion of 16.67% of test parcels "
        "due to persistent monsoon cloud obscuration. The framework currently lacks an all-weather synthetic aperture radar (SAR) fallback.\n"
        "2. Minority Class Fragility: Despite class weighting, Red Gram achieved only 20% recall on the test set. Findings regarding minority pulse classes "
        "carry wide confidence intervals due to small test support (N = 35).\n"
        "3. Single Agricultural Season: All satellite imagery and ground-truth data correspond to the 2023 cropping year. Inter-annual generalization "
        "across drought or extreme rainfall years remains untested.\n"
        "4. Geographic Scope and Spatial Autocorrelation: The evaluation is an independent intra-state administrative holdout across 35 unseen Mandals within "
        "Telangana, with zero Mandal overlap between training and test sets. Both cohorts originate from Telangana, and ten broader administrative "
        "districts overlap between cohorts. Crucially, disjoint Mandals provide administrative separation but do not establish geographic independence "
        "or eliminate spatial autocorrelation; adjacent Mandals frequently share regional climate, soil series, and canal access. Furthermore, this "
        "experiment does not demonstrate out-of-region transfer, cross-state generalization, contiguous spatial-block validation, or multi-year stability, "
        "and empirical performance across other agro-climatic zones requires dedicated external evaluation.\n"
        "5. Downstream Unimplemented Modules: Crop disease diagnosis (category), meteorological integration (ERA5), and FAO-56 evapotranspiration "
        "irrigation scheduling remain conceptual future work and are not validated by this study."
    )

    # SECTION 7
    add_h1("7 Conclusion and Future Directions")
    add_body(
        "This study has established a reproducible, leak-free benchmark for seven-class crop identification in smallholder semi-arid agriculture using "
        "multi-temporal Sentinel-2 imagery. By enforcing strict spatial blocking across 116 training Mandals and evaluating on 35 unseen test Mandals "
        "within Telangana, we demonstrated that a 140-dimensional Spectral-Temporal XGBoost model achieves robust operational performance (0.7226 CV "
        "Macro-F1; 0.7308 Independent Test Macro-F1; 81.55% Test Accuracy on the 2,265 evaluated parcels). We empirically showed that spatial texture "
        "features provide no statistically supported global benefit, and that fold-internal inverse-frequency weighting recovers minority pulse "
        "decision boundaries without spatial leakage.\n"
        "Future research will focus on multimodal SAR-optical fusion with Sentinel-1 to recover monsoon-obscured parcels, crop health status "
        "diagnosis using preserved category labels, and daily agrometeorological irrigation scheduling via FAO-56 Penman-Monteith coupling."
    )

    # ACKNOWLEDGMENTS (Anonymized)
    add_h1("Acknowledgments")
    add_body(
        "[Acknowledgments and institutional funding disclosures temporarily omitted for double-blind peer review. Ground-truth curation by agricultural "
        "authorities and European Space Agency Copernicus observations will be fully acknowledged in the final camera-ready version upon acceptance.]"
    )

    # REFERENCES
    add_h1("References")
    references = [
        "[1] Defourny, P., Bontemps, S., Bellemans, N., et al.: Near real-time agriculture monitoring at national scale at parcel resolution: Performance assessment of the Sen2-Agri automated system in various cropping systems around the world. Remote Sens. Environ. 221, 551–568 (2019). https://doi.org/10.1016/j.rse.2018.11.029",
        "[2] Jin, Z., Azzari, G., You, C., et al.: Smallholder maize area and yield mapping at national scales with Google Earth Engine. Remote Sens. Environ. 228, 115–128 (2019). https://doi.org/10.1016/j.rse.2019.04.016",
        "[3] Griffiths, P., Nendel, C., Hostert, P.: Intra-annual reflectance composites from Sentinel-2 and Landsat for national-scale crop and land cover mapping. Remote Sens. Environ. 220, 135–151 (2019). https://doi.org/10.1016/j.rse.2018.10.031",
        "[4] Roberts, D.R., Bahn, V., Ciuti, S., et al.: Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography 40(8), 913–929 (2017). https://doi.org/10.1111/ecog.02881",
        "[5] Meyer, H., Reudenbach, C., Hengl, T., et al.: Improving performance of spatio-temporal machine learning models using forward feature selection and target-oriented validation. Environ. Model. Softw. 101, 1–9 (2018). https://doi.org/10.1016/j.envsoft.2017.12.001",
        "[6] Vuolo, F., Neuwirth, M., Immitzer, M., et al.: How much does multi-temporal Sentinel-2 data improve crop type classification? Int. J. Appl. Earth Obs. Geoinf. 72, 122–130 (2018). https://doi.org/10.1016/j.jag.2018.06.007",
        "[7] Lebourgeois, V., Bégué, A., Labbé, S., et al.: A combined Random Forest and OBIA classification scheme for mapping smallholder agriculture at different nomenclature levels using multisource data. Remote Sens. 9(3), 259 (2017). https://doi.org/10.3390/rs9030259",
        "[8] Inglada, J., Arias, M., Tardy, B., et al.: Assessment of an operational system for crop type map production using high temporal and spatial resolution satellite optical imagery. Remote Sens. 7(9), 12356–12379 (2015). https://doi.org/10.3390/rs70912356",
        "[9] Maxwell, A.E., Warner, T.A., Fang, F.: Implementation of machine-learning classification in remote sensing: An applied review. Int. J. Remote Sens. 39(9), 2784–2817 (2018). https://doi.org/10.1080/01431161.2018.1433343",
        "[10] Chen, T., Guestrin, C.: XGBoost: A scalable tree boosting system. In: Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD '16), pp. 785–794. ACM (2016). https://doi.org/10.1145/2939672.2939785",
        "[11] Gumma, M.K., Thenkabail, P.S., Teluguntla, P., et al.: Crop type identification and spatial mapping using Sentinel-2 satellite data with focus on field-level information. Geocarto Int. 37(13), 3824–3841 (2020). https://doi.org/10.1080/10106049.2020.1864026",
        "[12] Gumma, M.K., Thenkabail, P.S., Panjala, P., et al.: Multiple agricultural cropland products of South Asia developed using Landsat-8 30 m and MODIS 250 m data using machine learning on the Google Earth Engine (GEE) cloud. GISci. Remote Sens. 59(1), 1048–1077 (2022). https://doi.org/10.1080/15481603.2022.2088651",
        "[13] Haralick, R.M., Shanmugam, K., Dinstein, I.: Textural features for image classification. IEEE Trans. Syst. Man Cybern. SMC-3(6), 610–621 (1973). https://doi.org/10.1109/TSMC.1973.4309314",
    ]
    for ref in references:
        p_ref = doc.add_paragraph()
        p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_ref.paragraph_format.left_indent = Inches(0.25)
        p_ref.paragraph_format.first_line_indent = Inches(-0.25)
        p_ref.paragraph_format.space_before = Pt(0)
        p_ref.paragraph_format.space_after = Pt(3)
        p_ref.paragraph_format.line_spacing = 1.05
        run = p_ref.add_run(ref)
        run.font.name = "Times New Roman"
        run.font.size = Pt(8.5)

    doc.save(str(DOCX_PATH))
    print(f"Word document saved to {DOCX_PATH}")

def export_pdf():
    print(f"Exporting PDF via Word COM automation: {PDF_PATH}...")
    import win32com.client
    word = win32com.client.Dispatch('Word.Application')
    word.Visible = False
    try:
        abs_docx = str(DOCX_PATH.resolve())
        abs_pdf = str(PDF_PATH.resolve())
        wdoc = word.Documents.Open(abs_docx)
        # wdFormatPDF = 17
        wdoc.SaveAs2(abs_pdf, FileFormat=17)
        page_count = wdoc.ComputeStatistics(2) # wdStatisticPages = 2
        wdoc.Close()
        print(f"PDF exported successfully! Page count: {page_count}")
        return page_count
    finally:
        word.Quit()

def build_latex():
    print(f"Building Springer LNCS LaTeX manuscript: {TEX_PATH}...")
    tex_content = r"""\documentclass[runningheads]{llncs}

\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{hyperref}
\usepackage{cite}
\usepackage{microtype}
\usepackage{array}
\usepackage{multirow}

\begin{document}

\title{Multi-Temporal Sentinel-2 Spectral-Temporal Features for Seven-Class Crop Classification in Telangana: Spatial Cross-Validation and Independent Test Evaluation\thanks{Working Title for Framework Context: \emph{AI-Based Remote Sensing Framework for Crop Monitoring and Irrigation Decision Support Using Multi-Temporal Sentinel-2 Satellite Imagery}}}

\titlerunning{Multi-Temporal Sentinel-2 Seven-Class Crop Classification in Telangana}

% Anonymized for Double-Blind Review
\author{Anonymous Authors for Double-Blind Peer Review}
\authorrunning{Anonymous et al.}
\institute{Paper submitted to ICIDA 2026 for Double-Blind Review\\
\email{author-email-placeholder@conference.org}}

\maketitle

\begin{abstract}
Accurate, parcel-level crop-type classification in smallholder agricultural landscapes is fundamental for regional food security, yield forecasting, and future agricultural water management. However, in semi-arid tropical regions such as Telangana, India, smallholder agricultural mapping is severely challenged by parcel fragmentation (median area $\sim 0.28$\,ha), optical cloud contamination during the monsoon season, extreme class imbalance, and performance over-optimism induced by spatial autocorrelation under conventional random train/test splits. In this study, we present an end-to-end, reproducible remote sensing and machine learning methodology that classifies seven major agricultural crops (\textbf{Paddy, Cotton, Maize, Bengal Gram, Groundnut, Chillies, and Red Gram}) using multi-temporal Sentinel-2 surface reflectance imagery. We engineer a 140-dimensional Spectral-Temporal ($S+T$) feature space combining full-season aggregations, phenological window compositing across early-season vegetative establishment, mid-season reproductive peak, and late-season senescence, alongside four dynamic transition trajectory deltas. To ensure authentic validation, we enforce strict 5-fold spatial cross-validation grouped by Sub-District (116 Mandals, zero Mandal group overlap) on $N = 7{,}843$ validated farm parcels, coupled with fold-internal inverse-frequency sample weighting. Our primary regularized XGBoost model achieves a pooled out-of-fold Macro-F1 score of \textbf{0.7226}, Balanced Accuracy of \textbf{0.7180}, and Overall Accuracy of \textbf{0.7881 (78.81\%)} across cross-validation. Fold-internal weighting rescues minority Red Gram F1-score from 0.3683 to 0.4420 (+7.37 percentage points). Furthermore, incorporating 20 Gray-Level Co-occurrence Matrix (GLCM) spatial texture features provided no statistically supported global improvement ($p = 0.8600$, 95\% CI $[-0.0066, +0.0092]$), while parcel-area stratification revealed a performance reduction on the smallest parcel quartile ($<1{,}257\,\text{m}^2$, $\Delta = -0.0133$), consistent with the physical hypothesis of boundary pixel contamination. We further evaluate the frozen champion model on an independent intra-state administrative holdout of $N = 2{,}718$ surveyed farm parcels spanning 35 unseen Mandals within Telangana (zero Mandal overlap with training). Across the $N = 2{,}265$ parcels (83.33\% of the surveyed cohort) where satellite features were successfully extracted across all four temporal windows, the frozen model achieves a Macro-F1 of \textbf{0.7308}, Balanced Accuracy of \textbf{0.7326}, and Overall Accuracy of \textbf{0.8155 (81.55\%)}, with 1,847 correct and 418 incorrect predictions. We document that persistent cloud cover caused the exclusion of 453 test parcels (16.67\%), with exclusion rates varying from 4.17\% on dry-season pulses to 25.91\% on monsoon paddy. The metric parity between cross-validation and independent test evaluation confirms the stability of the 140-feature representation across unseen administrative units within the state, while highlighting the necessity of accounting for cloud-driven attrition in operational agricultural monitoring.

\keywords{Crop Classification \and Sentinel-2 \and Multi-Temporal Remote Sensing \and Spatial Cross-Validation \and Smallholder Agriculture \and XGBoost \and Class Imbalance \and Telangana.}
\end{abstract}

\section{Introduction}
Accurate, timely, and spatially explicit crop-type maps are essential for agricultural monitoring, subsidy allocation, agricultural insurance verification, and regional water resource stewardship \cite{defourny2019,griffiths2019}. In semi-arid agrarian economies such as the state of Telangana in southern India, crop production is highly dynamic, relying on a complex mosaic of rainfed cultivation, minor tank storage, and expanding lift-irrigation networks \cite{gumma2020}. Farmers cultivate a diverse portfolio of summer and winter crops across distinct seasons (\emph{Kharif} and \emph{Rabi}), including major cereals (Paddy, Maize), industrial fibers (Cotton), high-value spices (Chillies), and protein-rich pulses and oilseeds (Groundnut, Bengal Gram, Red Gram). However, conventional agricultural statistics in the region rely heavily on manual field-by-field revenue inspections (\emph{patwari} surveys), which are labor-intensive, logistically demanding, prone to reporting latency, and aggregated to coarse administrative levels.

Satellite remote sensing provides an objective, scalable alternative for parcel-level crop mapping. The European Space Agency's (ESA) Copernicus Sentinel-2 constellation offers multispectral imagery at 10\,m and 20\,m spatial resolutions with a 5-day nominal revisit frequency, establishing an unprecedented capability for agricultural monitoring \cite{defourny2019,inglada2015}. Nevertheless, operational crop classification in smallholder environments faces four major scientific and practical hurdles:
\begin{enumerate}
    \item \textbf{Spectral Ambiguity in Fragmented Landscapes:} Smallholder agricultural holdings in Telangana are intensely fragmented, with a median parcel area of approximately 0.28\,ha ($2{,}779\,\text{m}^2$). At this scale, individual farm plots span only a few 10-meter pixels. Multiple co-occurring crops exhibit near-identical spectral reflectance during peak vegetative growth, making single-date optical acquisitions insufficient for reliable discrimination \cite{jin2019,vuolo2018}.
    \item \textbf{Spatial Autocorrelation and Evaluation Bias:} A critical flaw in much of the applied remote-sensing machine learning literature is the reliance on naive random $k$-fold cross-validation or unstratified random train/test splits \cite{meyer2018,roberts2017}. Because neighboring farm fields share identical soil types, micro-climates, irrigation canal access, and planting dates, spatial autocorrelation induces severe data leakage between training and evaluation subsets \cite{roberts2017}.
    \item \textbf{Severe Class Imbalance and Minority Erasure:} Agrarian land use is heavily skewed toward dominant cash crops such as Paddy and Cotton. High-value or subsistence pulses, such as Red Gram (\emph{Cajanus cajan}), frequently represent less than 3\% of regional farm parcels \cite{maxwell2018}.
    \item \textbf{Cloud-Driven Data Missingness:} The \emph{Kharif} cropping cycle coincides with the Southwest monsoon, generating dense, persistent cloud and cloud-shadow cover across optical acquisitions, resulting in parcel-level extraction failures.
\end{enumerate}

\subsection{Scope Demarcation}
To maintain strict scientific integrity, we explicitly define the implemented scope of this study. While the broader conceptual framework envisions an integrated pipeline coupling crop identification with disease diagnosis, crop water stress retrieval, and FAO-56 Penman-Monteith irrigation decision support, \textbf{the experimentally evaluated and validated contribution of this paper is seven-class crop-type classification}. Downstream secondary tasks represent planned future extensions and are not presented as validated operational systems.

\subsection{Primary Contributions}
\begin{enumerate}
    \item \textbf{Multi-Temporal Phenological Feature Engineering:} We design a 140-dimensional Spectral-Temporal ($S+T$) feature space combining full-season statistics, three phenological window aggregations, and four transition trajectory deltas, demonstrating an +8.95 percentage point Macro-F1 gain over the static full-season baseline ($0.6174 \to 0.7069$) from multi-temporal windowing, rising to 0.7226 (+10.52 pp total) with fold-internal class weighting.
    \item \textbf{Strict Spatial Group Cross-Validation:} We enforce 5-fold \texttt{StratifiedGroupKFold} validation partitioned strictly by Sub-District (116 Mandals, zero Mandal overlap), proving that regularized XGBoost achieves a robust pooled Macro-F1 of 0.7226 and Overall Accuracy of 78.81\% across 7,843 parcels without village co-location leakage.
    \item \textbf{Evaluation of Spatial Texture in Smallholder Parcels:} We demonstrate that incorporating 20 GLCM spatial texture features yields no statistically supported overall gain ($p = 0.8600$, 95\% CI $[-0.0066, +0.0092]$) and is associated with reduced performance in the smallest parcel quartile ($<1{,}257\,\text{m}^2$, $\Delta = -0.0133$), providing an empirical basis for selecting the parsimonious 140-feature representation.
    \item \textbf{Leak-Free Minority Pulse Recovery:} We implement fold-internal inverse-frequency sample weighting, boosting Red Gram F1-score from 0.3683 to 0.4420 (+7.37 pp) while preserving strict spatial validation boundaries.
    \item \textbf{Independent Intra-State Administrative Holdout with Cloud Accounting:} We execute frozen-model inference on an independent intra-state administrative holdout of 2,718 surveyed parcels across 35 unseen Mandals in Telangana (zero Mandal overlap with training), establishing that the model achieves 0.7308 Macro-F1 and 81.55\% accuracy on the 2,265 successfully extracted parcels while transparently documenting a 16.67\% cloud-exclusion rate.
\end{enumerate}

\section{Related Work}
\subsection{Multi-Temporal Satellite Optical Mapping}
Early automated crop mapping relied heavily on Landsat or MODIS imagery \cite{inglada2015,gumma2022}. The advent of Sentinel-2 MSI transformed smallholder mapping by providing 10-meter visible and near-infrared bands alongside red-edge spectral bands \cite{defourny2019}. Vuolo et al. \cite{vuolo2018} demonstrated that multi-temporal Sentinel-2 acquisitions increased classification accuracy by 5\% to 17\% compared to single-date imagery. Similarly, Griffiths et al. \cite{griffiths2019} proved that intra-annual phenological compositing metrics effectively separate spectrally overlapping crops. In India, Gumma et al. \cite{gumma2020} demonstrated the utility of Sentinel-2 NDVI time series for mapping field-level pulse and cereal crops in semi-arid environments, while Gumma et al. \cite{gumma2022} established cloud-based machine learning workflows on Google Earth Engine for multi-cropland products across South Asia.

\subsection{Spatial Cross-Validation and Spatial Autocorrelation}
The necessity of spatial blocking in geospatial machine learning has been established by Roberts et al. \cite{roberts2017} and Meyer et al. \cite{meyer2018}. Spatial autocorrelation violates the fundamental assumption that training and validation instances are independently and identically distributed. Roberts et al. \cite{roberts2017} documented that non-spatial cross-validation can underestimate true prediction error by 20\% to 60\% in spatially structured datasets.

\subsection{Class Imbalance and Cost-Sensitive Learning}
As noted by Maxwell et al. \cite{maxwell2018}, standard loss functions in gradient-boosted decision trees optimize global error rates at the expense of minority classes. Inverse-frequency sample weighting adjusts the gradient contribution of each class according to its empirical rarity, providing a principled mechanism to recover minority decision boundaries without distorting physical feature distributions \cite{chen2016,maxwell2018}.

\subsection{Smallholder Parcel Fragmentation and Spatial Texture}
In smallholder systems, individual field extents are often smaller than 0.5\,ha \cite{jin2019}. While spatial texture metrics, such as Haralick's Gray-Level Co-occurrence Matrix (GLCM; \cite{haralick1973}), have demonstrated value in homogeneous commercial farms \cite{vuolo2018}, Lebourgeois et al. \cite{lebourgeois2017} highlighted that plots under 0.25\,ha suffer from severe boundary contamination.

\section{Study Area and Dataset}
\subsection{Study Area}
The study was conducted across the state of Telangana in southern India (Fig.~\ref{fig:spatial_overview}), covering an area of approximately $112{,}077\,\text{km}^2$. The region experiences a semi-arid tropical climate characterized by three distinct seasons: the Southwest monsoon (\emph{Kharif}, June to October), the dry winter (\emph{Rabi}, November to March), and a hot summer (March to May). Annual rainfall ranges between 700\,mm and 1,200\,mm.

\begin{figure}[t]
\centering
\includegraphics[width=0.75\textwidth]{outputs/figures/telangana_spatial_overview.png}
\caption{Spatial distribution of training (116 Mandals) and independent test (35 Mandals) agricultural parcels across Telangana, India.}
\label{fig:spatial_overview}
\end{figure}

\subsection{Ground-Truth Parcel Cohorts}
The empirical ground-truth dataset consists of verified vector polygons derived from extensive field surveys conducted across Telangana:
\begin{enumerate}
    \item \textbf{Training Cohort (\texttt{Train.csv}):} 7,888 initial surveyed parcel polygons across 14 districts and 116 Mandals, yielding an authoritative cohort of $N = 7{,}843$ eligible farm parcels following topological and cloud validation.
    \item \textbf{Independent Test Cohort (\texttt{Test.csv}):} $N = 2{,}718$ farm parcels distributed across 35 Mandals in 12 districts within Telangana. A preflight spatial audit confirmed zero Mandal overlap with training, establishing an independent intra-state administrative holdout across unseen Mandals within Telangana (Table~\ref{tab:cohort}). Ten broader districts overlap between cohorts; disjoint Mandals provide administrative separation but do not guarantee geographic independence or zero spatial autocorrelation.
    \item \textbf{Target Nomenclature:} Seven crop classes: \textbf{Bengal Gram, Chillies, Cotton, Groundnut, Maize, Paddy, and Red Gram}.
\end{enumerate}

\begin{table}[t]
\centering
\caption{Cohort Composition and Spatial Partitioning.}
\label{tab:cohort}
\small
\begin{tabular}{lcc}
\toprule
\textbf{Cohort Property} & \textbf{Training Cohort} & \textbf{Independent Test Cohort} \\
\midrule
Initial Surveyed Parcels & 7,888 & 2,718 \\
Persistent Cloud / Shadow Failures & 45 (0.57\%) & 453 (16.67\%) \\
Eligible Evaluated Parcels & \textbf{7,843 (99.43\%)} & \textbf{2,265 (83.33\%)} \\
Administrative Units (Mandals) & 116 & 35 \\
Districts Covered & 14 & 12 (10 districts overlap) \\
Inter-Cohort Mandal Overlap & — & \textbf{0 Mandals (Disjoint Holdout)} \\
Validation Design & 5-Fold \texttt{StratifiedGroupKFold} & Frozen Out-of-Mandal Inference \\
Target Variable & \texttt{Crop} (7 Classes) & \texttt{Crop} (7 Classes) \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Parcel-Level Feature Extraction Coverage \& Cloud Exclusion Accounting.}
\label{tab:cloud}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Crop Class} & \textbf{Total Test} & \textbf{Evaluated} & \textbf{Excluded} & \textbf{Coverage (\%)} & \textbf{Loss (\%)} \\
\midrule
Bengal Gram & 48 & 46 & 2 & 95.83\% & 4.17\% \\
Groundnut & 266 & 241 & 25 & 90.60\% & 9.40\% \\
Maize & 699 & 631 & 68 & 90.27\% & 9.73\% \\
Red Gram & 41 & 35 & 6 & 85.37\% & 14.63\% \\
Cotton & 569 & 472 & 97 & 82.95\% & 17.05\% \\
Chillies & 381 & 311 & 70 & 81.63\% & 18.37\% \\
Paddy & 714 & 529 & 185 & 74.09\% & \textbf{25.91\%} \\
\midrule
\textbf{Total Cohort} & \textbf{2,718} & \textbf{2,265} & \textbf{453} & \textbf{83.33\%} & \textbf{16.67\%} \\
\bottomrule
\end{tabular}
\end{table}

\section{Proposed Methodology}
\subsection{Satellite Preprocessing and Cloud Masking}
Multi-temporal surface reflectance imagery was ingested from Copernicus Sentinel-2 Level-2A (\texttt{COPERNICUS/S2\_SR\_HARMONIZED}) via Google Earth Engine (GEE). Non-vegetated and compromised pixels were identified and masked using the Scene Classification Layer (SCL).

\subsection{Phenological Windowing and Feature Representation}
We establish four temporal extraction windows tailored to Telangana's calendar: Early Season (Jun 01--Jul 31), Mid Season (Aug 01--Sep 30), Late Season (Oct 01--Nov 30), and Full Season (Jun 01--Dec 31). Ten surface reflectance bands (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12) and 14 vegetation/water indices (NDVI, EVI, NDMI, NDWI, GNDVI, NDRE, MSAVI, CI$_\text{Green}$, CI$_\text{RedEdge}$, MSI, NDBI, SAVI, SIPI, NMDI) were extracted per window (34 features $\times 4 = 136$). Adding four phenological transition deltas ($\Delta_\text{growth}$, $\text{Ratio}_\text{peak/early}$, $\Delta_\text{senescence}$, $\Delta_\text{moisture}$) yields the exact \textbf{140-dimensional Spectral-Temporal ($S+T$) feature space} (Table~\ref{tab:features}).

\begin{table}[t]
\centering
\caption{140-Dimensional Spectral-Temporal Feature Representation.}
\label{tab:features}
\small
\begin{tabular}{lllcc}
\toprule
\textbf{Feature Group} & \textbf{Temporal Window} & \textbf{Description} & \textbf{Count} \\
\midrule
Spectral \& Biophysical ($S$) & Full Season (Jun 01--Dec 31) & 10 S2 bands + 14 spectral indices & 34 \\
Windowed Phenology ($W_1$) & Early Season (Jun 01--Jul 31) & Emergence / establishment & 34 \\
Windowed Phenology ($W_2$) & Mid Season (Aug 01--Sep 30) & Peak vegetative vigor / flowering & 34 \\
Windowed Phenology ($W_3$) & Late Season (Oct 01--Nov 30) & Maturation / senescence & 34 \\
Dynamic Deltas ($T$) & Multi-Window Trajectories & $\Delta_\text{growth}$, $\text{Ratio}$, $\Delta_\text{senesc}$, $\Delta_\text{moist}$ & 4 \\
\midrule
\textbf{Total Dimensionality} & — & \textbf{Spectral-Temporal Vector ($S+T$)} & \textbf{140} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Spatial Texture Features}
We constructed an augmented 160-dimensional representation ($S+T+P$) by extracting Gray-Level Co-occurrence Matrix (GLCM; \cite{haralick1973}) texture rasters across NIR (B8) at 10-meter resolution using a $3\times3$ sliding window (5 statistics $\times 4$ windows = 20 features).

\subsection{Model Architecture and Class Weighting}
We deploy XGBoost \cite{chen2016} with inverse-frequency sample weights $w_c$:
\begin{equation}
w_c = \frac{N_\text{train}}{C \cdot N_c}
\end{equation}
computed strictly inside each training fold to prevent validation leakage.

\subsection{Spatial Validation and Hyperparameter Protocol}
Validation is structured as a 5-fold \texttt{StratifiedGroupKFold} grouped by Mandal. Hyperparameters were evaluated via an exploratory bounded search of ten candidate configurations using 3-fold grouped inner cross-validation strictly on the training partition of Outer Fold 0 ($N = 6{,}335$ parcels across 93 Mandals; Outer Fold 0 validation partition has $N = 1{,}508$ parcels across 23 Mandals). Selection was based on inner Macro-F1 (0.6797). The selected configuration—\texttt{max\_depth=7}, \texttt{learning\_rate=0.05}, \texttt{n\_estimators=300}, \texttt{subsample=0.80}, \texttt{colsample\_bytree=0.75}, \texttt{min\_child\_weight=3}, \texttt{gamma=0.20}, \texttt{reg\_alpha=0.20}, and \texttt{reg\_lambda=5.0}—was frozen and evaluated across all 5 spatial folds and test inference (Table~\ref{tab:hyperparams}). This protocol is an exploratory parameter search on a single fold rather than fully nested cross-validation.

\begin{table}[t]
\centering
\caption{XGBoost Hyperparameter Specification and Tuning Protocol.}
\label{tab:hyperparams}
\small
\begin{tabular}{lccl}
\toprule
\textbf{Hyperparameter} & \textbf{Selected} & \textbf{Search Space} & \textbf{Tuning Protocol \& Context} \\
\midrule
\texttt{n\_estimators} & 300 & [100, 200, 300, 500] & Fixed during exploration \\
\texttt{max\_depth} & 7 & [5, 6, 7, 8] & Captures non-linear canopy interactions \\
\texttt{learning\_rate} ($\eta$) & 0.05 & [0.03, 0.05, 0.10] & Conservative shrinkage \\
\texttt{subsample} & 0.80 & [0.70, 0.80, 0.90] & Row sub-sampling per boosting iteration \\
\texttt{colsample\_bytree} & 0.75 & [0.60, 0.75, 0.80, 1.00] & Feature sub-sampling per tree split \\
\texttt{min\_child\_weight} & 3 & [1, 3, 5] & Prevents fitting smallholder clusters \\
\texttt{gamma} & 0.20 & [0.0, 0.1, 0.2, 0.3] & Minimum loss reduction to split leaf \\
\texttt{reg\_lambda} ($L_2$) & 5.0 & [1.0, 2.0, 3.0, 5.0, 10.0] & L2 regularization against spatial noise \\
\texttt{reg\_alpha} ($L_1$) & 0.20 & [0.0, 0.1, 0.2, 0.5, 1.0] & Mild sparsity enforcement \\
\texttt{tree\_method} & \texttt{hist} & [\texttt{hist}] & Fast exact histogram binning \\
Class Weighting & Inverse-Freq & [None, Bal, Inv-Freq] & $w_c = N / (C \cdot N_c)$; fold-internal \\
\bottomrule
\end{tabular}
\end{table}

\section{Results and Discussion}
\subsection{Spatial Cross-Validation Benchmark}
Across $N = 7{,}843$ parcels in 116 Mandals, the primary \texttt{Weighted S+T} model achieves pooled Macro-F1 of \textbf{0.7226}, Balanced Accuracy of \textbf{0.7180}, and Overall Accuracy of \textbf{0.7881 (78.81\%)} (Table~\ref{tab:cv}). Fold-mean Macro-F1 is $0.7117 \pm 0.0181$ (population SD) or $\pm 0.0202$ (sample SD) (Fig.~\ref{fig:fold_var}).

\begin{table}[t]
\centering
\caption{5-Fold Spatial Cross-Validation Performance (\texttt{Weighted S+T}, $N=7,843$).}
\label{tab:cv}
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Fold ID} & \textbf{Val Mandals} & \textbf{Val Parcels} & \textbf{Macro-F1} & \textbf{Accuracy} & \textbf{Bal-Acc} & \textbf{W-F1} \\
\midrule
Fold 0 & 23 & 1,508 & 0.7269 & 0.7960 & 0.7203 & 0.7932 \\
Fold 1 & 21 & 1,600 & 0.7381 & 0.8013 & 0.7329 & 0.7997 \\
Fold 2 & 22 & 1,595 & 0.6919 & 0.7717 & 0.6865 & 0.7681 \\
Fold 3 & 26 & 1,530 & 0.6945 & 0.7749 & 0.6911 & 0.7728 \\
Fold 4 & 24 & 1,610 & 0.7073 & 0.7966 & 0.7102 & 0.7946 \\
\midrule
\textbf{Fold Mean} & 23.2 & 1,568.6 & \textbf{0.7117} & \textbf{0.7881} & \textbf{0.7082} & \textbf{0.7857} \\
\textbf{Pooled} & \textbf{116} & \textbf{7,843} & \textbf{0.7226} & \textbf{0.7881} & \textbf{0.7180} & \textbf{0.7857} \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=0.7\textwidth]{outputs/reports/paper_materials/figures/fig3_fold_variability.png}
\caption{Performance metric dispersion across 5 outer spatial cross-validation folds.}
\label{fig:fold_var}
\end{figure}

\subsection{Feature Ablation and Systematic Evolution}
Multi-temporal windowing (140 S+T, unweighted) propelled Macro-F1 from 0.6174 (static full-season) to 0.7069 (+8.95 pp gain; Table~\ref{tab:ablation}, Fig.~\ref{fig:ablation}). Fold-internal sample weighting further raised Macro-F1 to 0.7226 (+1.57 pp), rescuing Red Gram F1 from 0.3683 to 0.4420 (+7.37 pp).

\begin{table}[t]
\centering
\caption{Model Evolution, Feature Ablations, and Significance Testing.}
\label{tab:ablation}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Model / Stage} & \textbf{Feats} & \textbf{Cohort} & \textbf{Weighting} & \textbf{Macro-F1} & \textbf{Accuracy} \\
\midrule
Early Single Window & 34 & 7,843 & Unweighted & 0.5058 & 0.6358 \\
Full Single Window & 34 & 7,843 & Unweighted & 0.6174 & 0.7208 \\
Unweighted S+T & 140 & 7,843 & Unweighted & 0.7069 & 0.7924 \\
Unweighted S+T+P & 160 & 7,843 & Unweighted & 0.7038 & 0.7900 \\
\textbf{Final Weighted S+T} & \textbf{140} & \textbf{7,843} & \textbf{Inverse-Freq} & \textbf{0.7226} & \textbf{0.7881} \\
Weighted S+T+P & 160 & 7,843 & Inverse-Freq & 0.7228 & 0.7905 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=0.7\textwidth]{outputs/reports/paper_materials/figures/fig5_pipeline_evolution.png}
\caption{Stepwise feature ablation progression from single-window baseline to champion weighted S+T pipeline.}
\label{fig:ablation}
\end{figure}

\subsection{Spatial Texture Evaluation and Parcel-Size Stratification}
The 160-dimensional \texttt{Weighted S+T+P} model achieved Macro-F1 of 0.7228, virtually identical to 140-feature \texttt{Weighted S+T} (0.7226, $\Delta = +0.0002$; cluster bootstrap $p = 0.8600$, paired $t$-test $p = 0.7765$). Quartile stratification (Table~\ref{tab:quartiles}, Fig.~\ref{fig:quartiles}) revealed a descriptive drop in Q1 ($<1{,}257\,\text{m}^2$, $\Delta = -0.0133$), consistent with boundary pixel contamination from $3\times3$ ($900\,\text{m}^2$) convolution kernels. Parsimony justifies selecting the 140-feature model.

\begin{table}[t]
\centering
\caption{Parcel Area Quartile Stratification: GLCM Texture vs Spectral-Temporal.}
\label{tab:quartiles}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Quartile} & \textbf{Area Range ($\text{m}^2$)} & \textbf{Parcels} & \textbf{S+T F1} & \textbf{S+T+P F1} & \textbf{Delta} \\
\midrule
Q1 & 54 – 1,257 & 1,961 & 0.6598 & 0.6465 & -0.0133 (-1.33 pp) \\
Q2 & 1,257 – 2,783 & 1,961 & 0.7332 & 0.7184 & -0.0148 (-1.48 pp) \\
Q3 & 2,784 – 5,517 & 1,960 & 0.7400 & 0.7530 & +0.0130 (+1.30 pp) \\
Q4 & 5,528 – 69,491 & 1,961 & 0.7253 & 0.7219 & -0.0034 (-0.34 pp) \\
\midrule
\textbf{Full} & \textbf{54 – 69,491} & \textbf{7,843} & \textbf{0.7226} & \textbf{0.7228} & \textbf{+0.0002 (+0.02 pp)} \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=0.7\textwidth]{outputs/reports/paper_materials/figures/fig4_parcel_size_quartiles.png}
\caption{Macro-F1 delta across parcel-area quartiles, illustrating boundary contamination vulnerability in smallholder plots.}
\label{fig:quartiles}
\end{figure}

\subsection{Independent Test Set Evaluation across 35 Unseen Mandals}
On the independent test cohort ($N = 2{,}718$), satellite features were successfully extracted for $N = 2{,}265$ parcels (83.33\%). The frozen model achieved Macro-F1 of \textbf{0.7308}, Overall Accuracy of \textbf{0.8155 (81.55\%)}, Balanced Accuracy of \textbf{0.7326}, and Weighted-F1 of \textbf{0.8137}, with 1,847 correct and 418 incorrect predictions (Table~\ref{tab:test}, Table~\ref{tab:classwise}, Table~\ref{tab:cm}, Fig.~\ref{fig:cm}, Fig.~\ref{fig:classwise}).

\begin{table}[t]
\centering
\caption{Independent Test Performance vs Spatial Cross-Validation Parity.}
\label{tab:test}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Metric} & \textbf{Spatial CV ($N=7,843$)} & \textbf{Test Set ($N=2,265$)} & \textbf{Delta} \\
\midrule
Macro-F1 & 0.7226 & \textbf{0.7308} & +0.0082 \\
Overall Accuracy & 0.7881 (78.81\%) & \textbf{0.8155 (81.55\%)} & +0.0274 \\
Balanced Accuracy & 0.7180 & \textbf{0.7326} & +0.0146 \\
Weighted-F1 & 0.7857 & \textbf{0.8137} & +0.0280 \\
Evaluated Sample Size & 7,843 & 2,265 & — \\
Excluded Cloud Loss & 45 (0.57\%) & 453 (16.67\%) & — \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Verified Class-Wise Performance Metrics (Spatial CV vs Independent Test).}
\label{tab:classwise}
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Crop Class} & \textbf{CV Prec} & \textbf{CV Rec} & \textbf{CV F1} & \textbf{Test Prec} & \textbf{Test Rec} & \textbf{Test F1} \\
\midrule
Bengal Gram & 0.7916 & 0.8667 & 0.8275 & 0.6792 & 0.7826 & 0.7273 \\
Chillies & 0.5249 & 0.5441 & 0.5343 & 0.7409 & 0.7814 & 0.7606 \\
Cotton & 0.7939 & 0.8261 & 0.8097 & 0.7505 & 0.8157 & 0.7817 \\
Groundnut & 0.8597 & 0.8698 & 0.8647 & 0.8340 & 0.8755 & 0.8543 \\
Maize & 0.7492 & 0.6730 & 0.7090 & 0.9016 & 0.7845 & 0.8390 \\
Paddy & 0.8664 & 0.8752 & 0.8708 & 0.8468 & 0.8885 & 0.8672 \\
Red Gram & 0.5467 & 0.3710 & \textbf{0.4420} & 0.5000 & 0.2000 & \textbf{0.2857} \\
\midrule
\textbf{Macro Average} & \textbf{0.7332} & \textbf{0.7180} & \textbf{0.7226} & \textbf{0.7504} & \textbf{0.7326} & \textbf{0.7308} \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Independent Test Set Confusion Matrix ($N=2,265$).}
\label{tab:cm}
\footnotesize
\begin{tabular}{lcccccccc}
\toprule
\textbf{True $\downarrow$ \textbackslash Pred $\rightarrow$} & \textbf{BG} & \textbf{CH} & \textbf{CO} & \textbf{GN} & \textbf{MZ} & \textbf{PD} & \textbf{RG} & \textbf{Total} \\
\midrule
Bengal Gram (BG) & \textbf{36} & 1 & 4 & 0 & 2 & 1 & 2 & 46 \\
Chillies (CH) & 0 & \textbf{243} & 40 & 5 & 12 & 11 & 0 & 311 \\
Cotton (CO) & 7 & 14 & \textbf{385} & 20 & 15 & 28 & 3 & 472 \\
Groundnut (GN) & 0 & 9 & 11 & \textbf{211} & 6 & 4 & 0 & 241 \\
Maize (MZ) & 6 & 41 & 36 & 12 & \textbf{495} & 40 & 1 & 631 \\
Paddy (PD) & 1 & 18 & 16 & 5 & 18 & \textbf{470} & 1 & 529 \\
Red Gram (RG) & 3 & 2 & 21 & 0 & 1 & 1 & \textbf{7} & 35 \\
\midrule
\textbf{Total Pred} & \textbf{53} & \textbf{328} & \textbf{513} & \textbf{253} & \textbf{549} & \textbf{555} & \textbf{14} & \textbf{2,265} \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=0.75\textwidth]{outputs/reports/paper_materials/figures/fig1_confusion_matrices.png}
\caption{Normalized confusion matrices under 5-fold spatial cross-validation (left) and independent out-of-Mandal test evaluation (right).}
\label{fig:cm}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.7\textwidth]{outputs/reports/paper_materials/figures/fig2_class_wise_performance.png}
\caption{Class-wise precision, recall, and F1-score across seven crop types on the independent test set.}
\label{fig:classwise}
\end{figure}

\subsection{Interpretation of CV vs Test Scores}
Numerical test gains (0.7308 vs 0.7226) reflect class distribution shifts (Maize rose from 17.54\% to 27.86\% of parcels) and cloud-driven selection bias (Paddy lost 25.91\% of parcels vs 4.17\% for Bengal Gram), rather than superior geographic generalization.

\section{Limitations and Operational Constraints}
\begin{enumerate}
    \item \textbf{Cloud-Driven Attrition:} 16.67\% of test parcels were excluded due to persistent monsoon cloud obscuration.
    \item \textbf{Minority Class Fragility:} Red Gram test recall was 20.00\% ($N=35$ evaluated), with 60\% confused with Cotton due to overlapping sowing calendars and woody broadleaf canopies.
    \item \textbf{Single Cropping Year:} Evaluation is based on 2023 acquisitions; multi-year climate variability remains untested.
    \item \textbf{Geographic Scope:} Independent intra-state administrative holdout across 35 unseen Mandals within Telangana; ten broader districts overlap. Does not establish cross-state transfer or zero spatial autocorrelation.
    \item \textbf{Downstream Unimplemented Modules:} Disease diagnosis, ERA5 reanalysis, and FAO-56 irrigation scheduling are unvalidated future directions.
\end{enumerate}

\section{Conclusion}
This study established a reproducible benchmark for seven-class smallholder crop classification using multi-temporal Sentinel-2 imagery. A 140-feature Spectral-Temporal XGBoost model achieves 0.7226 spatial CV Macro-F1 and 0.7308 independent test Macro-F1 (81.55\% accuracy on 2,265 evaluated parcels). GLCM texture features provided no statistically supported benefit, and fold-internal weighting recovered minority pulse boundaries. Future work will integrate Sentinel-1 SAR to recover cloud-obscured parcels.

\subsubsection*{Acknowledgments}
\emph{[Acknowledgments and funding information omitted for double-blind peer review. Will be restored in camera-ready version upon acceptance.]}

\begin{thebibliography}{13}
\bibitem{defourny2019} Defourny, P., et al.: Near real-time agriculture monitoring at national scale at parcel resolution. Remote Sens. Environ. 221, 551--568 (2019)
\bibitem{jin2019} Jin, Z., et al.: Smallholder maize area and yield mapping at national scales with Google Earth Engine. Remote Sens. Environ. 228, 115--128 (2019)
\bibitem{griffiths2019} Griffiths, P., Nendel, C., Hostert, P.: Intra-annual reflectance composites from Sentinel-2 and Landsat for national-scale crop and land cover mapping. Remote Sens. Environ. 220, 135--151 (2019)
\bibitem{roberts2017} Roberts, D.R., et al.: Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography 40(8), 913--929 (2017)
\bibitem{meyer2018} Meyer, H., et al.: Improving performance of spatio-temporal machine learning models using forward feature selection and target-oriented validation. Environ. Model. Softw. 101, 1--9 (2018)
\bibitem{vuolo2018} Vuolo, F., et al.: How much does multi-temporal Sentinel-2 data improve crop type classification? Int. J. Appl. Earth Obs. Geoinf. 72, 122--130 (2018)
\bibitem{lebourgeois2017} Lebourgeois, V., et al.: A combined Random Forest and OBIA classification scheme for mapping smallholder agriculture. Remote Sens. 9(3), 259 (2017)
\bibitem{inglada2015} Inglada, J., et al.: Assessment of an operational system for crop type map production using high temporal and spatial resolution satellite optical imagery. Remote Sens. 7(9), 12356--12379 (2015)
\bibitem{maxwell2018} Maxwell, A.E., Warner, T.A., Fang, F.: Implementation of machine-learning classification in remote sensing: An applied review. Int. J. Remote Sens. 39(9), 2784--2817 (2018)
\bibitem{chen2016} Chen, T., Guestrin, C.: XGBoost: A scalable tree boosting system. In: Proc. KDD '16, pp. 785--794. ACM (2016)
\bibitem{gumma2020} Gumma, M.K., et al.: Crop type identification and spatial mapping using Sentinel-2 satellite data with focus on field-level information. Geocarto Int. 37(13), 3824--3841 (2020)
\bibitem{gumma2022} Gumma, M.K., et al.: Multiple agricultural cropland products of South Asia developed using Landsat-8 30 m and MODIS 250 m data. GISci. Remote Sens. 59(1), 1048--1077 (2022)
\bibitem{haralick1973} Haralick, R.M., Shanmugam, K., Dinstein, I.: Textural features for image classification. IEEE Trans. Syst. Man Cybern. SMC-3(6), 610--621 (1973)
\end{thebibliography}

\end{document}
"""
    with open(TEX_PATH, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"LaTeX document saved to {TEX_PATH}")

def main():
    print("=" * 60)
    print("Building ICIDA 2026 Submission Artifacts")
    print("=" * 60)
    build_docx()
    page_count = export_pdf()
    build_latex()
    print("=" * 60)
    print(f"Artifact generation complete!")
    print(f"Word Doc: {DOCX_PATH}")
    print(f"PDF Doc:  {PDF_PATH} ({page_count} pages)")
    print(f"LaTeX:    {TEX_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()
