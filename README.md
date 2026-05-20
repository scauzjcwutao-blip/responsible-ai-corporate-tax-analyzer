# Responsible AI Corporate Tax Analyzer

**A transparent and explainable AI pipeline for corporate tax analysis and compliance risk assessment.**

This repository implements a **Responsible AI** framework specifically designed for corporate tax (Unternehmenssteuer) analysis. It combines large language models (LLM) with explainable machine learning to automatically analyze tax law provisions, predict effective tax rates (ETR), detect potential tax planning risks, and provide **source-grounded, human-interpretable explanations**.

The project directly aligns with the principles of the ETH Zürich SNSF project **"Responsible AI for the Swiss Judiciary"** by emphasizing transparency, traceability, and accountability in AI-assisted legal decision support.

---

## ✨ Key Features

- **LLM-powered Tax Law Understanding**  
  RAG-based retrieval and analysis of corporate tax regulations with explicit source citation (reduces hallucination).

- **Explainable ML Tax Prediction**  
  XGBoost / LightGBM models predict Effective Tax Rate (ETR) with **SHAP** global and local explanations.

- **Tax Risk Assessment Module**  
  Automatically flags high-risk tax planning behaviors (e.g., abnormal deferred tax, aggressive deductions, thin capitalization).

- **Strict Responsible AI Design**  
  - Full traceability to legal sources  
  - Model-agnostic explanations  
  - No black-box decisions  
  - Reproducible out-of-sample validation

- **Interactive Demo Notebook**  
  One-click execution with synthetic and real-world corporate financial data.
---
## 🇩🇪 German Corporate Tax AI System – Live Demo

**Responsible AI for Corporate Taxation**  
A production-ready, explainable AI system that combines **machine learning prediction**, **SHAP interpretability**, and **Retrieval-Augmented Generation (RAG)** for German corporate tax law.

This demo showcases the full pipeline: from synthetic data modeling to transparent predictions and automatic legal grounding.

---

### 📊 1. Synthetic Data & Model Training
```bash
Generating synthetic German corporate tax data...
Training set: 400 samples | Test set: 100 samples
Features: 22 | Target: effective_tax_rate (mean = 0.2892, std = 0.0312)
```
Training XGBoost model...
→ MAE: 0.0043 (0.43 percentage points)
→ R²:  0.9512
```
📋 2. Prediction Examplesbash
Prediction breakdown (first 5 test samples):

Unternehmen 42   → statutory 30.18% | predicted 27.85% | deviation -2.33 pp
Unternehmen 107  → statutory 29.37% | predicted 29.01% | deviation -0.36 pp
...
```
🔍 3. Explainability with SHAP
Computing SHAP explanations...
Top 10 features by mean |SHAP| value:

1. hebesatz                → 0.012453
2. beteiligungsertraege    → 0.008721
3. verlustvortraege        → 0.007234
4. organschaft             → 0.005102
5. zinsaufwand             → 0.003891
6. mieten_pachten          → 0.003204
...
```
📚  4. Tax Law Retrieval (RAG)
✅ 12 German tax provisions loaded into vector database

❓ Query: Wie werden Beteiligungserträge bei der Körperschaftsteuer behandelt?
   📖 §8b Abs. 1 KStG – Freistellung von Beteiligungserträgen (Relevance: 0.891)

❓ Query: Wann gehen Verlustvorträge bei Anteilsübertragung unter?
   📖 §8c Abs. 1 KStG – Verlustabzug bei Körperschaften (Relevance: 0.856)

❓ Query: Welche Hinzurechnungen gibt es bei der Gewerbesteuer?
   📖 §8 Nr. 1 GewStG – Hinzurechnungen (Relevance: 0.912)
 ```
🔗 COMBINED: SHAP finding → automatic law lookup
Top SHAP feature: 'hebesatz'
```bash
→ Automatic legal lookup triggered...
   📖 §11 Abs. 1-3 GewStG – Steuermesszahl und Hebesatz
   📝 "Die Gewerbesteuer wird auf der Grundlage des Steuermessbetrags festgesetzt..."
```
✅ Demo complete.
```
## 🎯 Motivation

Corporate tax compliance and planning involve complex, jurisdiction-specific rules. Traditional manual review is time-consuming and prone to inconsistency. This project demonstrates how **Responsible AI** can support tax authorities, companies, and legal professionals by providing transparent, auditable, and explainable tax analysis — particularly relevant for Swiss and EU corporate tax regimes.

---

## 📊 Tech Stack

- Python 3.10+
- scikit-learn, XGBoost, LightGBM
- SHAP (explainability)
- LangChain / LlamaIndex (RAG)
- Pandas, NumPy, Matplotlib
- Jupyter Notebook

---
```bash
Structure
responsible-ai-corporate-tax-analyzer/
tax_law_database/
├── manifest.json                        ← Main manifest (all version indexes + checksums)
├── src/                                 ← Source code
│   ├── main.py                          ← Application entry point
│   ├── config.py                        ← Configuration & environment variables
│   ├── prediction/                      ← Module 1: Tax rate prediction (ML)
│   │   ├── model.py
│   │   ├── features.py
│   │   ├── train.py
│   │   └── scenarios.py
│   ├── versioning/                      ← Module 2: Temporal law versioning
│   │   ├── timeline.py
│   │   ├── resolver.py
│   │   └── diff.py
│   ├── application/                     ← Module 3: Intelligent law application
│   │   ├── rules_engine.py
│   │   ├── loss_deduction.py            ← §8c KStG logic
│   │   ├── interest_barrier.py          ← §4h EStG / §8a KStG
│   │   ├── trade_tax.py                 ← GewSt additions & deductions
│   │   └── reorganization.py            ← UmwStG logic
│   ├── rates/                           ← Module 4: Tax rate database
│   │   ├── scraper.py
│   │   ├── municipality.py
│   │   └── property_tax.py
│   ├── sync/                            ← Module 5: Change detection & delta sync
│   │   ├── monitor.py
│   │   ├── bgbl_fetcher.py
│   │   ├── bmf_fetcher.py
│   │   └── notifications.py
│   ├── comparison/                      ← Module 6: Synopsis & version comparison
│   │   ├── synopsis.py
│   │   ├── diff_renderer.py
│   │   └── changelog.py
│   ├── export/                          ← Module 7: Download & export
│   │   ├── json_export.py
│   │   ├── csv_export.py
│   │   ├── markdown_export.py
│   │   ├── zip_builder.py
│   │   └── checksum.py                  ← SHA-256 integrity
│   ├── calculation/                     ← Module 8: Tax calculation & planning
│   │   ├── corporate_tax.py
│   │   ├── trade_tax_calc.py
│   │   ├── combined_burden.py
│   │   └── scenario_runner.py
│   ├── api/                             ← REST API layer
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   └── middleware.py
│   └── utils/                           ← Shared utilities
│       ├── logger.py
│       ├── validators.py
│       └── constants.py
├── current/                             ← Latest version of all provisions
│   ├── KStG_P8c.json
│   ├── EStG_P7g.json
│   ├── GewStG_P8_Nr._1.json
│   └── ...
├── historical/                          ← Historical versions
│   ├── VZ_2015/                         ← 2015 assessment period snapshot
│   │   ├── KStG_P8c.json
│   │   └── ...
│   ├── VZ_2020/
│   │   └── ...
│   └── KStG_P8c/                        ← §8c complete history
│       ├── KStG_P8c_v1_2008.json
│       ├── KStG_P8c_v2_bverfg_2017.json
│       └── KStG_P8c_v3_2018.json
├── tax_rates/
│   ├── tax_rates_2023.json
│   └── tax_rates_2024.json
└── export/
    └── VZ_2023/
        ├── tax_law_VZ2023.json          ← Machine-readable
        ├── tax_law_VZ2023.csv           ← Excel-compatible
        ├── tax_law_VZ2023.md            ← Human-readable
        └── tax_law_VZ2023_complete.zip  ← Full package
```
## 🚀 Quick Start

```bash
git clone https://github.com/your-repo/tax-law-database.git
cd tax-law-database
pip install -r requirements.txt
python src/main.py
```
## 📡 API Usage
```bash
# Get current version of §8c KStG
GET /api/v1/current/KStG/8c

# Get historical version for assessment period 2016
GET /api/v1/historical/KStG/8c?vz=2016

# Get synopsis comparison between two assessment periods
GET /api/v1/compare/KStG/8c?from_vz=2015&to_vz=2024

# Get trade tax rate for a specific municipality
GET /api/v1/rates/trade_tax?municipality=Frankfurt&year=2024

# Run tax rate prediction scenario
POST /api/v1/predict/corporate_tax
{
  "horizon": 5,
  "scenario": "base_case",
  "gdp_growth": 1.2,
  "debt_ratio": 65.0
}

# Calculate combined tax burden
POST /api/v1/calculate/combined
{
  "taxable_income": 1000000,
  "municipality": "München",
  "vz": 2024
}
```
