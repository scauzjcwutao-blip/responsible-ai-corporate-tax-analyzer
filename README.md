# responsible-ai-corporate-tax-analyzer
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
├── src/
│   ├── tax_models.py         
│   ├── rag_tax_law.py        
│   ├── shap_explainer.py    
│   └── utils.py
├── notebooks/
│   └── demo_corporate_tax.ipynb 
├── data/
│   ├── synthetic_corporate_financials.csv   # 合成公司数据（含税前利润、扣除项等）
│   └── tax_rules_sample.json                # 公司税法规则示例（可替换为瑞士/欧盟/中国税法）
├── requirements.txt
├── README.md
└── LICENSE (MIT)
```
## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/scauzjcwutao-blip/responsible-ai-corporate-tax-analyzer.git
cd responsible-ai-corporate-tax-analyzer

# 2. Create and activate environment
conda create -n tax-ai python=3.11 -y
conda activate tax-ai

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the demo
jupyter notebook notebooks/demo_corporate_tax.ipynb
