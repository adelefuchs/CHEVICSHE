# **CHEVICSHE**
**Code and experiments for evaluating and predicting cold-start vulnerability in drug–target affinity (DTA) models using the DAVIS and Pharos datasets.**  
This repository contains scripts, processed data, model outputs, and analysis workflows developed as part of CHEVICSHE.

---

## **Overview**
The goal of this project is to systematically evaluate how machine-learning models behave when tested on *novel, previously unseen proteins*, and to predict this vulnerability using metrics derived from cold-protein experiments. The workflow includes:

- Data curation and preprocessing for DAVIS and Pharos  
- Training, evaluating, and extracting embeddings from multiple DTA architectures  
- Computing cold-start reliability metrics 
- Experiments for model-prediction correction and analysis  

---

## **Repository Structure**

### **`DAVIS/` and `pharos/`**
Notebooks and scripts for:
- Data curation and cleaning  
- Sequence and SMILES extraction  
- Mapping drug/protein IDs  

These folders provide the full pipeline for transforming publicly available datasets into the formats used by all downstream models.

---

### **Model Folders**
Each model has its own directory:

- **`GraphDTA/`**  
- **`ColdDTA/`**  
- **`LLMDTA/`**  

Each directory contains:
- Training and evaluation scripts  
- Data loaders and processing utilities  
- Model predictions across folds  
- Configuration files  
- Three analysis notebooks:
  - **Analysis of Predictions** – exploratory evaluation of predictions across cold and warm settings  
  - **Metrics** – computation of metrics and related components  
  - **Correction** – experiments for prediction adjustment and error reduction  
The notebooks produce all plots used in the thesis that rely on data.

**Note:**  
The **original implementations** of these models are published by their respective authors and should be referenced for standalone reproduction:

- *GraphDTA* – original author GitLab repo  
- *ColdDTA* – original author GitLab repo  
- *LLMDTA* – original author GitLab repo  

This repository includes specific restructuring and extensions supporting nested cold-protein blinding, per-fold prediction storage, and embedding extraction.

---

## **Data and Embeddings**
Some processed data files—particularly:
- large **ESM-1b protein embedding dictionaries** (`.pkl`)  
- **graph-encoded drug embeddings** (`.pkl`)  
- **processed PyTorch dataset files** (`.pt`)  

were **too large to push to GitLab**.

However:
- All necessary scripts to regenerate them are included.