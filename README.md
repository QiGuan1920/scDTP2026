***
# scDTP: Single-Cell Drug Toxicity Perturbation Prediction Framework

## Overview
**scDTP** is a deep generative framework designed to predict the transcriptional responses of single cells to drug-induced toxicological perturbations. By leveraging baseline (control) cellular states and integrating multi-modal drug properties, scDTP accurately forecasts the complex shifts in single-cell gene expression distributions after drug exposure.

This model is built to handle multiple experimental scenarios (e.g., unseen cell types, novel toxic agents, or complex multi-drug/multi-cell combinations) and serves as a robust computational tool for predictive toxicology, drug safety assessment, and single-cell biology.

## Core Architecture
The scDTP framework is powered by a unified architecture built upon core pillars:

* **Multi-Modal Integration**: 
  The model seamlessly fuses diverse perturbagen data, including molecular attributes, structural representations, imaging features, and network characteristics, to provide a comprehensive representation of drug toxicity.
  
* **Prototype Generation Module**: 
  Instead of relying solely on direct embedding transformations, scDTP dynamically constructs **response prototypes**. By learning from reference combinations (known control-to-perturbed mappings), the model generates a context-aware prototype that bridges the gap between the baseline cell state and the multi-modal features of the target drug.
  
* **Conditional Normalizing Flows**: 
  To capture the complex, high-dimensional stochasticity of single-cell responses, scDTP utilizes **Conditional Normalizing Flows**. Conditioned on the generated prototypes and multi-modal embeddings, the flow model transforms a tractable base distribution into the highly complex, non-linear distribution of the perturbed gene expression profile. This allows for exact log-likelihood estimation and highly realistic cellular sampling.

* **Comprehensive Evaluation**:
  Includes built-in evaluation modules to assess prediction quality across multiple biological and statistical dimensions (e.g., DEG identification rate, R² scores, Pearson correlation, etc.).

## Repository Structure

```text
scDTP_Project/
│
├── core/                       # Core architectural and functional modules
│   ├── __init__.py
│   ├── utils.py                # Environment configuration and reproducibility controls
│   ├── models.py               # Normalizing Flows, Prototype Generation, and network architectures
│   ├── dataset.py              # Data scenario detection and PyTorch dataloaders
│   ├── train.py                # Training loops, likelihood optimization, and distribution loss
│   ├── evaluate.py             # R², DEG identification rates, and distance metrics
│   └── visualize.py            # Training curves and evaluation bar plots
│
├── main.py                     # Main execution pipeline and evaluation scripts
├── requirements.txt            # Python environment dependencies
└── README.md                   # Project documentation
```

## Installation

We recommend using a virtual environment (e.g., Conda) to manage dependencies.

1. Clone or download the repository to your local machine or cloud server.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

The main entry point for running predictions and evaluations is `main.py`. You can easily integrate the pipeline into your own analytical scripts.

### Example Usage

```python
import scanpy as sc
from main import main_unified_drug_prediction

all_results, metrics_df = main_unified_drug_prediction(
    adata=adata,
    drug_attr=drug_attr,
    drug_sm=drug_sm,
    drug_img=drug_img,
    drug_net=drug_net,
    mode='evaluate_all',
    num_epochs=30,          # Configurable hyperparameters
    batch_size=128,
    save_adata=True,
    save_dir="./results/h5ad_final/"
)

# Review the final evaluation metrics
print(metrics_df.head())
```

## Outputs
1. **Evaluation Metrics (`.csv` / DataFrame)**: Comprehensive statistics including Mean/Median R², Pearson/Spearman correlations, Euclidean distances, and Top-N DEG Identification Rates.
2. **AnnData Objects (`.h5ad`)**: Reconstructed datasets integrating control cells, true drug-perturbed cells (if available), and the generated predictions.
3. **Visualizations**: Performance summary plots and training loss curves (NLL, Reconstruction, etc.).
