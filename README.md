# CGMN: Confounding-Aware Bilateral Graph Matching for Two-Sided Match Quality Prediction

Official artifact repository for:

> **CGMN: Confounding-Aware Bilateral Graph Matching for Two-Sided Match Quality Prediction**

CGMN studies **two-sided match quality prediction**, where each observation consists of two heterogeneous entities and the objective is to predict their post-match outcome. The framework preserves the bilateral structure of a matched pair, models within-side feature dependencies and cross-side compatibility separately, and incorporates nuisance-robust representation learning.

## Overview

CGMN is motivated by two characteristics of matching data:

1. **Bilateral heterogeneous structure.**
   A match consists of two distinct entities, such as a passenger and a driver or a recipient and a donor. Flattening both sides into one feature vector may obscure their different provenance and relation types.

2. **Observational nuisance variation.**
   Historical matching records may contain patterns associated with selection, exposure, availability, or other observational mechanisms that do not generalize reliably.

CGMN addresses these issues through:

* **Bilateral feature graphs**, which preserve the provenance of the two entities;
* **Intra-side graph modeling**, which captures dependencies among features within each entity;
* **Cross-side attention**, which models compatibility between the two entities;
* **Variational Confounding-Aware Inference (VCI)**, which separates stable and nuisance-sensitive predictive structure.

The empirical focus is **prediction and robustness**. The framework is not intended to identify treatment effects or claim identification of real-world causal confounders from observational data.

## Repository Structure

```text
CGMN-Bilateral-Matching/
├── CGMN-Bilateral-Matching.zip
├── baselines/
│   ├── README.md
│   ├── requirements.txt
│   └── tabular_models/
│       ├── TALENT/
│       └── ...
└── orgin_train_data.csv
```

### `CGMN-Bilateral-Matching.zip`

This archive contains a runnable bilateral matching quality evaluation toolkit, including:

```text
matchingquality/
├── __init__.py
├── cli.py
├── io.py
└── metrics.py

examples/
├── predicted.csv
└── reference.csv
```

The toolkit supports loading prediction/reference files, computing matching-quality metrics, and command-line evaluation.

### `baselines/`

This directory contains source code for the tabular-learning baselines used in our experimental pipeline.

| Model          | Implementation              |
| -------------- | --------------------------- |
| TabM           | TALENT-based implementation |
| TabPFN         | TALENT-based implementation |
| TabICL         | TALENT-based implementation |
| TabTransformer | TALENT-based implementation |
| FTTransformer  | TALENT-based implementation |
| AMFormer       | TALENT-based implementation |
| xRFM           | Bundled xRFM implementation |

See [`baselines/README.md`](baselines/README.md) for the corresponding source locations.

## Installation

Clone the repository:

```bash
git clone https://github.com/jhfeng0215/CGMN-Bilateral-Matching.git
cd CGMN-Bilateral-Matching
```

### Evaluation toolkit

Extract the evaluation package:

```bash
unzip CGMN-Bilateral-Matching.zip
cd CGMN-Bilateral-Matching
```

Create a Python environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Tabular baselines

The dependencies for the included tabular baselines can be installed separately:

```bash
pip install -r baselines/requirements.txt
```

The baseline environment includes packages such as PyTorch, scikit-learn, NumPy, SciPy, Optuna, category-encoders, and related dependencies.

## Quick Start: Matching-Quality Evaluation

After extracting `CGMN-Bilateral-Matching.zip`, the included example can be evaluated with:

```bash
bash scripts/run_matching_quality.sh \
  --predicted examples/predicted.csv \
  --reference examples/reference.csv
```

To save the evaluation report:

```bash
bash scripts/run_matching_quality.sh \
  --predicted examples/predicted.csv \
  --reference examples/reference.csv \
  --output outputs/example_report.json
```

If the prediction file contains multiple candidate pairs and a `score` column, one-to-one best-match selection can be applied before evaluation:

```bash
bash scripts/run_matching_quality.sh \
  --predicted examples/predicted.csv \
  --reference examples/reference.csv \
  --select-best
```

## Input Format

The evaluation utility expects paired identifiers such as:

| Column     | Description                                 |
| ---------- | ------------------------------------------- |
| `left_id`  | Identifier of the entity on the first side  |
| `right_id` | Identifier of the entity on the second side |
| `score`    | Optional predicted matching-quality score   |

Custom column names can also be specified through the command-line interface.

## Evaluation Metrics

The included evaluation utility reports:

* number of predicted pairs;
* number of reference pairs;
* number of true-positive pairs;
* precision;
* recall;
* F1;
* average prediction score, when available;
* score coverage.

## Experimental Baselines

The manuscript compares CGMN with baselines from several model families, including:

### Classical and tree-based methods

* Decision Tree
* Random Forest
* AdaBoost
* XGBoost
* LightGBM

### Deep tabular methods

* DNN
* TabTransformer
* FTTransformer
* TabM
* TabPFN
* TabICL
* AMFormer
* xRFM

### Graph and robustness-oriented methods

* GAT
* CAL
* PrunE
* CausalDiff

The `baselines/` directory currently provides the source implementations for the included modern tabular baselines.

## Data Availability

The study uses both public and proprietary data.

The **ride-hailing dataset cannot be publicly released** because it contains proprietary platform information and user-related attributes. Therefore, experiments requiring the private ride-hailing data cannot be reproduced without authorized access to the original data.

We provide all data and source materials that can legally be shared in this repository.

The transplantation experiments are retrospective predictive benchmarks. They should not be interpreted as autonomous organ-allocation rules, clinical decision systems, or replacements for clinical policy.

## Reproducibility Scope

The public artifact is intended to support:

* inspection of the provided baseline implementations;
* execution of the bilateral matching evaluation utility;
* evaluation using shareable data;
* inspection of the experimental dependencies and baseline source code.

Experiments that depend on proprietary ride-hailing records require access to the corresponding private data.

## Baseline Licenses

Part of the tabular baseline code is derived from the corresponding upstream implementations.

* TALENT-derived source files retain the upstream license under `baselines/tabular_models/LICENSE`.
* The bundled xRFM source retains its corresponding license in the xRFM package directory.

Please follow the original licenses when reusing these components.




The citation information will be updated after publication.

## Contact

For questions about the code or experiments, please contact us.
University of Science and Technology of China
Email: [jiahuifeng@mail.ustc.edu.cn](mailto:jiahuifeng@mail.ustc.edu.cn)
