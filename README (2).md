# Decoupled Graph Attention Networks (DGAT)

## Overview

**DGAT** is a scalable, parameter-efficient graph neural network designed specifically for **heterophilic graphs** .Standard message-passing GNNs (GCN, GAT) assume homophily, meaning neighbours are likely to share labels. This assumption breaks down in heterophilic settings, where neighbourhood aggregation introduces inter-class noise and degrades representation quality sometimes performing worse than a featureless MLP. DGAT resolves this with a **fully pre-computable signed propagation primitive** that partitions neighbours into positive (similar) and negative (dissimilar) channels and applies signed aggregation, achieving heterophily-aware representation learning with no graph operations at training time.

## Repository Structure

```
.
├── configs.py          # Hyperparameters and dataset list
├── data.py             # Dataset loading (WebKB, WikipediaNetwork, Actor)
├── graph_utils.py      # Edge preprocessing, node homophily computation
├── propagation.py      # Core DGAT signed propagation (compute_dgat)
├── model.py            # Linear classifier (LogSoftmax over dropout(Z))
├── trainer.py          # Training loop with early stopping
├── main.py             # Benchmark runner, result aggregation, table printer
└── requirements.txt    # Python dependencies
```

---

## Installation

```bash
pip install torch torch_geometric
```

For GPU support, install the appropriate CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) before installing `torch_geometric`.

---

## Usage

Run the full benchmark across all datasets and K values:

```bash
python main.py
```
---

## Results

Node classification accuracy (mean ± std %, 10 random splits):

| Model | Texas | Wisconsin | Cornell | Squirrel | Chameleon | Actor |
|---|---|---|---|---|---|---|
| MLP | 69.23±6.49 | 74.53±2.37 | 70.92±6.49 | 35.25±1.78 | 42.11±2.18 | 34.53±0.70 |
| GCN | 54.87±7.87 | 55.00±4.31 | 52.31±2.61 | 26.03±0.82 | 40.35±1.46 | 27.32±1.10 |
| GAT | 56.41±2.81 | 49.62±5.36 | 56.41±2.29 | 40.67±1.44 | 43.64±2.04 | 27.44±0.89 |
| H2GCN | 72.15±2.65 | 75.77±4.65 | 70.46±1.64 | 36.48±1.86 | 56.66±2.11 | 35.70±1.00 |
| SGC | 60.81±7.12 | 50.98±6.06 | 42.97±5.76 | 37.82±1.83 | 54.25±4.07 | 29.36±1.14 |
| **DGAT** | **77.03±5.90** | **80.78±4.69** | **71.62±2.20** | **42.13±1.61** | **59.93±2.45** | 35.20±0.82 |

DGAT achieves the highest accuracy on 5 of 6 datasets and is competitive on Actor.

---

## License

Code is released for research use. See individual source files for details.
