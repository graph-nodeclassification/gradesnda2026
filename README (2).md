# Decoupled Graph Attention Networks (DGAT)

> **[Regular Research Paper]** Accepted at GRADES-NDA '26 — 9th Joint Workshop on Graph Data Management Experiences & Systems (GRADES) and Network Data Analytics (NDA), co-located with SIGMOD 2026, Bengaluru, India.

---

## Overview

**DGAT** is a scalable, parameter-efficient graph neural network designed specifically for **heterophilic graphs** — real-world networks where connected nodes tend to belong to *different* classes (e.g., buyer–seller networks, web hyperlink graphs).

Standard message-passing GNNs (GCN, GAT) assume homophily, meaning neighbours are likely to share labels. This assumption breaks down in heterophilic settings, where neighbourhood aggregation introduces inter-class noise and degrades representation quality — sometimes performing worse than a featureless MLP. DGAT resolves this with a **fully pre-computable signed propagation primitive** that partitions neighbours into positive (similar) and negative (dissimilar) channels and applies signed aggregation, achieving heterophily-aware representation learning with no graph operations at training time.

---

## Key Contributions

- **Pre-computable signed propagation** — cosine-similarity-based neighbourhood partitioning into positive and negative channels, computed once offline before training begins.
- **Scalability guarantee** — precomputation cost of O(K|E|d) and training cost of O(NdC) per epoch, matching SGC and enabling linear scalability.
- **Theoretical result** — formal proof that signed message propagation strictly improves class separation over SGC under heterophily by a factor of (2ρ−1)²(1+β)² relative to standard smoothing (Theorem 1).
- **Strong empirical performance** — up to 1.31× accuracy improvement over standard baselines across six heterophilic benchmarks.

---

## Method

### 1. Feature Normalisation & Cosine Similarity

Each node feature vector is L2-normalised, and the cosine similarity between connected nodes i and j is computed as their inner product:

```
s_ij = <x̃_i, x̃_j>  ∈ [-1, 1]
```

### 2. Signed Neighbourhood Partitioning

Edges are split into two disjoint channels using a threshold τ:

- **Positive channel E⁺** — edges where s_ij ≥ τ (feature-aligned, likely same-class)
- **Negative channel E⁻** — edges where s_ij < τ (feature-misaligned, likely cross-class)

Default: τ = 0.0 (boundary at feature orthogonality).

### 3. Per-Channel Softmax Attention

Softmax attention weights are computed independently per channel using the raw cosine similarity as the unnormalised logit. All weights depend only on fixed input features — **no learnable parameters, no recomputation during training**.

### 4. Signed Attentive Aggregation (K hops)

At each hop t:

```
H⁺_i = Σ_{j∈N⁺(i)} α⁺_ij · H_{j}^{t-1}      (pull toward similar neighbours)
H⁻_i = Σ_{j∈N⁻(i)} α⁻_ij · H_{j}^{t-1}      (collect dissimilar neighbours)
H_i^t = H⁺_i  −  β · H⁻_i                    (signed update)
```

where β > 0 controls the weight of the negative channel.

### 5. Skip Connection & Linear Classifier

After K hops, the propagated features are concatenated with the original input:

```
Z = [H^(K) ∥ X]  ∈ ℝ^{N×2d}
```

Z is entirely pre-computed. At training time, a **single linear layer** (multinomial logistic regression) is optimised over Z — no graph operations, no neighbourhood lookup.

---

## Complexity

| Phase | Complexity |
|---|---|
| Pre-computation | O(K \|E\| d) |
| Training (per epoch) | O(N · d · C) |

DGAT matches SGC's complexity class while handling heterophily.

---

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

Results are printed per dataset and summarised in a final table:

```
===================================================================
  DGAT — Test Accuracy mean ± std %
  (tau=0.0, beta=0.5, dropout=0.6, lr=0.01, wd=0.0005)
===================================================================
  K       Cornell            Texas              Wisconsin     ...
  ---------------------------------------------------------------
  K=1     71.62±2.20         77.03±5.90         80.78±4.69   ...
  ...
```

---

## Configuration

All hyperparameters are set in `configs.py`:

| Parameter | Default | Description |
|---|---|---|
| `K_VALUES` | [1,2,3,4,6,8,10] | Propagation hops to sweep |
| `RUNS` | 10 | Number of data splits per dataset |
| `EPOCHS` | 1000 | Max training epochs |
| `LR` | 0.01 | Adam learning rate |
| `WEIGHT_DECAY` | 5e-4 | L2 regularisation |
| `DROPOUT` | 0.6 | Dropout on Z before linear layer |
| `PATIENCE` | 200 | Early stopping patience |
| `TAU` | 0.0 | Cosine similarity threshold for channel split |
| `BETA` | 0.5 | Negative channel weight in signed update |

---

## Datasets

| Dataset | Nodes | Edges | Features | Classes | Type |
|---|---|---|---|---|---|
| Wisconsin | 251 | 499 | 1,703 | 5 | Heterophilic (WebKB) |
| Texas | 183 | 295 | 1,703 | 5 | Heterophilic (WebKB) |
| Cornell | 183 | 280 | 1,703 | 5 | Heterophilic (WebKB) |
| Actor | 7,600 | 26,752 | 931 | 5 | Heterophilic |
| Squirrel | 5,201 | 198,493 | 2,089 | 5 | Heterophilic (Wikipedia) |
| Chameleon | 2,277 | 31,421 | 2,325 | 5 | Heterophilic (Wikipedia) |

All datasets are downloaded automatically via PyTorch Geometric on first run.

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

## Citation

```bibtex
@inproceedings{dgat2026,
  title     = {Decoupled Graph Attention Networks},
  booktitle = {Proceedings of the 9th Joint Workshop on Graph Data Management
               Experiences \& Systems (GRADES) and Network Data Analytics (NDA)},
  series    = {GRADES-NDA '26},
  year      = {2026},
  address   = {Bengaluru, India},
  publisher = {ACM},
  doi       = {XXXXXXX.XXXXXXX}
}
```

---

## License

Code is released for research use. See individual source files for details.
