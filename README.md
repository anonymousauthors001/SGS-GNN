# Instruction

# SGS-GNN: A Supervised Graph Sparsifier for Graph Neural Networks

SGS-GNN is a novel supervised graph sparsification algorithm that learns the sampling probability distribution of edges and samples sparse subgraphs of a user-specified size to reduce the memory required by GNNs for inference tasks on large graphs.

# Installation:

These are the necessary packages for installation from scratch and other related packages.

```
Python version: 3.11
Pytorch version: 2.0.1
Cuda: 11.7
Cudnn: 8.6
Pytorch-Geometric: 2.3.1
```

For direct installation, Conda packages are in `environment.yml`, and PIP packages are in `requirements.txt` and can be imported as,

```
conda env create -f environment.yml
pip install -r requirements.txt
```

## Run 

python main.py --dataset SmallCora --mode learned --runs 3 --epochs 250 --save_csv True --edge_mlp_type GCN --GNN GCN --log True --sparse_edge_mlp True --conditional True --reg1 True --reg2 True --hybrid_checkpoint --pipeline hybrid

## Demo run

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

COMMON_ARGS=(
  --dataset Reddit
  --mode learned
  --runs 1
  --epochs 10
  --save_csv True
  --edge_mlp_type GCN
  --GNN GCN
  --log False
  --sparse_edge_mlp True
  --conditional True
  --reg1 True
  --reg2 True
  --stats True
  --hybrid_checkpoint True
)

run_pipeline () {
  local pipeline="$1"
  local log_file="${ROOT_DIR}/pipeline_${pipeline}.log"
  echo "=== Running pipeline: ${pipeline} ==="
  (cd "${ROOT_DIR}" && python main.py "${COMMON_ARGS[@]}" --pipeline "${pipeline}") | tee "${log_file}"
  echo "--- Stats (${pipeline}) ---"
  grep -n "\\[stats\\]" "${log_file}" || true
  echo ""  
}

run_pipeline "two_pass"
run_pipeline "straight_through"
run_pipeline "hybrid"
```

# Diagram (SGS-GNN is the hybrid version in the paper)

Below are corrected, syntax‑valid Mermaid diagrams for the three pipelines. Solid arrows for forward pass and dashed arrows for backward/grad flow. Key differences (straight‑through vs detach vs two‑pass recompute) are also highlighted.

**Straight‑Through**

```mermaid
flowchart LR
  X["batch.x"] --> E["EdgeProbMLP (grad)"]
  EI["batch.edge_index"] --> E
  E --> P["edge_probs_full"]

  P --> GS["gumbel_softmax_sampling (straight-through)"]
  EI --> GS
  GS --> SEI["sampled_edge_index"]
  GS --> SEW["sampled_edge_weight (straight-through)"]

  X --> G["GNN"]
  SEI --> G
  SEW --> G

  G --> L["loss"]
  SEW --> R1["reg1 BCE"]
  SEI --> R1
  G --> R2["reg2 consistency"]
  SEW --> R2
  R1 --> L
  R2 --> L

  L -.-> G
  L -.-> E
  L -.-> GS
  GS -.-> P
```

**Hybrid**

```mermaid
flowchart LR
  X["batch.x"] --> E["EdgeProbMLP (grad, optional checkpoint)"]
  EI["batch.edge_index"] --> E
  E --> P["edge_probs_full"]

  P -->|"detach"| GS["gumbel_softmax_sampling"]
  EI --> GS
  GS --> SEI["sampled_edge_index"]

  P --> IDX["index_select by sampled_edge_index"]
  IDX --> SEW["edge_probs_sampled"]

  X --> G["GNN"]
  SEI --> G
  SEW --> G

  G --> L["loss"]
  SEW --> R1["reg1 BCE"]
  SEI --> R1
  G --> R2["reg2 consistency"]
  SEW --> R2
  R1 --> L
  R2 --> L

  L -.-> G
  L -.-> E
  L -.-> IDX
```

**Two‑Pass**

```mermaid
flowchart LR
  X["batch.x"] --> E1["EdgeProbMLP pass1 (no grad)"]
  EI["batch.edge_index"] --> E1
  E1 --> P["edge_probs_full (detached)"]

  P --> GS["gumbel_softmax_sampling"]
  EI --> GS
  GS --> SEI["sampled_edge_index"]

  X --> E2["EdgeProbMLP pass2 (grad, sampled only)"]
  SEI --> E2
  E2 --> SEW["edge_probs_sampled"]

  X --> G["GNN"]
  SEI --> G
  SEW --> G

  G --> L["loss"]
  SEW --> R1["reg1 BCE"]
  SEI --> R1
  G --> R2["reg2 consistency"]
  SEW --> R2
  R1 --> L
  R2 --> L

  L -.-> G
  L -.-> E2
```

# Full Diagram

Below are full, syntax‑valid Mermaid diagrams for each pipeline, with separate subgraphs for the conditional gate, random baseline branch, and optimizer steps. Solid arrows are forward pass; dashed arrows are backward/grad flow.

**Straight‑Through**

```mermaid
flowchart LR
  subgraph Forward_Learned["Forward (learned path)"]
    X["batch.x"] --> E["EdgeProbMLP (grad)"]
    EI["batch.edge_index"] --> E
    E --> P["edge_probs_full"]

    P --> GS["gumbel_softmax_sampling (straight-through, temp, degree bias)"]
    EI --> GS
    GS --> SEI["sampled_edge_index"]
    GS --> SEW["sampled_edge_weight (straight-through)"]

    X --> G["GNN (learned edges)"]
    SEI --> G
    SEW --> G
    G --> Lout["learned_out"]
  end

  subgraph Random_Baseline["Random baseline (conditional)"]
    RP["F.softmax(batch.prob)"] --> RSEL["random_edge_sample"]
    RSEL --> RSEI["random_sampled_edge_index"]
    X --> RG["GNN (random edges)"]
    RSEI --> RG
    RG --> Rout["random_out"]
  end

  subgraph Regularizers["Regularizers (learned path)"]
    SEW --> R1["reg1 BCE"]
    SEI --> R1
    Lout --> R2["reg2 consistency"]
    SEW --> R2
  end

  subgraph Gate["Conditional update gate"]
    Lout --> F1L["F1(learned_out)"]
    Rout --> F1R["F1(random_out)"]
    F1L --> CMP["compare F1"]
    F1R --> CMP
    CMP -->|learned wins| Llearn["loss(learned_out + regs)"]
    CMP -->|random wins| Lrand["loss(random_out)"]
  end

  R1 --> Llearn
  R2 --> Llearn

  subgraph Optimizers["Optimizers"]
    Llearn --> Oe["optimizer_edge_prob.step()"]
    Llearn --> Og["optimizer_gnn.step()"]
    Lrand --> Ogr["optimizer_gnn.step()"]
  end

  Llearn -.-> G
  Llearn -.-> E
  Llearn -.-> GS
  Lrand -.-> RG
```

**Hybrid**

```mermaid
flowchart LR
  subgraph Forward_Learned["Forward (learned path)"]
    X["batch.x"] --> E["EdgeProbMLP (grad, optional checkpoint)"]
    EI["batch.edge_index"] --> E
    E --> P["edge_probs_full"]

    P -->|"detach"| GS["gumbel_softmax_sampling (temp, degree bias)"]
    EI --> GS
    GS --> SEI["sampled_edge_index"]

    P --> IDX["index_select by sampled_edge_index"]
    IDX --> SEW["edge_probs_sampled"]

    X --> G["GNN (learned edges)"]
    SEI --> G
    SEW --> G
    G --> Lout["learned_out"]
  end

  subgraph Random_Baseline["Random baseline (conditional)"]
    RP["F.softmax(batch.prob)"] --> RSEL["random_edge_sample"]
    RSEL --> RSEI["random_sampled_edge_index"]
    X --> RG["GNN (random edges)"]
    RSEI --> RG
    RG --> Rout["random_out"]
  end

  subgraph Regularizers["Regularizers (learned path)"]
    SEW --> R1["reg1 BCE"]
    SEI --> R1
    Lout --> R2["reg2 consistency"]
    SEW --> R2
  end

  subgraph Gate["Conditional update gate"]
    Lout --> F1L["F1(learned_out)"]
    Rout --> F1R["F1(random_out)"]
    F1L --> CMP["compare F1"]
    F1R --> CMP
    CMP -->|learned wins| Llearn["loss(learned_out + regs)"]
    CMP -->|random wins| Lrand["loss(random_out)"]
  end

  R1 --> Llearn
  R2 --> Llearn

  subgraph Optimizers["Optimizers"]
    Llearn --> Oe["optimizer_edge_prob.step()"]
    Llearn --> Og["optimizer_gnn.step()"]
    Lrand --> Ogr["optimizer_gnn.step()"]
  end

  Llearn -.-> G
  Llearn -.-> E
  Llearn -.-> IDX
  Lrand -.-> RG
```

**Two‑Pass**

```mermaid
flowchart LR
  subgraph Forward_Learned["Forward (learned path)"]
    X["batch.x"] --> E1["EdgeProbMLP pass1 (no grad)"]
    EI["batch.edge_index"] --> E1
    E1 --> P["edge_probs_full (detached)"]

    P --> GS["gumbel_softmax_sampling (temp, degree bias)"]
    EI --> GS
    GS --> SEI["sampled_edge_index"]

    X --> E2["EdgeProbMLP pass2 (grad, sampled only)"]
    SEI --> E2
    E2 --> SEW["edge_probs_sampled"]

    X --> G["GNN (learned edges)"]
    SEI --> G
    SEW --> G
    G --> Lout["learned_out"]
  end

  subgraph Random_Baseline["Random baseline (conditional)"]
    RP["F.softmax(batch.prob)"] --> RSEL["random_edge_sample"]
    RSEL --> RSEI["random_sampled_edge_index"]
    X --> RG["GNN (random edges)"]
    RSEI --> RG
    RG --> Rout["random_out"]
  end

  subgraph Regularizers["Regularizers (learned path)"]
    SEW --> R1["reg1 BCE"]
    SEI --> R1
    Lout --> R2["reg2 consistency"]
    SEW --> R2
  end

  subgraph Gate["Conditional update gate"]
    Lout --> F1L["F1(learned_out)"]
    Rout --> F1R["F1(random_out)"]
    F1L --> CMP["compare F1"]
    F1R --> CMP
    CMP -->|learned wins| Llearn["loss(learned_out + regs)"]
    CMP -->|random wins| Lrand["loss(random_out)"]
  end

  R1 --> Llearn
  R2 --> Llearn

  subgraph Optimizers["Optimizers"]
    Llearn --> Oe["optimizer_edge_prob.step()"]
    Llearn --> Og["optimizer_gnn.step()"]
    Lrand --> Ogr["optimizer_gnn.step()"]
  end

  Llearn -.-> G
  Llearn -.-> E2
  Lrand -.-> RG
```


