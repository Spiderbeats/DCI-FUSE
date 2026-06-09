# DCI-FUSE

## Introduction & Background

This repository contains the official implementation and experimental analysis of **DCI-FUSE**. 

Our work is highly inspired by the pioneering framework **FUSE** (as discussed in the *FUSE-verifiers* context). Since the original source code of FUSE has not been publicly released, we have meticulously reproduced its methodology based strictly on the descriptions provided in their publication. However, owing to potential engineering artifacts, subtle implementation details, or reproduction divergences, our direct replication did not fully match the performance metrics reported in the original paper. 

To bridge this gap and optimize performance across diverse evaluation scenarios, we introduced specific, dataset-tailored engineering adjustments in our replication, which are structured as follows:
* **`FUSE on HLE`**: Our reproduced FUSE implementation optimized for the Hard Math Expressions (HLE) dataset.
* **`FUSE on IMO`**: Our reproduced FUSE implementation optimized for the International Mathematical Olympiad (IMO) dataset.

### Theoretical Contribution: DCI-FUSE

To address the limitations of standard verification regimes and enhance generalization, we propose **DCI-FUSE**. The comprehensive theoretical foundations, algorithmic innovations, and rigorous mathematical proofs of our method can be found in our technical paper: 
📄 **`DCI-FUSE_A_Low-Rank_Generalization_of_Unsupervised_Score_Ensembling_under_Latent_Verification_Regimes`**

Our core implementations are categorized by datasets and located in the following directories:
* **`DCI-FUSE on IMO`**: Implementation of our proposed DCI-FUSE framework evaluated on the IMO dataset.
* **`DCI-FUSE on HLE`**: Implementation of our proposed DCI-FUSE framework evaluated on the HLE dataset.

### Baselines & Evaluation

To establish a rigorous comparative benchmark, we implemented two standard baselines for reference:
1. **Pass@1**: Serves as the fundamental anchor. The results obtained via our pipeline are mathematically identical to those reported in the original FUSE paper, verifying the correctness of our underlying evaluation pipeline.
2. **Naive Ensemble**: A standard majority-voting/unweighted ensembling baseline. Minor performance variations compared to existing literature may exist due to minor discrepancies in granular implementation details.

The respective baseline implementations are available in:
* **`Baseline on HLE`**
* **`Baseline on IMO`**

### Experimental Results

For a deep dive into our comprehensive empirical evaluations, quantitative comparisons, and Ablation Studies, please refer to our detailed documentation:
📊 **`Analysis`**
