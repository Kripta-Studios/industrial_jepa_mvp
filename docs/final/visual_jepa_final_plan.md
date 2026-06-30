# Dense Visual-JEPA: Next-Phase Architectural Blueprint and Roadmap

This document outlines the blueprint and roadmap for rebuilding the visual line of the `industrial_jepa_mvp` project. It addresses the current architectural limits and sets a clear path towards dense, patch-level self-supervision.

## 1. Limitations of the Current Visual-JEPA MVP
The current Visual-JEPA implementation is a simple global MVP that fails to provide competitive results:
* **Global Embedding Bottleneck**: It compresses the entire image (after masking) into a single global vector. Industrial defects are highly localized (e.g., small scratches, cracks, punctures) and their signature is washed out in a global pool.
* **Coarse Heatmaps**: Anomalies are mapped back to pixels through coarse, low-resolution gradients or convolutional feature maps.
* **Underperforming Baselines**: On the MVTec `bottle` category, the current Visual-JEPA reaches an image AUROC of only $\sim 0.67$ and pixel AUROC of $\sim 0.70$, losing to a trivial pixel-statistic baseline (which scores $\sim 0.80$ image AUROC and $\sim 0.86$ pixel AUROC).
* **Missing Structural Elements**: It lacks key self-supervised features like a target encoder updated via Exponential Moving Average (EMA) or position-conditioned mask token predictors.

Continuing to fine-tune this global architecture is obsolete.

## 2. Key Insights from the Literature

### 2.1. I-JEPA (Image Joint Embedding Predictive Architecture)
* **Block Masking**: Instead of masking tiny random pixels, I-JEPA targets relatively large semantic blocks to encourage semantic representation learning.
* **Target and Context Splits**: The target block represents the masked space, and the context block represents the visible space (without target overlap).
* **EMA Target Encoder**: The target encoder does not share parameters directly in a single forward pass; it is updated as a slow Exponential Moving Average (EMA) of the context encoder.
* **Mask Token Predictor**: The predictor is a lightweight transformer that takes visible context embeddings and target positional mask tokens to predict the targets.

### 2.2. V-JEPA 2.1 (Video Joint Embedding Predictive Architecture)
* **Dense Predictive Loss**: Applies losses not only to masked blocks but also to visible context tokens to retain local geometry.
* **Context Proximity Weighting**: Predictor losses on visible tokens are distance-weighted toward masked regions to avoid trivial identity learning.
* **Intermediate Supervision**: Supervises multiple intermediate layers of the encoder, providing robust multi-scale features for downstream tasks.

### 2.3. DINOv3 (Self-Distillation with Multi-scale Anchoring)
* **Global-Local Distillation**: Combines global CLS distillation with local patch-level self-distillation (iBOT).
* **Register Tokens**: Uses dummy tokens to prevent the Vision Transformer (ViT) from packing massive outliers into background patches.
* **Gram Anchoring**: Regularizes patch similarity structures by anchoring student similarity matrices to teacher ones, preventing dense feature degradation during training.

## 3. Mandatory Reference Baselines
No visual anomaly model can be validated without comparison against the following established dense baselines:
1. **DINOv2 / DINOv3 Frozen Features + kNN**: Extracted features from a frozen, pre-trained transformer are a state-of-the-art baseline for zero-shot anomaly detection.
2. **PatchCore (and PatchCore-lite)**: Uses local patch embeddings stored in a memory bank, scoring anomalies based on k-nearest neighbor distance. Memory reduction is achieved via coreset subsampling.
3. **PaDiM (Patch Distribution Modeling)**: Models each patch position across training images as a multivariate Gaussian distribution, scoring anomalies via Mahalanobis distance.

## 4. Re-Architecture: The `DenseVisualJEPA` Design

The proposed architecture shifts from image-level prediction to patch-token prediction:

```
[ Input Image ] ---> Patch Projection (ViT) ---> Context Tokens [B, N_ctx, D]
                                                      |
                                                      v
                                            [ Student Encoder ]
                                                      |
                                                      v
                                            [ Context Embeds ]
                                                      |
                                                      v
    [ Positional Mask Tokens ] ------------> [ Predictor ] <--- (EMA Update)
                                                      |              |
                                                      v              |
                                           [ Predicted Targets ]     |
                                                      |              |
                                            (Loss: MSE / L2)         |
                                                      ^              |
                                                      |              v
[ Original Image ] ---> (Target Block Masking) -> [ Target Encoder (EMA) ]
```

### Key Components:
1. **ViT Patch Tokenizer**: Splits images into $14\times 14$ or $16\times 16$ patches.
2. **Context Encoder (Student)**: Process visible context patches.
3. **Target Encoder (Teacher)**: Process the complete image, updated via EMA ($\alpha = 0.999$).
4. **Predictor**: Conditioned on context embeddings and positional mask tokens corresponding to target blocks.
5. **Loss Function**:
   $$\mathcal{L} = \mathcal{L}_{\text{masked\_target}} + \lambda \mathcal{L}_{\text{context\_distance}}$$
   where $\mathcal{L}_{\text{context\_distance}}$ weights the loss on visible tokens by their inverse distance to target blocks.
6. **Multi-scale Feature Pool**: Anomaly scoring extracts features from the final layer as well as intermediate layers (e.g., layers 8 and 11 in a 12-layer ViT).

## 5. Implementation Roadmap
* **Fase 1: ViT & EMA Boilerplate (Weeks 1-2)**:
  * Implement ViT patch tokenization and positional encoding.
  * Setup EMA teacher update hook.
  * Create block-mask sampler.
* **Fase 2: Dense Predictor & Losses (Weeks 3-4)**:
  * Build the transformer predictor.
  * Implement masked target loss and context-proximity regularizer.
  * Write visualizers for cosine similarities of learned patches.
* **Fase 3: Dense Baselines & Scoring (Weeks 5-6)**:
  * Integrate PatchCore-lite memory-bank and coreset mapping.
  * Implement PaDiM-lite Gaussian models.
  * Set up frozen DINOv2 feature extractor as the target baseline.
* **Fase 4: Benchmark Suite (Weeks 7-8)**:
  * Execute comparative evaluations on MVTec (bottle, cable, transistor), VisA, and KolektorSDD.
  * Plot AUROC/AUPRC scores and verify that DenseVisualJEPA out-performs global baselines.
