MobileNetV2-MSCA-PFR
Multi-Scale Context Aggregation and Pooling-Refinement Network for Skin Disease Classification
Publication: Under Review / 2025

Project Overview
This repository contains the official PyTorch implementation of:

"A Novel Multi-Scale Context Aggregator and Pooling Feature Refinement Network for Medical Image Classification"

The goal of this work is to develop a lightweight yet highly accurate deep learning model for skin disease classification using 4 limited datasets.  
The model introduces two key components on top of the MobileNetV2 backbone:

1. Multi-Scale Context Aggregator (MSCA)  
2. Pooling Feature Refinement (PFR) block  

These modules significantly enhance feature representation while maintaining computational efficiency suitable for real-time and resource-constrained environments.

The model addresses limitations of:
Traditional CNNs (limited receptive field and insufficient context modeling)
Transformer-based models (high computational cost and memory usage on small datasets)

Proposed Architecture
The MobileNetV2-MSCA-PFR architecture integrates three main components:

MobileNetV2 Backbone
Uses pretrained MobileNetV2 feature extractor
Efficient depthwise separable convolutions
Lightweight and suitable for portable devices

Multi-Scale Context Aggregator (MSCA)
Parallel feature extraction branches
1×1 convolution
3×3 convolution
5×5 convolution
Global context pooling branch
Outputs a fused multi-scale enriched representation

Pooling Feature Refinement (PFR)
Inspired by PoolFormer token-mixing
Applies spatial pooling interactions
Refines local–global dependencies
Enhances discriminative regions before classification

Classification Head
Global Average Pooling
Fully Connected layers
Softmax output (4 classes)

Datasets Used
The model was trained and evaluated on multiple skin-disease datasets, including:

MSID Dataset (Monkeypox Skin Image Dataset)
Containing four classes:
Chickenpox
Measles
Monkeypox
Normal skin

Additional cross-domain datasets evaluated:
Mendeley Data: Monkeypox Dataset (2024)
https://data.mendeley.com/datasets/r9bfpnvyxr/6

Kidney Stone Detection Dataset (2025)
https://github.com/yildirimozal/Kidney_stone_detection

Retinal OCT Images Dataset (Kermany et al.)
https://www.kaggle.com/datasets/paultimothymooney/kermany2018

Input size: 224×224  
Data augmentation strategies were used to address dataset imbalance.

Training Configuration
Framework: PyTorch  
Optimizer: Adam  
Learning rate: 0.0001  
Batch size: 32  
Epochs: 60–100 (with early stopping)  
Loss function: CrossEntropyLoss  
Backbone: MobileNetV2 (40–50% layers frozen)  
Modules added: MSCA + PFR  

Performance Results
On MSID and cross-domain datasets, the MobileNetV2-MSCA-PFR model demonstrated:

High sensitivity to discriminative lesion patterns  
Strong generalization ability on unseen datasets  
Low computational complexity suitable for deployment  

Explainability
Grad-CAM visualizations highlight lesion regions contributing to final predictions.  
Saliency maps reveal pixel-level importance, improving model interpretability for clinical use.

Repository Structure
```
MobileNetV2_MSCAPFR/
├── train.py                        Training pipeline
├── evaluate.py                     Evaluation pipeline
├── requirements.txt                Python dependencies
│
├── models/
│   ├── msca.py                     Multi-Scale Context Aggregator (MSCA)
│   ├── pfr_block.py                Pooling Feature Refinement block (PFR)
│   ├── mobilenetv2_msca_pfr.py     Full model architecture
│
├── utils/
│   ├── dataset_loader.py           Custom dataset loader
│   ├── metrics.py                  Evaluation metrics
│   ├── plot_tools.py               Plotting utilities
│   └── paths.py                    Configuration for dataset paths
│
├── visualization/
│   ├── gradcam.py                  Grad-CAM generation
│   └── saliency.py                 Saliency map computation
│
└── README.md
---
Proposed Model
(Insert figure here, or model diagram once available)

Citation
If you use this work, please cite:

Your Name, et al.  
"A Novel Multi-Scale Context Aggregator and Pooling Feature Refinement Network for Medical Image Classification."  
2025.

Notes
This repository is intended for academic and research purposes.  
The implementation follows the methodology described in the manuscript, including the MSCA module, PFR block, training procedures, evaluation metrics, and interpretability tools.
