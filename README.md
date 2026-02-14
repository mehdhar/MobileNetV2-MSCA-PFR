MobileNetV2-MSCA-PFR

Multi-Scale Context Aggregation and Pooling-Feature Refinement Network for Skin Disease Classification
Publication: 2025

Project Overview

This repository contains the official PyTorch implementation of:

"MobileNetV2-MSCA-PFR: A Multi-Scale Context Aggregation and Pooling-Feature Refinement Network for Medical Skin Disease Classification"

The goal of this work is to develop a lightweight yet powerful deep learning model capable of accurately classifying skin diseases while maintaining computational efficiency suitable for real-time clinical applications.

The model addresses limitations of:

Traditional CNNs (limited contextual representation)

Transformer-based models (high computational overhead)

Proposed Architecture

MobileNetV2-MSCA-PFR integrates three main components:

MobileNetV2 Backbone

Serves as the base feature extractor

Preserves efficiency while providing robust low-level representations

Multi-Scale Context Aggregator (MSCA)

Enhances contextual understanding across multiple receptive fields

Strengthens feature representation in complex medical images

Pooling-Feature Refinement (PFR)

Uses pooling-based attention to refine and emphasize discriminative lesion features

Supports improved interpretability and classification stability

Dataset
Datasets Used

The experiments in this study were conducted using the following publicly available datasets:

Monkeypox Skin Images Dataset (MSID)
D. Bala, M. S. Hossain, Version 6
Mendeley Data
https://data.mendeley.com/datasets/r9bfpnvyxr/6

Monkeypox Skin Lesion Dataset
S. N. Ali, Kaggle, 2022
https://www.kaggle.com/datasets/nafin59/monkeypox-skin-lesion-dataset

Mpox Skin Lesion Dataset (Version 2.0)
J. Paul, Kaggle, 2023
https://www.kaggle.com

Custom Dataset
Additional images collected to improve class diversity and balance

The datasets include categories such as:

Monkeypox

Chickenpox

Measles

Normal skin

Input size: 224 × 224
Data augmentation techniques were applied to handle class imbalance and improve generalization.

Training Configuration

Framework: PyTorch

Optimizer: Adam

Learning rate: 0.0001

Batch size: 32

Epochs: 60–100 with early stopping

Loss function: CrossEntropyLoss

Partial freezing of backbone layers to stabilize training

Performance Results

The proposed MobileNetV2-MSCA-PFR model demonstrated:

Strong classification accuracy

Excellent cross-dataset generalization

Improved contextual understanding

Lower computational cost compared to heavier architectures

(Insert numerical results here.)

Additional Evaluation

The model was further validated across multiple datasets with varying imaging conditions.
Its performance remained stable, confirming robustness and suitability for real-world medical use.

Grad-CAM Interpretability

Grad-CAM visualization highlights clinically relevant lesion regions.
This confirms that the model’s attention aligns well with dermatological features such as textures, color changes, and lesion boundaries.

Repository Structure
```
MobileNetV2_MSCAPFR/
├── train.py                        # Training pipeline
├── evaluate.py                     # Evaluation and metrics
├── requirements.txt                # Python dependencies
│
├── models/
│   ├── msca.py                     # Multi-scale context aggregation module
│   ├── pfr_block.py                # Pooling-feature refinement block
│   └── mobilenetv2_msca_pfr.py     # Integrated full architecture
│
├── utils/
│   ├── dataset_loader.py           # Dataset loader
│   ├── metrics.py                  # Evaluation metrics
│   ├── plot_tools.py               # Visualization tools
│   └── paths.py                    # Path configuration
│
├── visualization/
│   ├── gradcam.py                  # Grad-CAM generation
│   └── saliency.py                 # Saliency map visualization
│
└── README.md
```

Citation

If you use this work, please cite:

Your Name, et al.
"MobileNetV2-MSCA-PFR: A Multi-Scale Context Aggregation and Pooling-Feature Refinement Network for Medical Skin Disease Classification."
2025.

Notes

This repository is intended for academic and research use.
The implementation follows the methodology described in the manuscript and includes the full architecture, training pipeline, evaluation suite, and visualization tools.
