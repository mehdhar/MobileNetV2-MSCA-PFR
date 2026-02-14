# MobileNetV2-MSCA-PFR  
Multi-Scale Context Aggregation and Pooling-Feature Refinement Network for Skin Disease Classification  
Publication: 2025  

---

 1. Project Overview

This repository contains the official PyTorch implementation of:

"MobileNetV2-MSCA-PFR: A Multi-Scale Context Aggregation and Pooling-Feature Refinement Network for Skin Disease Classification"

The goal of this work is to develop a lightweight yet highly accurate deep learning model for skin lesion classification, capable of capturing both local and global dependencies while maintaining computational efficiency for real-time medical applications.

The model addresses limitations of:  
- Traditional CNNs (limited contextual understanding)  
- Transformer-based models (high computational overhead)

---

2. Proposed Architecture

MobileNetV2-MSCA-PFR integrates three main components:

1. MobileNetV2 Backbone  
   - Retains general low-level feature extraction  
   - Avoids heavy deeper layers  
   - Reduces computational cost  

2. Multi-Scale Context Aggregator (MSCA)  
   - Enhances contextual understanding across different spatial scales  
   - Strengthens semantic feature representation  

3. Pooling-Feature Refinement (PFR) Block  
   - Pooling-based attention  
   - Feature refinement and enhancement  
   - Produces more discriminative lesion representations  

---

3. Dataset  
   Datasets Used

The experiments in this study were conducted using the following publicly available datasets:

1. Monkeypox Skin Images Dataset (MSID), Version 6  
   D. Bala, M. S. Hossain  
   https://data.mendeley.com/datasets/r9bfpnvyxr/6  

2. Monkeypox Skin Lesion Dataset  
   S. N. Ali, Kaggle, 2022  
   https://www.kaggle.com/datasets/nafin59/monkeypox-skin-lesion-dataset  

3. Mpox Skin Lesion Dataset (Version 2.0)  
   J. Paul, Kaggle, 2023  
   https://www.kaggle.com  

4. Additional custom dataset  
   Supplementary real-world images collected for model robustness  

The combined dataset includes:

- Monkeypox  
- Chickenpox  
- Measles  
- Normal skin  

Input size: 224×224  
Data augmentation was applied to reduce class imbalance.

---

4. Training Configuration

- Framework: PyTorch  
- Optimizer: Adam  
- Learning rate: 0.0001  
- Batch size: 32  
- Epochs: 60–100 (with early stopping)  
- Loss function: CrossEntropyLoss  

---

5. Performance Results

The MobileNetV2-MSCA-PFR model demonstrated:

- High classification accuracy  
- Strong generalization across mixed datasets  
- Low computational complexity  
- Stable learning across all classes  

---

6. Additional Evaluation

The model was further evaluated on mixed-source skin lesion datasets.  
It maintained high reliability and robustness across varying image conditions, supporting its potential real-world applicability.

---

7. Grad-CAM Interpretability

Grad-CAM visualizations were applied to highlight lesion-related regions.  
The results confirm that the model focuses on meaningful clinical features such as lesion boundaries, texture, and inflamed areas.

---

## Repository Structure

```
MobileNetV2_MSCAPFR/
├── train.py                        # Training pipeline
├── evaluate.py                     # Evaluation script
├── requirements.txt                # Python dependencies
│
├── models/
│   ├── __init__.py
│   ├── msca.py                     # Multi-scale context aggregation module
│   ├── pfr_block.py                # Pooling-feature refinement module
│   └── mobilenetv2_msca_pfr.py     # Full integrated architecture
│
├── utils/
│   ├── dataset_loader.py           # Dataset preparation and loaders
│   ├── metrics.py                  # Metrics and evaluation tools
│   ├── plot_tools.py               # Plotting utilities
│   └── paths.py                    # Path configuration
│
├── visualization/
│   ├── gradcam.py                  # Grad-CAM heatmap generation
│   └── saliency.py                 # Saliency map visualization
│
└── README.md
```

<img width="667" height="278" alt="Proposed model" src="https://github.com/user-attachments/assets/900f6a5c-d232-4521-be28-580cb3df8a0f" />

---

9. Citation

If you use this work, please cite:
Al-Gaashani, M. S., Mahel, A. S. B., Khayyat, M. M., & Muthanna, A. (2026). A novel multi-scale context aggregation and feature pooling network for Mpox classification. Biomedical Signal Processing and Control, 111, 108254.

---

10. Notes

This repository is intended for research and academic purposes.

The implementation follows the methodology described in the published paper and includes the core architecture, training pipeline, evaluation metrics, and interpretability tools.
