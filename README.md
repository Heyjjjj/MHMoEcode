## [Morphology-Aware Hierarchical MoE for Chest X-ray Anatomy Segmentation]
## Introduction
MHMoE is a novel deep learning architecture specifically designed for medical image segmentation, with a focus on chest X-ray (CXR) anatomy segmentation. This repository provides the official implementation of our paper "Morphology-aware Hierarchical MoE for Chest X-ray Anatomy Segmentation", which introduces a groundbreaking approach to address the unique challenges in medical image analysis.

## Prepare Datasets
You can refer to the  [https://vindr.ai/ribcxr]

Expected Structure
```bash
Data_Annotations/
├── train.txt          # Training file list
├── test.txt           # Testing file list  
├── img/               # Input images
│   ├── image1.png
│   └── image2.png
└── Annotations/       # Ground truth masks
    ├── 0/
    │   └── image1.png
    ├── 1/
    └── 2/
```


## Requirements
pip install -r requirements.txt

## Training
python train.py 
## Testing&Evaluation
python predict.py
