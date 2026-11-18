# ConvNeXt-CSR
Official implementation of ConvNeXt-CSR, a density-supervised deep learning framework for simultaneous crowd counting and head localization without a point-level localization loss.
Overview

ConvNeXt-CSR is a density-supervised deep learning framework designed for simultaneous crowd counting and head localization without requiring a point-level localization loss during training.

The framework integrates:

ConvNeXt-Tiny backbone for hierarchical visual feature extraction.

Two-scale Feature Pyramid Network (FPN) for multiscale feature fusion.

Dual Convolutional Block Attention Module (CBAM) for channel and spatial feature refinement.

Density prediction head for spatial density estimation.

Geometry-adaptive Gaussian density supervision.

Peak-based decoding for head localization.

SGDR learning-rate scheduling and Exponential Moving Average (EMA) for training and model-weight stabilization.

Architecture

The proposed framework follows a density-supervised learning pipeline:

Input Image → ConvNeXt-Tiny → Two-Scale FPN → Dual CBAM → Density Prediction Head → Predicted Density Map.

Two outputs are obtained from the predicted density map:

Crowd Counting: Integration of the predicted density values.

Head Localization: Extraction of local maxima to estimate individual head coordinates.


Citation
If you find the code and dataset useful, please cite our paper.
@article{aonty2026convnext,
  title={ConvNeXt-CSR: A density-supervised deep framework for simultaneous crowd counting and head localization without a point-level localization loss},
  author={Aonty, Shuhena Salam and Deb, Kaushik},
  journal={Array},
  pages={101472},
  year={2026},
  publisher={Elsevier}
}
