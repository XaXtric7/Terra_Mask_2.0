# U-Net vs STEGO Model Comparison Framework

This framework provides a comprehensive comparison between supervised U-Net and self-supervised STEGO models for land cover semantic segmentation.

## Overview

The comparison framework includes:

1. **Supervised U-Net Model**: Your existing trained model using Segmentation Models PyTorch
2. **Self-supervised STEGO Model**: A simplified implementation using ResNet backbone as feature extractor
3. **Comprehensive Evaluation**: Detailed metrics, visualizations, and statistical analysis

## Architecture Comparison

### Supervised U-Net Model

- **Architecture**: U-Net with EfficientNet-B0 encoder
- **Training**: Supervised learning with labeled data
- **Features**: Pre-trained on ImageNet, fine-tuned on land cover data
- **Input**: 256x256 patches
- **Output**: Multi-class segmentation masks

### Self-supervised STEGO Model

- **Architecture**: ResNet-50 backbone + STEGO segmentation head
- **Training**: Self-supervised learning with frozen backbone
- **Features**: Pre-trained ResNet features, learnable segmentation head
- **Input**: 224x224 images
- **Output**: Multi-class segmentation masks

## Files Structure

```
src/
├── models/
│   ├── stego_model.py          # Full STEGO implementation (requires DINO)
│   └── simple_stego.py         # Simplified STEGO implementation
├── train_simple_stego.py       # Training script for STEGO
├── compare_models.py           # Comprehensive comparison script
├── run_comparison.py           # Main orchestration script
└── ... (existing files)
```

## Usage

### Quick Start

Run the complete comparison pipeline:

```bash
cd src
python run_comparison.py
```

### Step-by-Step

1. **Train STEGO model** (if not already trained):

```bash
python train_simple_stego.py
```

2. **Run comparison**:

```bash
python compare_models.py
```

### Command Line Options

```bash
# Skip training phase (use existing models)
python run_comparison.py --skip-training

# Skip comparison phase (only train models)
python run_comparison.py --skip-comparison
```

## Evaluation Metrics

The framework provides comprehensive evaluation including:

### Quantitative Metrics

- **Overall Accuracy**: Pixel-wise accuracy
- **Mean IoU**: Intersection over Union averaged across classes
- **Per-class IoU**: IoU for each land cover class
- **Precision, Recall, F1**: Per-class performance metrics
- **Confusion Matrix**: Detailed classification analysis

### Qualitative Analysis

- **Side-by-side Visualizations**: Original image, ground truth, and both model predictions
- **Error Maps**: Highlighting prediction errors
- **Class-wise Analysis**: Performance breakdown by land cover type

## Output Structure

```
output/
└── model_comparison/
    ├── detailed_comparison.csv      # Per-class metrics
    ├── summary_report.json          # Overall performance summary
    ├── comparison_plots.png         # Visualization plots
    └── *_comparison.png             # Individual image comparisons
```

## Key Findings

Based on the research paper analysis and implementation:

### Expected Results

- **U-Net (Supervised)**: Higher accuracy due to labeled training data
- **STEGO (Self-supervised)**: Lower accuracy but demonstrates feasibility without labels
- **Performance Gap**: U-Net typically outperforms STEGO by 10-20% in accuracy

### Research Insights

1. **Supervised Learning Advantage**: Labeled data provides significant performance boost
2. **Self-supervised Potential**: STEGO shows promise for scenarios with limited labeled data
3. **Feature Quality**: Pre-trained features (ResNet) provide good foundation for segmentation
4. **Domain Adaptation**: Both models can be adapted to different land cover datasets

## Technical Details

### STEGO Implementation

- **Backbone**: ResNet-50 (frozen weights)
- **Head**: 3-layer CNN with batch normalization
- **Training**: Only segmentation head is trained
- **Loss**: Cross-entropy loss
- **Optimizer**: Adam with learning rate 0.001

### Comparison Methodology

- **Same Dataset**: Both models tested on identical test images
- **Same Classes**: Consistent class mapping and evaluation
- **Fair Comparison**: Same preprocessing and post-processing steps
- **Statistical Analysis**: Multiple metrics and visualizations

## Dependencies

Additional dependencies for STEGO model:

```
seaborn==0.12.2
```

## Research Paper Reference

This implementation is based on:

- **Paper**: "Unsupervised Semantic Segmentation by Distilling Feature Correspondences"
- **Authors**: Hamilton et al.
- **Key Innovation**: Self-supervised segmentation using feature correspondences
- **Original**: Uses DINO backbone, simplified to ResNet for accessibility

## Future Improvements

1. **Full DINO Integration**: Implement complete DINO backbone
2. **Advanced STEGO**: Add feature correspondence distillation
3. **More Backbones**: Test with different pre-trained models
4. **Hyperparameter Tuning**: Optimize STEGO architecture
5. **Cross-dataset Evaluation**: Test on multiple land cover datasets

## Troubleshooting

### Common Issues

1. **CUDA Memory Error**: Reduce batch size in training
2. **Model Loading Error**: Ensure models are trained before comparison
3. **Dependency Issues**: Install all requirements from requirements.txt

### Performance Tips

1. **GPU Usage**: Ensure CUDA is available for faster training
2. **Memory Management**: Use appropriate batch sizes
3. **Data Loading**: Increase num_workers for faster data loading

## Citation

If you use this comparison framework, please cite:

```bibtex
@misc{terra_mask_comparison:2025,
  author = {Your Name},
  title = {U-Net vs STEGO: A Comprehensive Comparison for Land Cover Segmentation},
  year = {2025},
  howpublished = {GitHub repository},
  url = {https://github.com/your-repo/terra_mask}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
