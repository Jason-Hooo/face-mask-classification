# Face Mask Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13fvln9eDDOxi-MM0HW2jN5qUhr39I9-N?usp=sharing)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace-FFD21E)](https://jasonhoooooo-face-mask-classification.hf.space)

An end-to-end deep learning project implementing EfficientNet for face mask classification.
Rather than a standard fine-tuning exercise, this repository focuses on core AI fundamentals and software engineering practices:

- **Architecture from Scratch**: Deliberately built from the ground up to deeply understand EfficientNet's internal scaling, bypassing high-level wrapper APIs.
- **Modular Engineering**: Refactored from an experimental Colab notebook into a clean, modular Python codebase to ensure maintainability and reproducibility.

## Demo

![Face Mask Classification Demo](images/face-mask-classification-demo.gif)

## Features

- **4-Class Classification**: Detects the following mask-wearing states:
  - Mask on chin
  - Mask not covering nose
  - Mask properly worn
  - No mask
- **EfficientNet Architecture**: Implements EfficientNet-B0 to B7 variants with custom scaling
- **CLI Training**: Command-line interface for model training with configurable hyperparameters
- **Gradio Web Interface**: Interactive web app for real-time mask classification

## Project Structure

```
face-mask-classification/
├── data/                  # Example images
├── images/                # Demo images and GIFs
├── notebooks/
│   └── face_mask_classification_training.ipynb  # Colab notebook
├── src/
│   ├── __init__.py
│   ├── data_setup.py      # Data preparation utilities
│   ├── dataset.py         # Dataset class and data transforms
│   ├── model.py           # EfficientNet model architecture
│   └── train.py           # Training pipeline CLI
├── weights/               # Trained model weights
├── app.py                 # Gradio web interface
├── LICENSE                # MIT License
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face-mask-classification.git
cd face-mask-classification
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Train the model using the CLI interface:

```bash
python src/train.py --dataset_root Face_Mask_Dataset \
                    --batch_size 32 \
                    --epochs 25 \
                    --learning_rate 1e-3 \
                    --model_name B1 \
                    --save_path weights/trained_model_parameters.pth
```

**Available Arguments:**
- `--dataset_root`: Path to dataset directory (default: `Face_Mask_Dataset`)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 25)
- `--learning_rate`: Learning rate for optimizer (default: 1e-3)
- `--model_name`: EfficientNet variant - B0 to B7 (default: B1)
- `--save_path`: Path to save trained model weights (default: `weights/trained_model_parameters.pth`)
- `--seed`: Random seed for reproducibility (default: 87)

### Running the Web Interface

Launch the Gradio web app for interactive classification:

```bash
python app.py
```

The web interface will be available at `http://127.0.0.1:7860`

**Online Demo**: [![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace-FFD21E)](https://jasonhoooooo-face-mask-classification.hf.space)

## Colab Training

You can also train the model using the Google Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13fvln9eDDOxi-MM0HW2jN5qUhr39I9-N?usp=sharing)

## Model Architecture

This project implements EfficientNet from scratch with the following components:

- **ConvBnSiLUBlock**: Standard convolution block with BatchNorm and SiLU activation
- **SEBlock**: Squeeze-and-Excitation block for channel-wise feature recalibration
- **MBConvBlock**: Mobile Inverted Bottleneck Convolution with optional residual connection
- **EfficientNet**: Main model with dynamic scaling (B0-B7 variants)

**Reference**: This implementation is based on the original EfficientNet paper by Mingxing Tan and Quoc V. Le (2019): [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

## Dataset

The model is trained on a face mask dataset aggregated and cleaned from multiple Kaggle sources:

- [Face Mask Detection](https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection)
- [Face Mask Detector](https://www.kaggle.com/datasets/spandanpatnaik09/face-mask-detectormask-not-mask-incorrect-mask)
- [Face Mask Dataset](https://www.kaggle.com/datasets/shiekhburhan/face-mask-dataset)
- [Face Mask Dataset TVT](https://www.kaggle.com/datasets/busrabetulcavusoglu/face-mask-dataset-tvt)

These datasets were manually integrated, classified, and cleaned to create the final training and test set. You can download the final dataset here: [Google Drive: Face_Mask_Dataset.zip](https://drive.google.com/file/d/1dbgt3_jVqyQ9uB3f59eZOGCVmSNd-Itj/view?usp=share_link)

**Directory Structure:**

```
Face_Mask_Dataset/
├── test/
│   ├── incorrect_mask_mc/
│   ├── incorrect_mask_mmc/
│   ├── with_mask/
│   └── without_mask/
└── train/
    ├── incorrect_mask_mc/
    ├── incorrect_mask_mmc/
    ├── with_mask/
    └── without_mask/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
