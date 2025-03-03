# Multi-Scale ViT for 3D Medical Imaging

A PyTorch implementation of a Multi-Scale Vision Transformer (ViT) for 3D medical image classification. This model utilizes self-attention and multi-scale feature extraction to analyze volumetric medical images using the OrganMNIST3D dataset.

## Features
- 3D Vision Transformer (ViT) for volumetric medical imaging
- Multi-Scale Attention Mechanism for feature extraction
- OrganMNIST3D dataset from MedMNIST
- Sinusoidal Positional Embeddings for 3D spatial encoding
- Patch-Based Image Tokenization
- GPU Acceleration (CUDA) for efficient training

## Dataset
This project uses the OrganMNIST3D dataset from MedMNIST, which consists of 3D grayscale volumes of 11 organ classes.

### Download the Dataset
The dataset is automatically downloaded using `medmnist`:

```python
from medmnist.dataset import OrganMNIST3D
train_dataset = OrganMNIST3D(split="train", download=True)
```

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/Multi-Scale-ViT-3D.git](https://github.com/sarathir-dev/Multi-Scale-ViT-for-3D-Medical-Imaging.git)
cd Multi-Scale-ViT-3D
pip install -r requirements.txt
```

## Model Architecture
The model follows the Vision Transformer (ViT) architecture, adapted for 3D data:

- Patch Embedding: Converts 3D volumes into smaller patches
- Multi-Head Self-Attention (MHSA): Extracts global features
- Feed-Forward Networks (FFN): Enhances representation learning
- Classification Head: Outputs predictions for 11 organ classes

## Training

Run the training script:

```bash
python training/train.py
```

## Evaluation

To evaluate the model on the test dataset:

```bash
python training/test.py
```

## Results
The model is trained for 10 epochs using CrossEntropyLoss and the Adam optimizer. Performance is measured using accuracy on the test set.

## Contributing
Contributions are welcome. Please open an issue or pull request to improve the project.

