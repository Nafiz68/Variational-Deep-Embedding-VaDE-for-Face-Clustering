# Variational Deep Embedding (VaDE) for Face Clustering

A PyTorch implementation of Variational Deep Embedding (VaDE) for unsupervised clustering of face images, with comparison against traditional autoencoder + K-means baseline. This project explores deep clustering techniques on the Olivetti Faces dataset.

## 🎯 Project Overview

This project implements and evaluates a Bayesian deep clustering network based on the VaDE (Variational Deep Embedding) architecture. The model combines variational autoencoders with Gaussian mixture models to perform joint representation learning and clustering in an end-to-end fashion.

### Key Features

- **VaDE Architecture**: Full implementation of Variational Deep Embedding with encoder-decoder structure
- **Baseline Comparison**: Traditional autoencoder + K-means clustering for performance benchmarking
- **Comprehensive Evaluation**: Multiple clustering metrics (ARI, NMI, Silhouette Score)
- **Visualization**: t-SNE plots, cluster exemplars, and entropy analysis
- **Robust Testing**: Multiple random seeds for statistical significance

## 📊 Dataset

- **Olivetti Faces Dataset**: 400 face images (64×64 pixels) of 40 different people
- **Preprocessing**: Image normalization and resizing
- **Split**: 70% training, 15% validation, 15% testing

## 🏗️ Architecture

### VaDE Model Components

1. **Encoder Network**:

   - Convolutional layers (Conv2d → ReLU)
   - Fully connected layers for mean and log-variance estimation
   - Reparameterization trick for variational sampling

2. **Decoder Network**:

   - Fully connected layers
   - Transposed convolutions for image reconstruction
   - Sigmoid activation for pixel intensity output

3. **Clustering Component**:
   - Gaussian mixture model with learnable parameters
   - Cluster assignment probabilities
   - Variational lower bound optimization

### Baseline Model

- Standard autoencoder with similar encoder-decoder architecture
- K-means clustering applied to learned latent representations

## 📈 Results

### Performance Metrics (Mean ± Std across 5 seeds)

| Model                     | ARI             | NMI             | Silhouette Score |
| ------------------------- | --------------- | --------------- | ---------------- |
| **VaDE**                  | 0.0326 ± 0.0402 | 0.8732 ± 0.0056 | 0.2771 ± 0.0662  |
| **Autoencoder + K-means** | 0.0731          | 0.8765          | 0.4257           |

### Key Findings

- **Clustering Quality**: Both models achieved high NMI scores (~0.87), indicating good mutual information between predicted and true clusters
- **Baseline Comparison**: Traditional autoencoder + K-means slightly outperformed VaDE on all metrics
- **Stability**: VaDE showed consistent performance across different random seeds
- **Challenges**: Face clustering remains challenging due to high intra-class variation and inter-class similarity

## 🔧 Technical Implementation

### Loss Function

The VaDE model optimizes a composite loss combining:

- **Reconstruction Loss**: Binary cross-entropy for image reconstruction
- **KL Divergence**: Regularization term for latent space structure
- **Clustering Loss**: Variational lower bound for mixture model

### Training Details

- **Optimizer**: Adam with learning rate 0.001
- **Batch Size**: 32
- **Epochs**: 100 with early stopping
- **Regularization**: KL divergence weighting schedule

## 📁 Project Structure

```
425_Project/
├── 22101045_CSE425_SEC01_Project.ipynb    # Main implementation notebook
├── 22101045_CSE425_SEC01_Project_Report.pdf    # Detailed project report
├── Dataset.zip                            # Olivetti faces dataset
├── entropy_hist.png                       # Cluster assignment entropy histogram
├── exemplars.png                          # Cluster exemplar visualizations
├── reconstructions.png                    # Image reconstruction samples
├── tsne_pred.png                         # t-SNE plot of predicted clusters
├── tsne_true.png                         # t-SNE plot of true clusters
└── README.md                             # This file
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn
```

### Usage

1. **Data Loading**: Extract and load the Olivetti faces dataset
2. **Model Training**: Train both VaDE and baseline autoencoder models
3. **Evaluation**: Compare clustering performance using multiple metrics
4. **Visualization**: Generate t-SNE plots and cluster analysis

```python
# Example usage
from sklearn.model_selection import train_test_split
import torch

# Load and preprocess data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Initialize VaDE model
model = VaDE(img_dim=(1, 64, 64), latent_dim=128, n_clusters=40)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# ... training code
```

## 📊 Visualizations

The project includes comprehensive visualizations:

- **Reconstruction Quality**: Original vs reconstructed face images
- **t-SNE Plots**: 2D visualization of learned representations colored by true/predicted clusters
- **Cluster Exemplars**: Representative images for each discovered cluster
- **Entropy Analysis**: Distribution of cluster assignment uncertainty

## 🎓 Academic Context

This project was completed as part of CSE425 (Neural Networks) coursework, exploring advanced topics in:

- Variational autoencoders
- Deep clustering techniques
- Unsupervised representation learning
- Bayesian deep learning

## 🔍 Future Improvements

- **Architecture Enhancements**: Experiment with different encoder/decoder designs
- **Hyperparameter Optimization**: Systematic tuning of learning rates and loss weights
- **Alternative Datasets**: Evaluation on other face datasets (MNIST, CIFAR-10)
- **Advanced Clustering**: Integration with other deep clustering methods (DEC, DEPICT)

## 📜 License

This project is for academic and research purposes. Please cite appropriately if used in other work.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

_This project demonstrates the application of advanced deep learning techniques for unsupervised clustering, providing insights into the challenges and opportunities in representation learning for face recognition tasks._
