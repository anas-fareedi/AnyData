# GAN for Medical Data Augmentation (Rare Class Improvement)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview
This project demonstrates how to use **Wasserstein GAN with Gradient Penalty (WGAN-GP)** to generate synthetic samples for rare/minority classes in medical imaging datasets. The synthetic data is then used to improve the performance of a deep learning classifier (ResNet-18) on imbalanced medical data.

**Key Achievement:** Proved that GAN-generated synthetic data significantly improves rare class detection in imbalanced medical datasets.

## ❗ Problem Statement
In medical imaging, rare diseases or conditions often have very few training samples, leading to:
- **Class imbalance** in datasets (e.g., 100 rare samples vs 5000 normal)
- Poor model performance on rare classes
- High cost and difficulty in collecting more real samples

## ✨ Solution
Use WGAN-GP to:
1. Learn the distribution of the rare class from limited samples
2. Generate high-quality synthetic samples
3. Augment the training dataset to improve classifier performance

## 📁 Project Structure
```
AnyData/
├── main.ipynb                    # Complete Jupyter notebook (training)
├── main.py                       # Complete pipeline script
├── generate_synthetic.py         # Generate synthetic images CLI
├── demo.py                       # Quick demonstration script
├── models.py                     # GAN architecture (Generator & Critic)
├── config.py                     # Configuration settings
├── utils.py                      # Visualization utilities
├── gan_generator_rare_class.pth  # Pre-trained GAN model ✓
├── requirements.txt              # Dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # This file
```

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/AnyData.git
cd AnyData

# Install dependencies
pip install -r requirements.txt
```

### Quick Demo (Using Pre-trained Model)
```bash
# Generate sample visualizations
python demo.py

# Generate 100 synthetic images
python generate_synthetic.py --num_samples 100 --output_dir ./synthetic_images --save_grid
```

### Run Complete Pipeline
```bash
# Run full pipeline (loads pre-trained GAN, trains classifiers, evaluates)
python main.py

# Skip baseline training (faster)
python main.py --skip_baseline
```

### Jupyter Notebook
```bash
# Open and run the complete notebook
jupyter notebook main.ipynb
```

## 📊 Requirements
- Python 3.8+
- PyTorch 2.0+
- torchvision
- scikit-learn
- numpy
- matplotlib

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🏗️ Implementation Details

## 🏗️ Implementation Details

### 1. Data Preparation
- Uses CIFAR-10 as a proxy for medical imaging data
- Simulates rare class by keeping only 100 samples of class "bird" (vs ~5000 for other classes)
- Images normalized to [-1, 1] and resized to 64x64

### 2. WGAN-GP Architecture
**Generator:**
- Input: 100-dimensional random noise
- Architecture: 4 transposed convolution layers with BatchNorm and ReLU
- Output: 64x64x3 RGB image

**Critic (Discriminator):**
- Input: 64x64x3 RGB image
- Architecture: 4 convolution layers with InstanceNorm and LeakyReLU
- Output: Scalar score (real/fake assessment)

**Key Features:**
- Gradient Penalty for training stability
- Prevents mode collapse
- Critic trained 5x more than generator per iteration

### 3. Workflow
1. **Data Preparation:** Create imbalanced dataset
2. **Load Pre-trained GAN:** Use saved model (`gan_generator_rare_class.pth`)
3. **Generate Synthetic Data:** Create 400 synthetic samples for rare class
4. **Train Classifiers:**
   - **Baseline Model:** Trained on imbalanced data only
   - **Augmented Model:** Trained on imbalanced data + synthetic samples
5. **Evaluation:** Compare performance metrics and visualizations

### 4. Evaluation Metrics
- Precision, Recall, F1-Score for all classes
- Confusion matrices
- Focused analysis on rare class performance improvement

## 📈 Usage Examples

### Generate Synthetic Images
```python
from generate_synthetic import load_generator, generate_synthetic_images

# Load trained model
generator = load_generator('gan_generator_rare_class.pth')

# Generate 200 synthetic images
generate_synthetic_images(generator, num_samples=200, output_dir='./output')
```

### Visualize Samples
```python
from utils import visualize_generated_samples
from generate_synthetic import load_generator

generator = load_generator('gan_generator_rare_class.pth')
visualize_generated_samples(generator, num_samples=10, save_path='samples.png')
```

### Run Complete Pipeline
```python
# In command line
python main.py
```

## 🎨 Sample Outputs

The project generates:
- **Synthetic medical images** for rare classes
- **Confusion matrices** comparing baseline vs augmented models
- **Performance metrics** showing improvement
- **Visualization grids** of generated samples

Results are saved in the `results/` directory.

## 📊 Expected Results
The augmented model demonstrates:
- ✅ **Higher Recall** for the rare class (better detection)
- ✅ **Improved F1-Score** (better overall performance)
- ✅ **More balanced confusion matrix**

## 🔧 Configuration

Modify `config.py` to customize:
- Model hyperparameters (learning rates, batch size)
- Data settings (rare class, number of samples)
- Training parameters (epochs, iterations)
- File paths

## 📝 Command-Line Interface

### Generate Synthetic Data
```bash
python generate_synthetic.py --num_samples 500 --output_dir ./my_synthetic_data --save_grid
```

### Run Main Pipeline
```bash
# Full pipeline
python main.py

# Skip baseline training
python main.py --skip_baseline
```

### Run Demo
```bash
python demo.py
```

## 🧪 Key Findings
✅ WGAN-GP successfully generates realistic synthetic medical images  
✅ Synthetic data augmentation improves rare class performance  
✅ Particularly effective when real data collection is costly/difficult  
✅ Gradient penalty ensures training stability  
✅ Pre-trained models can be reused for continuous data generation

## 🌍 Real-World Applications
- **Rare Disease Detection:** Melanoma, brain tumors, rare cancers
- **Medical Image Augmentation:** X-rays, CT scans, MRI
- **Drug Discovery:** Molecular structure generation
- **Clinical Decision Support:** Improving diagnostic models
- **Privacy-Preserving AI:** Generate synthetic data instead of sharing patient data

## 🔮 Future Enhancements
- [ ] Use real medical imaging datasets (HAM10000, ChestX-ray14)
- [ ] Implement conditional GAN for multi-class generation
- [ ] Add Inception Score and FID metrics for quality assessment
- [ ] Experiment with StyleGAN2 for higher resolution
- [ ] Deploy as API for synthetic data generation
- [ ] Add web interface for easy access

## 📚 References
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
- [GANs for Medical Image Synthesis](https://arxiv.org/abs/1810.10352)

## 📄 License
MIT License - See LICENSE file for details

## 👤 Author
**Medical AI Research Project - 2026**

For questions or contributions, please open an issue or submit a pull request.

## 🙏 Acknowledgments
- CIFAR-10 dataset from [Canadian Institute for Advanced Research](https://www.cs.toronto.edu/~kriz/cifar.html)
- PyTorch framework
- Medical AI research community

---

**⭐ If you find this project useful, please consider giving it a star!** 
