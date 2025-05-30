# 🍎 CNN Fruits Classification

A deep learning project for classifying fruits using Convolutional Neural Networks (CNN) trained on the Fruits-360 dataset.

## 📋 Overview

This project implements an improved CNN architecture for fruit classification with data augmentation and enhanced training techniques. The model can classify over 200 different types of fruits with high accuracy.

## 🎯 Features

- **Improved CNN Architecture**: Enhanced model with batch normalization, dropout, and deeper layers
- **Data Augmentation**: Random crops, flips, rotations, and color jittering for better generalization
- **Advanced Training**: Learning rate scheduling, label smoothing, and validation tracking
- **Easy Prediction**: Simple script to predict fruit classes from images
- **Google Colab Support**: Complete script for training on Google Colab with GPU acceleration

## 🏗️ Project Structure

```
CNN-Fruits/
├── model.py                 # Basic CNN architecture
├── model_improved.py        # Enhanced CNN with better performance
├── dataset.py              # Basic data loading
├── dataset_improved.py     # Data loading with augmentation
├── train.py                # Basic training script
├── train_improved.py       # Enhanced training with validation
├── evaluate.py             # Model evaluation and metrics
├── predict.py              # Image prediction script
├── main.py                 # Basic training pipeline
├── main_improved.py        # Enhanced training pipeline
├── requirements.txt        # Python dependencies
├── colab_training.py       # Complete Google Colab script
└── README.md              # This file
```

## 🚀 Quick Start

### Local Training

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Basic training:**
   ```bash
   python main.py
   ```

3. **Improved training:**
   ```bash
   python main_improved.py
   ```

4. **Make predictions:**
   ```bash
   python predict.py your_fruit_image.jpg
   ```

### Google Colab Training (Recommended)

1. Upload `colab_training.py` to Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Upload your `data.zip` file
4. Run the script

## 📊 Model Architectures

### Basic CNN
- 3 convolutional blocks
- Simple dropout and fully connected layers
- ~1M parameters

### Improved CNN
- 4 convolutional blocks with batch normalization
- Progressive dropout and regularization
- Adaptive global average pooling
- ~15M parameters

## 🎛️ Training Features

### Data Augmentation
- Random horizontal flips
- Random rotations (±15°)
- Random crops
- Color jittering
- Proper normalization

### Training Enhancements
- AdamW optimizer with weight decay
- Learning rate scheduling
- Label smoothing
- Early stopping
- Validation tracking

## 📈 Performance

The improved model achieves significantly better accuracy compared to the basic version:

- **Basic CNN**: ~85% validation accuracy
- **Improved CNN**: ~95%+ validation accuracy

## 🖼️ Usage Examples

### Training
```python
from model_improved import ImprovedCNN
from train_improved import train_improved_model

# Create model
model = ImprovedCNN(num_classes=201)

# Train
train_improved_model(model, train_loader, test_loader, device, epochs=25)
```

### Prediction
```python
from predict import predict_image

# Predict a single image
predict_image("apple.jpg")
```

### Expected output:
```
🔍 Prediction Results:
📸 Image: apple.jpg
🍎 Predicted Class: Apple Golden 1
📊 Confidence: 95.23%
```

## 📦 Requirements

- Python 3.7+
- PyTorch 2.0+
- torchvision
- scikit-learn
- matplotlib
- seaborn
- tqdm
- Pillow

## 🎓 Dataset

This project uses the [Fruits-360 dataset](https://www.kaggle.com/moltean/fruits) which contains:
- 90,000+ images of fruits and vegetables
- 201 classes
- 100x100 pixel images
- Training and test splits

## 🚀 Training on Google Colab

For faster training with GPU acceleration:

1. **Open Google Colab**
2. **Enable GPU**: Runtime → Change runtime type → GPU
3. **Upload the complete Colab script**
4. **Upload your dataset**
5. **Run the script**

Expected training time:
- **CPU**: 4-6 hours
- **GPU (Colab)**: 30-60 minutes

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

If you have any questions or suggestions, please open an issue on GitHub.

---

**Happy fruit classification! 🍊🍌🍇** 