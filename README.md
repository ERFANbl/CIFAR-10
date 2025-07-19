# CIFAR-10 Image Classification with PyTorch

This repository contains a PyTorch-based implementation for image classification using the **CIFAR-10** dataset. It includes data loading, preprocessing, model definition, training, and evaluation with performance metrics like accuracy and F1-score.

## 📁 Dataset

The model is trained on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class:

* Training: 50,000 images
* Test: 10,000 images

The dataset is automatically downloaded and extracted.

## 📦 Requirements

* Python 3.8+
* PyTorch
* torchvision
* NumPy
* Matplotlib (for optional visualizations)

Install requirements with:

```bash
pip install torch torchvision numpy matplotlib
```

## 🧠 Model Architecture

The model is a custom Convolutional Neural Network (CNN) with the following structure:

* Several convolutional layers (with ReLU and MaxPool)
* Fully connected layers
* Dropout for regularization
* Softmax output for 10-class classification

## 🚀 Training

The model is trained using CrossEntropyLoss and the **Adam optimizer**. The training loop includes:

* Epoch-wise training with running average of loss
* Evaluation on validation set per epoch
* Logging of accuracy and F1-score

Training loss and validation performance improves steadily:

| Epoch | Val Loss | Train Loss | Val Acc | Val F1 |
| ----- | -------- | ---------- | ------- | ------ |
| 1     | 0.6399   | 0.9346     | 0.7735  | 0.7704 |
| 5     | 0.6033   | 0.8748     | 0.7976  | 0.7945 |
| 10    | 0.5931   | 0.8568     | 0.7989  | 0.7965 |
| 15    | 0.5691   | 0.8353     | 0.8063  | 0.8026 |

Final accuracy: **\~80.6%**, F1-score: **\~80.3%**

## 📊 Evaluation

After training, the model is evaluated on the validation set using:

* Accuracy
* F1-score
* Optionally: confusion matrix or class-wise breakdown

## 📝 Notebook Structure

1. **Data Preparation**
2. **Model Definition**
3. **Training Loop**
4. **Validation & Metrics**
5. **Results Logging**

## 🔮 Future Work

* Try advanced architectures (ResNet, DenseNet)
* Use data augmentation (random crop, flip, normalization)
* Hyperparameter tuning (batch size, learning rate, regularization)
