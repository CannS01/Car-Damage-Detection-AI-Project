# Car-Damage-Dataset-AI-Project

This repository contains the implementation of a deep learning-based multitask classification system for detecting and categorizing vehicle damage. The model predicts both:

- Damage Type: Normal / Crushed / Breakage
- View Orientation: Front / Rear

This work was developed as part of the SE3508 Introduction to Artificial Intelligence course.

## Dataset

The dataset is sourced from Kaggle: `samwash94/comprehensive-car-damage-detection`.

Initially, it consists of six folders corresponding to six combined labels:

- front_crushed, front_breakage, front_normal
- rear_crushed, rear_breakage, rear_normal

### Multitask Label Conversion

To improve generalization and simplify model design, the classification problem was reformulated into a multitask learning structure with two independent targets:

- View Orientation: Front / Rear
- Damage Type: Normal / Crushed / Breakage

This approach allows the model to learn both tasks concurrently and promotes knowledge sharing between the damage and view classification components.

## Model Architectures

Two separate convolutional neural network models were implemented and compared.

### ResNet50

- Pretrained ResNet50 backbone (from torchvision)
- Dual output heads:
  - Damage head: 2048 → 256 → 3
  - View head: 2048 → 128 → 2
- Includes ReLU activation, dropout, and fully connected layers

### DenseNet121

- Pretrained DenseNet121 backbone
- Dual output heads:
  - Damage head: 1024 → 256 → 3
  - View head: 1024 → 128 → 2
- Demonstrated better generalization performance and was integrated into the final UI system

## Training Configuration

- Image Size: 128x128
- Optimizer: Adam
- Learning Rate: 1e-4 (ResNet50), 5e-5 (DenseNet121)
- Batch Size: 32
- Loss Function: CrossEntropyLoss (for both outputs)
- Scheduler: StepLR (step size = 3, gamma = 0.5)
- Early Stopping: Patience = 3
- Epochs: 30

### Data Augmentation

Applied during training to reduce overfitting:

- RandomHorizontalFlip
- RandomRotation (±10 degrees)
- ColorJitter (brightness, contrast, saturation)
- Resizing to 128x128
- Normalization

Validation set was not augmented.

## Evaluation and Results

Classification performance was measured using accuracy, precision, recall, and F1-score.

| Model       | Damage Accuracy | View Accuracy |
|-------------|------------------|----------------|
| ResNet50    | 74%              | 95%            |
| DenseNet121 | 93%              | 98%            |

Confusion matrices and classification reports were generated for both tasks. DenseNet121 outperformed ResNet50 and was used in the final system deployment.

### Visual Outputs

- Classification reports
- Confusion matrices (damage and view)
- Training and validation loss curves

## File Structure

project/
├── dataset/ # Raw image folders (6 classes)
├── model/ # Network definitions
│ ├── multitask_resnet50.py
│ └── multitask_densenet121.py
├── utils/
│ └── dataset.py # Custom Dataset class
├── train.py # Training loop
├── evaluate_model.py # Evaluation script
├── plot_confusion.py # Confusion matrix and loss plot
├── report.tex # LaTeX report file
├── multitask_resnet50.pth # Saved ResNet50 model
├── multitask_densenet121.pth # Saved DenseNet121 model
└── README.md

## Models and .pth Files drive links

https://drive.google.com/file/d/1xwwhOy2Mpm_2mr6HzUwcVLd-88WIytAI/view?usp=sharing
https://drive.google.com/file/d/11aChmV5i5NeHvT-UnqJN0t5zsP8AkqM8/view?usp=sharing



