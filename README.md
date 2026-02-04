# Human Activity Recognition using ANN (MLP)

## Overview
This project builds and evaluates a Neural Network to classify human activities 
using smartphone sensor data (accelerometer & gyroscope).

The goal is to predict activities like:
- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

based purely on numerical sensor features.

All experiments and code are inside: `main.ipynb`

---

## Dataset
Dataset used: UCI Human Activity Recognition (HAR)

- Source: Smartphone sensors
- Samples:
  - Train: 7352
  - Test: 2947
- Features: 561 numerical features
- Classes: 6 activities

Each row represents one time window of sensor readings.

---

## Project Pipeline

1. Load train & test data
2. Check for missing values
3. Split features and labels
4. Train classical ML baseline (Random Forest)
5. Train ANN (MLP)
6. Evaluate on unseen test data
7. Analyze errors with confusion matrix
8. Experiment with deeper ANN + dropout

---

## Data Preprocessing

- No missing values found
- Features already normalized
- Labels encoded using LabelEncoder
- Converted to one-hot vectors for ANN

---

## Baseline Model (Classical ML)

Model: Random Forest  
Trees: 100  

Test Accuracy: 92.6%

---

## ANN v1 (Simple MLP)

Architecture:
- Input: 561 neurons
- Hidden layer: 128 neurons (ReLU)
- Output: 6 neurons (Softmax)

Optimizer: Adam  
Loss: Categorical Crossentropy  
Epochs: 30  

Test Accuracy: 93.07%

### Model Architecture
<img width="770" height="315" alt="image" src="https://github.com/user-attachments/assets/f775a869-df02-4acd-a4f2-cf4cb263d65f" />


---

## ANN v2 (Deeper + Dropout)

Architecture:
- Dense(256) → ReLU
- Dropout(0.3)
- Dense(128) → ReLU
- Output → Softmax

Test Accuracy: 92.67%

Result:
- Overfitting reduced
- But slight drop in accuracy

Demonstrates bias–variance tradeoff.

---

## Confusion Matrix Analysis

Main confusions:
- Sitting vs Standing
- Walking Upstairs vs Walking Downstairs

These errors make sense physically since sensor patterns are similar.

<img width="539" height="432" alt="confusion_matrix" src="https://github.com/user-attachments/assets/3d012e15-c4c9-46b9-a5bb-629bf195d6a7" />

---

## Learning Curves

Shows training dynamics and mild overfitting.

<img width="577" height="432" alt="learning_curve_loss_vs_epoch" src="https://github.com/user-attachments/assets/0273ff85-0e70-47b9-9b01-3b775acd98f2" />
<img width="577" height="432" alt="learning_curve_accuracy_vs_epoch" src="https://github.com/user-attachments/assets/132c92d5-6636-4dbd-a54c-13ab8d66548d" />

---

## Key Learnings

- ANN can outperform classical ML on non-linear sensor data
- Regularization (dropout) reduces overfitting but may reduce accuracy
- Proper evaluation is more important than chasing higher numbers
- Data leakage must be avoided at all costs

---

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib, Seaborn

---

## Future Work

- Use raw time-series with LSTM / GRU
- Convert signals to spectrograms and apply CNN
- Deploy as Streamlit web app

---

## Author
**Mayank Jindal**

Built as a learning project to deeply understand:
- ANN training
- Model comparison
- Overfitting & regularization
- Real ML evaluation practices

