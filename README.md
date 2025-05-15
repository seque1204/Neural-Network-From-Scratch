# 🧠 Feedforward Neural Network from Scratch — Bank Marketing Classification

This project implements a **simple feedforward neural network for binary classification** using only `numpy`, without any high-level machine learning frameworks. It is trained on a real-world **bank marketing dataset** to predict whether a client will subscribe to a term deposit.

---

## 🚀 Overview

- 📊 **Goal**: Predict client subscription outcome (`yes`/`no`) based on features like job, marital status, loan history, and more.
- 🔨 **Method**: One hidden layer neural network built from scratch with support for:
  - Forward & backward propagation
  - Custom binary cross-entropy loss
  - Mini-batch gradient descent
- 📈 **Outputs**:
  - Loss and accuracy plots
  - Accuracy and confusion matrix metrics

---

## 📁 Dataset

- 📌 Source: [Bank Marketing Data](https://raw.githubusercontent.com/madmashup/targeted-marketing-predictive-engine/master/banking.csv)
- 🧹 Preprocessing:
  - Dropped unused categorical variables (e.g., `month`, `contact`)
  - Applied one-hot encoding to remaining categorical fields
  - Standardized all numerical features

---

## 🧠 Model Architecture

- **Input Layer**: ~50 one-hot encoded & numerical features  
- **Hidden Layer**: 100 ReLU-activated neurons  
- **Output Layer**: Single neuron with sigmoid activation  
- **Loss Function**: Binary Cross-Entropy (numerically stable implementation)  
- **Optimizer**: Manual implementation of mini-batch gradient descent

---

## 📦 Requirements
Install dependencies using pip:
pip install numpy pandas scikit-learn matplotlib

---

## 🧪 Running the Code
Simply run:
python your_script.py

## ✍️ Author Notes
This project demonstrates how to build and train a basic neural network without any deep learning libraries, ideal for learning purposes and understanding the inner workings of backpropagation and mini-batch optimization.

---

## Contact

For questions or collaboration opportunities, feel free to reach out via [GitHub Issues](../../issues) or [LinkedIn](https://www.linkedin.com/in/josequeira).
