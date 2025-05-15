'''
This script implements a simple feedforward neural network for binary classification using numpy.
It includes functions for forward and backward propagation, custom loss functions, and training using mini-batches.
The model is trained on a dataset from a bank marketing campaign, predicting whether a client will subscribe to a term deposit.
The dataset is preprocessed by dropping unnecessary categorical variables and one-hot encoding the remaining ones.
The model's performance is evaluated using accuracy and confusion matrix metrics, and the loss and accuracy curves are plotted.
'''
# Import necessary libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Activation function: Sigmoid
def sigmoid(t):
    return 1.0/(1 + np.exp(-t))

# Custom loss function for binary classification
def loss(y, z):
    # y: true labels, z: model output (pre-sigmoid)
    loss = np.mean(-y*(mylog(z)) - (1-y)*mylogminus(z))
    return loss

# Numerically stable log(sigmoid(t))
def mylog(t):
    y = 0*t
    m = t.shape[0]
    for i in range(m):
        if t[i] < 0:
            y[i] = t[i] - np.log(1 + np.exp(t[i]))
        else:
            y[i] = -np.log(1 + np.exp(-t[i]))
    return y

# Numerically stable log(1-sigmoid(t))
def mylogminus(t):
    y = 0*t
    m = t.shape[0]
    for i in range(m):
        if t[i] < 0:
            y[i] = -np.log(1 + np.exp(t[i]))
        else:
            y[i] = -t[i] - np.log(1 + np.exp(-t[i]))
    return y

# ReLU activation function
def ReLU(t):
    return np.maximum(t, 0)

# Derivative of ReLU for backpropagation
def ReLUprime(t):
    return np.sign(np.maximum(t, 0))

# Predict class labels from model output
def predict(output):
    # output: model output (pre-sigmoid)
    # Returns: array of 0s and 1s
    pred_class = [1 if i > 0 else 0 for i in output]
    return np.array(pred_class)

# Compute accuracy metric
def accuracy(y, y_predict):
    accuracy = np.sum(y == y_predict) / len(y)
    return accuracy

# Compute confusion matrix values
def confusion_matrix(y, y_predict):
    tp = np.sum((1-y)*(1-y_predict))
    fp = np.sum(y*(1-y_predict))
    tn = np.sum(y*y_predict)
    fn = np.sum((1-y)*y_predict)
    return tp, fp, fn, tn

# Forward pass of the neural network
def model(w, b, W_one, b_one, X):
    # X: input data, w: output weights, b: output bias
    # W_one: hidden layer weights, b_one: hidden layer bias
    H = ReLU((np.matmul(X, W_one.T) + b_one).T)
    z = np.dot(w, H) + b
    return z

# Compute gradients for backpropagation
def gradients(w, b, W_one, b_one, X, y):
    N = X.shape[0]
    # Forward pass
    A = (np.matmul(X, W_one.T) + b_one).T
    H = ReLU(A)
    z = np.dot(w, H) + b
    y_hat = sigmoid(z)
    # Backward pass
    r = y_hat - y
    dw = (1/N)*np.dot(H, r)
    db = (1/N)*np.sum(r)
    db_one = 0*b_one
    dW_one = 0*W_one
    for i in range(N):
        R = r[i]*w
        R = R*ReLUprime(A[:,i])
        db_one += R
        dW_one += np.outer(R, X[i,:])
    db_one *= 1/N
    dW_one *= 1/N
    return z, dw, db, dW_one, db_one

# Mini-batch training loop
def minibatch_train(w, b, W_one, b_one, X, y, epoch, lr, bs):
    N, n = X.shape
    losses = []
    acc = []
    for e in range(epoch):
        # Shuffle data each epoch
        permutation = np.random.permutation(N)
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]
        for j in range(0, N, bs):
            # Mini-batch selection
            X_batch = X_shuffled[j:j+bs]
            y_batch = y_shuffled[j:j+bs]
            # Forward and backward pass
            z, dw, db, dW_one, db_one = gradients(w, b, W_one, b_one, X_batch, y_batch)
            # Parameter update
            w -= lr * dw
            b -= lr * db
            W_one -= lr * dW_one
            b_one -= lr * db_one
            # Track loss and accuracy
            l = loss(y_batch, z)
            losses.append(l)
            accuracy_val = accuracy(np.squeeze(y_batch), predict(np.squeeze(z)))
            acc.append(accuracy_val)
        # Print epoch summary
        print(f"Epoch {e+1}/{epoch} - Loss: {np.mean(losses[-N//bs:])} - Accuracy: {np.mean(acc[-N//bs:])}")
    return w, b, W_one, b_one, losses, acc

# --- Data Preparation ---

# Load dataset
url = 'https://raw.githubusercontent.com/madmashup/targeted-marketing-predictive-engine/master/banking.csv'
data = pd.read_csv(url)

# Drop categorical variables not used
cat_vars=['default','education','contact','month','day_of_week',]
data=data.drop(cat_vars, axis=1)

# One-hot encode remaining categorical variables
cat_vars=['job','marital','housing','loan','poutcome']
for va in cat_vars:
    cat_list = pd.get_dummies(data[va])
    data1=pd.concat([data,cat_list], axis=1)
    data=data1.drop(va, axis=1)

# Separate features and label
X = data.loc[:, data.columns != 'y']
y = data.loc[:, data.columns == 'y']

# Normalize features
X_normalized = (X - X.mean()) / X.std()
X = X_normalized

columns = X.columns
X = X.to_numpy()
y = y.to_numpy()

# Print class distribution
print('# of class 1 cases =', np.sum(y))
print('# of class 0 cases =', np.sum(1-y))

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)

# Print shapes for verification
print(y_train.shape)
print(y_test.shape)
print(columns)

# --- Model Initialization and Training ---

l_one = 100  # Number of hidden layer units
lr = 0.05    # Learning rate
n = X_train.shape[1]  # Number of input features

# Initialize weights and biases
w = np.random.randn(l_one)
b = 0
W_one = np.random.randn(l_one, X_train.shape[1])
b_one = 0 * np.random.rand(l_one)

# Train the neural network
w, b, W_one, b_one, loss, acc = minibatch_train(
    w, b, W_one, b_one, X_train, y_train, epoch=30, lr=lr, bs=128
)

# Plot loss and accuracy curves
plt.figure()
plt.plot(loss)
plt.title('Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.savefig('loss_curve.png')
plt.figure()
plt.plot(acc)
plt.title('Accuracy Curve')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.savefig('accuracy_curve.png')
plt.savefig('accuracy_curve.png')