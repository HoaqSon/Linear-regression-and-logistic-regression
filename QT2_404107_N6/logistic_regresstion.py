import numpy as np
import csv
import matplotlib.pyplot as plt

# Đọc data
X = []
y = []
with open("dataset_logistic.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  
    for row in reader:
        daily_time = float(row[0])
        age = float(row[1])
        income = float(row[2])
        internet_usage = float(row[3])
        male = float(row[4])
        label = int(row[5])   

        X.append([daily_time, age, income, internet_usage, male])
        y.append(label)

X = np.array(X, dtype=float)
y = np.array(y).reshape(-1, 1)

# Train/Test split
m = X.shape[0]
idx = np.arange(m)
np.random.shuffle(idx)
split = int(0.7 * m)
train_idx, test_idx = idx[:split], idx[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Chuẩn hóa
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Loss
def compute_loss(y, y_hat):
    m = y.shape[0]
    eps = 1e-15
    return -(1/m) * np.sum(y*np.log(y_hat+eps) + (1-y)*np.log(1-y_hat+eps))

# Khởi tạo
n_features = X_train.shape[1]
weights = np.zeros((n_features, 1))
bias = 0.0

# Hyperparameters
lr = 0.01
epochs = 2000
loss_history = []

# Training
for epoch in range(epochs):
    z = np.dot(X_train, weights) + bias
    y_hat = sigmoid(z)

    # Gradient descent
    m = X_train.shape[0]
    dw = (1/m) * np.dot(X_train.T, (y_hat - y_train))
    db = (1/m) * np.sum(y_hat - y_train)

    weights -= lr * dw
    bias -= lr * db

    loss = compute_loss(y_train, y_hat)
    loss_history.append(loss)

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Prediction
def predict(X, weights, bias, threshold=0.5):
    return (sigmoid(np.dot(X, weights) + bias) > threshold).astype(int)

y_pred_train = predict(X_train, weights, bias)
y_pred_test = predict(X_test, weights, bias)

# Evaluation
def metrics(y_true, y_pred):
    acc = np.mean(y_true == y_pred)
    tp = np.sum((y_true==1) & (y_pred==1))
    tn = np.sum((y_true==0) & (y_pred==0))
    fp = np.sum((y_true==0) & (y_pred==1))
    fn = np.sum((y_true==1) & (y_pred==0))
    prec = tp / (tp+fp+1e-15)
    rec = tp / (tp+fn+1e-15)
    f1 = 2*prec*rec/(prec+rec+1e-15)
    return acc, prec, rec, f1, (tp,tn,fp,fn)

print("\n--- Train metrics ---")
acc, prec, rec, f1, cm = metrics(y_train, y_pred_train)
print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
print("Confusion matrix (TP, TN, FP, FN):", cm)

print("\n--- Test metrics ---")
acc, prec, rec, f1, cm = metrics(y_test, y_pred_test)
print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
print("Confusion matrix (TP, TN, FP, FN):", cm)

print("\nWeights:", weights.reshape(-1))
print("Bias:", bias)

# Vẽ đồ thị Loss
plt.figure(figsize=(6,4))
plt.plot(loss_history, label="Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over training")
plt.legend()
plt.show()
