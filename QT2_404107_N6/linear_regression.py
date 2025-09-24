import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Đọc data

df = pd.read_csv("dataset_linear.csv")
feature_names = ["Hours Studied", "Previous Scores", "Sleep Hours", "Sample Question Papers Practiced"]
X_raw = df[feature_names].values 
y_raw = df["Performance Index"].values.reshape(-1, 1)
m = X_raw.shape[0]

# Chuẩn hóa feature

mu = X_raw.mean(axis=0)      # mean mỗi cột (n,)
sigma = X_raw.std(axis=0, ddof=0)  # std mỗi cột (n,)
sigma_nozero = np.where(sigma == 0, 1, sigma)
X_norm = (X_raw - mu) / sigma_nozero
X = np.c_[np.ones((m, 1)), X_norm]  # m x (n+1)
y = y_raw

# Hàm cost, gradient_descent

def compute_cost(X, y, theta):
    m = len(y)
    preds = X.dot(theta)
    error = preds - y
    cost = (1.0 / (2*m)) * np.sum(error ** 2)
    return cost

def gradient_descent(X, y, theta, alpha, num_iters, verbose=False):
    m = len(y)
    cost_history = []
    for it in range(num_iters):
        preds = X.dot(theta)                 # m x 1
        error = preds - y                    # m x 1
        grad = (1.0/m) * (X.T.dot(error))    # (n+1) x 1
        theta = theta - alpha * grad
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        if verbose and (it % (num_iters//10 + 1) == 0):
            print(f"Iter {it}/{num_iters} - cost: {cost:.6f}")
    return theta, cost_history

# Khởi tạo & huấn luyện

n_plus1 = X.shape[1]
theta_init = np.zeros((n_plus1, 1))

alpha = 0.01        # learning rate
num_iters = 400     # epoch

theta, cost_history = gradient_descent(X, y, theta_init, alpha, num_iters, verbose=True)

print("Learned theta (including bias):")
print(theta.flatten())
print("Final cost:", cost_history[-1])


# Vẽ cost history

plt.figure()
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost J(theta)")
plt.title("Gradient Descent Convergence")
plt.grid(True)
plt.show()

# Đánh giá trên toàn bộ dataset
# Tạo hàm predict dùng theta và transform ngược data

def predict(X_raw_new, theta, mu, sigma_nozero):
    # X_raw_new: shape (k, n) hoặc (n,) cho 1 mẫu
    x_arr = np.array(X_raw_new, ndmin=2)
    x_norm = (x_arr - mu) / sigma_nozero
    x_with_bias = np.c_[np.ones((x_norm.shape[0], 1)), x_norm]
    return x_with_bias.dot(theta)   # trả về (k,1)

# Dự đoán cho dữ liệu gốc
y_pred = predict(X_raw, theta, mu, sigma_nozero)

# Tính MSE, RMSE, R^2
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res/ss_tot

print("MSE:", mse(y, y_pred))           # Mean Squared Error
print("RMSE:", rmse(y, y_pred))         # Root Mean Squared Error
print("R2:", r2_score(y, y_pred))       # Coefficient of Determination

# Plot Actual vs Predicted
plt.figure()
plt.scatter(y, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Performance Index")
plt.ylabel("Predicted Performance Index")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()

# Dự đoán 
sample = [4, 82, 4, 2]  # [Hours Studied, Previous Scores, Sleep Hours, Papers]
pred_sample = predict(sample, theta, mu, sigma_nozero)
print("Predicted Performance Index for sample:", float(pred_sample))