import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Generate a random data set
np.random.seed(2021)
X = np.random.rand(1000, 2)

# Defining a color map
cmap_bold = ListedColormap(['blue','#FFFF00','black','green'])

# Data normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test samples
X_train, X_test, y_train, y_test = train_test_split(X_scaled, np.arange(1000), test_size=0.25, random_state=42)

# Training a KNN regressor with different values of K
k_values = list(range(1, 21))
scores = []

for k in k_values:
  knn = KNeighborsRegressor(n_neighbors=k)
  knn.fit(X_train, y_train)
  score = knn.score(X_test, y_test)
  scores.append(score)

# Selecting the value of K for the best performance
best_k = k_values[scores.index(max(scores))]

# Estimation of KNN regressor with optimal K
best_score = scores[best_k - 1]

# Visualization of results
plt.plot(k_values, scores)
plt.xlabel("Кількість найближчих сусідів (K)")
plt.ylabel("Точність регресії")
plt.show()

# Prediction using a KNN regressor with optimal K
knn_best = KNeighborsRegressor(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred = knn_best.predict(X_test)

# Visualization of forecasts
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='o', s=30, cmap='viridis')
plt.plot(X_test[:, 0], y_pred, color='red')
plt.show()

print(f"The best value of K: {best_k}")
print(f"The accuracy of the regression with K={best_k}: {best_score:.2%}")
