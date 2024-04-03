import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Loading data
iris = load_iris()

# Shuffling records
np.random.seed(2021)
np.random.shuffle(iris.data)
np.random.shuffle(iris.target)

# Converting data to a DataFrame
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['label'] = iris.target

# Saving the label names
features_dict = {k: v for k, v in enumerate(iris.target_names)}
df_iris['label_names'] = df_iris.label.apply(lambda x: features_dict[x])

# Data normalization
scaler = StandardScaler()

# Selecting numerical features (assuming that columns 0 to 3 are numerical)
numerical_features = df_iris.iloc[:, :4]
df_iris_norm = pd.DataFrame(scaler.fit_transform(numerical_features), columns = numerical_features.columns)
# Combine the normalized features with the original "label" column
df_iris_norm['label'] = df_iris['label']

# Splitting the data into 70% training and 30% test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_iris_norm.drop('label', axis=1), df_iris_norm['label'], test_size=0.3, random_state=2021)

# Training a KNN classifier with different values of K
from sklearn.neighbors import KNeighborsClassifier

# Defining the range of K values
k_range = list(range(1, 31))

# Creating a dictionary to store the results
scores = {}

for k in k_range:
  # Training the KNN classifier
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train, y_train)

  # Evaluation of accuracy on the test sample
  score = knn.score(X_test, y_test)

  # Storing the results
  scores[k] = score

# Search for the best value of K
k_best = max(scores, key=scores.get)
score_best = scores[k_best]

print('Best k value:', k_best)
print('Classification accuracy with the best k:', score_best)

# Visualize the results
import matplotlib.pyplot as plt

plt.plot(k_range, scores.values())
plt.xlabel('Number of nearest neighbors (k)')
plt.ylabel('Classification accuracy')
plt.show()
