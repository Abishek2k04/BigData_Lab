import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Step 2: Load the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create a Naive Bayes classifier
model = GaussianNB()

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Predict on the test data
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Naive Bayes classifier: {accuracy:.2f}")

# Plotting decision boundaries and data points
plt.figure(figsize=(8, 6))

# Scatter plot for true labels
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Set1, marker='o', label='True Labels', alpha=0.6)

# Scatter plot for predicted labels
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.Set2, marker='x', label='Predicted Labels', alpha=0.6)

plt.title("Naive Bayes Classification")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.colorbar(label='Species')
plt.grid(True)
plt.show()

# Print sample predictions
print("\nSample Predictions:")
for _ in range(5):
    print(f"Predicted: {iris.target_names[y_pred[i]]}, Actual: {iris.target_names[y_test[i]]}")
