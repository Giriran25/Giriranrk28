# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

# Load Iris Dataset
iris = load_iris()
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                    columns=iris['feature_names'] + ['target'])

# Display Dataset Info
print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset Information:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())

# Data Visualization: Pairplot
sns.pairplot(data=data, hue='target', palette='bright', markers=['o', 's', 'D'])
plt.title("Pairplot of Iris Dataset")
plt.show()

# Data Preprocessing
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
print("\nFeature set (X):")
print(X.head())
print("\nTarget set (y):")
print(y.head())

# Splitting Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"\nTraining Set Size: {X_train.shape[0]} | Test Set Size: {X_test.shape[0]}")

# Standardizing Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA for Visualization
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# PCA Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k')
plt.title('PCA of Training Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Target')
plt.show()

# Training Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Model Predictions
y_pred = clf.predict(X_test_scaled)

# Evaluation Metrics
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred):.2f}")

# Feature Importance
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature Importance:")
for f in range(X.shape[1]):
    print(f"{X.columns[indices[f]]}: {importances[indices[f]]:.4f}")

# Visualize Feature Importance
plt.figure(figsize=(8, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()

# Hyperparameter Tuning (Random Forest)
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Best Parameters
print("\nBest Parameters from Grid Search:")
print(grid_search.best_params_)

# Train and Evaluate Model with Best Parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, y_train)
y_best_pred = best_model.predict(X_test_scaled)

print("\nClassification Report (Tuned Model):")
print(classification_report(y_test, y_best_pred))
print(f"Accuracy Score (Tuned Model): {accuracy_score(y_test, y_best_pred):.2f}")

# Save Model
import joblib
joblib.dump(best_model, "best_model_iris.pkl")
print("\nModel saved as 'best_model_iris.pkl'")

# Load Model
loaded_model = joblib.load("best_model_iris.pkl")
print("\nLoaded model evaluation:")
print(f"Loaded Model Accuracy: {loaded_model.score(X_test_scaled, y_test):.2f}")

# Additional Visualizations: Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_best_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix (Heatmap)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
