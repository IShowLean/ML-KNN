import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from KNNWrapper import CustomKNNWrapper
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from training import train

best_params, X_train, y_train, X_test, y_test = train()

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

param_dist  = {
    'n_neighbors': np.arange(1, 15, 2),
    'metric': ['euclidean', 'minkowski', 'cosine'],
    'kernel': ['uniform', 'gaussian', 'triangular'],
    'window_type': ['fixed', 'variable'],
    'window_size': np.linspace(0.1, 2.0, 10),
    'p': [1, 2, 3],
    'a': [1, 2, 3],
    'b': [1, 2, 3]
}

random_search = RandomizedSearchCV(
    estimator=CustomKNNWrapper(),
    param_distributions=param_dist,
    scoring=None,
    cv=skf,
    n_iter=20,
    verbose=3,
    random_state=42,
    error_score='raise',
    n_jobs=7
)

fixed_params = {key: best_params[key] for key in best_params if key != 'n_neighbors'}

n_neighbors_values = np.arange(1, 21)
train_scores = []
test_scores = []

for n_neighbors in n_neighbors_values:
    knn_model = CustomKNNWrapper(n_neighbors=n_neighbors, **fixed_params)
    
    cv_train_scores = cross_val_score(knn_model, X_train, y_train, cv=5, scoring='accuracy')
    train_scores.append(np.mean(cv_train_scores))
    
    knn_model.fit(X_train, y_train)
    test_accuracy = knn_model.score(X_test, y_test)
    test_scores.append(test_accuracy)

plt.figure(figsize=(10, 6))
plt.plot(n_neighbors_values, train_scores, label='Train Accuracy (CV)', marker='o')
plt.plot(n_neighbors_values, test_scores, label='Test Accuracy', marker='s')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Neighbors')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_vs_n_neighbors.png')
plt.show()

fixed_params = {key: best_params[key] for key in best_params if key != 'window_size'}

window_sizes = np.linspace(0.1, 2.0, 10)
train_scores = []
test_scores = []


for window_size in window_sizes:
    knn_model = CustomKNNWrapper(window_size=window_size, **fixed_params)
    
    cv_train_scores = cross_val_score(knn_model, X_train, y_train, cv=5, scoring='accuracy')
    train_scores.append(np.mean(cv_train_scores))

    knn_model.fit(X_train, y_train)
    test_accuracy = knn_model.score(X_test, y_test)
    test_scores.append(test_accuracy)

plt.figure(figsize=(10, 6))
plt.plot(window_sizes, train_scores, label='Train Accuracy', marker='o')
plt.plot(window_sizes, test_scores, label='Test Accuracy', marker='s')
plt.xlabel('Window Size')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Window Size')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_vs_window_size.png')
plt.show()

knn_model_best = CustomKNNWrapper(**best_params)
knn_model_best.fit(X_train, y_train)
y_pred_best = knn_model_best.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted classes')
plt.ylabel('True classes')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png') 
plt.show()
