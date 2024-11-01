import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, make_scorer
from KNNWrapper import CustomKNNWrapper
import time
from statsmodels.nonparametric.smoothers_lowess import lowess
from KNN import gaussian_kernel, CustomKNN

def lowess(x_train, y_train, array, best_params):
    knn = CustomKNN(
        n_neighbors=best_params['n_neighbors'],
        metric=best_params['metric'], 
        kernel=best_params['kernel'],
        window_type=best_params['window_type'],
        window_size=best_params['window_size'],
        p=best_params['p'],
        a=best_params['a'],
        b=best_params['b']
    )
    
    for i in range(len(x_train)):
        x_train_temp = np.delete(x_train, i, axis=0)
        y_train_temp = np.delete(y_train, i)

        knn.fit(x_train_temp, y_train_temp)

        prediction = knn.predict(x_train[i].reshape(1, -1))[0]
        
        weight = gaussian_kernel(y_train[i] - prediction)
        array.append(weight)

def train():

    start_time = time.time() 

    data = pd.read_csv('vehicles_data.csv')

    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].astype('float32')

    le = LabelEncoder()
    data['Company'] = le.fit_transform(data['Company'])

    class_counts = data['Company'].value_counts()
    classes_to_keep = class_counts[class_counts >= 10].index
    data = data[data['Company'].isin(classes_to_keep)]

    X = data.drop('Company', axis=1).values
    y = data['Company'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

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

    scorer = make_scorer(accuracy_score)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=CustomKNNWrapper(),
        param_distributions=param_dist,
        scoring=scorer,
        cv=skf,
        n_iter=20,
        verbose=3,
        random_state=42,
        error_score='raise',
        n_jobs=7
    )

    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    print("Лучшие гиперпараметры:", best_params)
    print(f"Точность модели на тренировочной выборке: {random_search.best_score_:.2f}")

    weights = []
    lowess(X_train, y_train, weights, best_params)
    weights = np.array(weights)

    knn_model_no_weights = CustomKNNWrapper(**best_params)
    knn_model_no_weights.fit(X_train, y_train)
    test_accuracy_no_weights = knn_model_no_weights.score(X_test, y_test)
    print(f"Точность на тестовом множестве без взвешивания: {test_accuracy_no_weights:.2f}")

    knn_model_with_weights = CustomKNNWrapper(**best_params)
    knn_model_with_weights.fit(X_train, y_train, sample_weight=weights)

    test_accuracy_with_weights = knn_model_with_weights.score(X_test, y_test)
    print(f"Точность на тестовом множестве со взвешиванием: {test_accuracy_with_weights:.2f}")

    end_time = time.time() 
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Время выполнения программы: {int(hours)} часов, {int(minutes)} минут, {seconds:.2f} секунд")\

    best_params = random_search.best_params_

    return best_params, X_train, y_train, X_test, y_test

train()