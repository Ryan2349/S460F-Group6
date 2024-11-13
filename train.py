from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import joblib  # For saving the model

# Path to your training data file
train_data_path = './data/libsvm/train.txt'

# Function to preprocess training data
def preprocess_train_data(train_file):
    with open(train_file, 'r') as f:
        train_data = f.readlines()
    
    # Create vectorizer and fit on training data
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_data)
    
    # Extract labels 
    y_train = [int(line.split()[0]) for line in train_data]
    
    return X_train, y_train, vectorizer

# Preprocess training data
X_train, y_train, vectorizer = preprocess_train_data(train_data_path)

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf']
}

# Initialize SVM
svc = SVC()

# Grid Search with Cross-Validation
grid_search = GridSearchCV(svc, param_grid, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Output best model parameters
print("Best Parameters:", grid_search.best_params_)

# Save the best model and vectorizer
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'best_svm_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Save results for reference
results = pd.DataFrame(grid_search.cv_results_)
results.to_csv('grid_search_results.csv', index=False)