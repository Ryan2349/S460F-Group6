from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import random
import joblib  # For saving the model

# Paths to your training and test data files
train_data_path = './data/libsvm/train(LDA).txt'
test_data_path = './data/libsvm/test(LDA).txt'

# Function to load pre-vectorized data
def load_data(data_file):
    # Load the data in libSVM format
    X, y = load_svmlight_file(data_file)
    return X, y

# Load the pre-vectorized training and testing data
X_train, y_train = load_data(train_data_path)
X_test, y_test = load_data(test_data_path)

# Check if the number of features is different between train and test
if X_train.shape[1] != X_test.shape[1]:
    # Find the difference in feature dimensions
    feature_diff = X_train.shape[1] - X_test.shape[1]
    
    if feature_diff > 0:
        # Add zero columns to X_test to match the number of features in X_train
        zero_padding = np.zeros((X_test.shape[0], feature_diff))
        X_test = hstack([X_test, zero_padding])
    else:
        raise ValueError("Test data has more features than training data, which should not happen.")

# Generate a random seed for reproducibility but with randomness
random_seed = random.randint(1, 10000)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_seed)

# Define a pipeline with a scaler (important for SVM) and the SVC model
pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),  # with_mean=False because of sparse data
    ('svc', SVC())  # SVC for kernel-based SVMs, such as RBF or polynomial
])

# Define the hyperparameter grid for tuning
param_grid = {
    'svc__C': [0.1, 1, 10, 100],  # Regularization strength
    'svc__gamma': [0.001, 0.01, 0.1, 1],  # Kernel coefficient for RBF and polynomial kernels
    'svc__kernel': ['rbf', 'poly'],  # RBF and polynomial kernels
    'svc__degree': [2, 3, 4]  # Degree of the polynomial kernel (if using poly kernel)
}

# Define a cross-validation strategy with shuffling and a random seed
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)

# Set up GridSearchCV for hyperparameter tuning and cross-validation
grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=1)

# Fit the model with hyperparameter tuning on the training data
grid_search.fit(X_train, y_train)

# Get the best estimator from the grid search
best_model = grid_search.best_estimator_

# Print best hyperparameters and corresponding validation accuracy
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_}")

# Test the model on the validation set
y_val_pred = best_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy}")

# Test the model on the test set
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy}")

# Save the best model to a file
C_param = grid_search.best_params_['svc__C']
gamma_param = grid_search.best_params_['svc__gamma']
kernel_param = grid_search.best_params_['svc__kernel']
model_filename = f'kernel_svm_c{C_param}_g{gamma_param}_{kernel_param}.pkl'
joblib.dump(best_model, model_filename)

print(f"Best model saved as {model_filename}")

# Save the GridSearchCV results to a CSV file
results_df = pd.DataFrame(grid_search.cv_results_)
results_df.to_csv('kernel_svm_hyperparameter_results.csv', index=False)

print(f"All hyperparameter tuning results saved to kernel_svm_hyperparameter_results.csv")