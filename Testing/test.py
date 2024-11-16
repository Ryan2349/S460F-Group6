from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_svmlight_file
import joblib  # For loading the model
from scipy.sparse import hstack, csr_matrix
import numpy as np

# Path to your test data file
test_data_path = './data/libsvm/test(LDA).txt'

# Load the trained model
model = joblib.load('kernel_svm_c10_g1_poly.pkl')

# Function to load pre-vectorized libSVM data
def load_test_data(test_file):
    # Load the test data in libSVM format
    X_test, y_test = load_svmlight_file(test_file)
    
    return X_test, y_test

# Preprocess test data
X_test, y_test = load_test_data(test_data_path)

# Get the number of features the model was trained on
n_features_train = model.shape_fit_[1]

# Check if the number of features in the test data matches the training data
if X_test.shape[1] < n_features_train:
    # If test data has fewer features, pad with zeros
    padding = csr_matrix((X_test.shape[0], n_features_train - X_test.shape[1]))
    X_test = hstack([X_test, padding])
elif X_test.shape[1] > n_features_train:
    # If test data has more features, truncate extras (this is rare but possible)
    X_test = X_test[:, :n_features_train]

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Test Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, predictions))

# Save predictions to a file
with open('prediction_result.txt', 'w') as f:
    for prediction in predictions:
        f.write(f"{prediction:.0f}\n")