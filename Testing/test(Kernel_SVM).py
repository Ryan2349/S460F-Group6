from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_svmlight_file
import joblib  # For loading the model
from scipy.sparse import hstack, csr_matrix
import numpy as np

# Path to your test data file
test_data_path = './data/libsvm/test(LDA).txt'

# Load the trained model (which is a pipeline)
model = joblib.load('kernel_svm_c0.1_g1_poly.pkl')

# Function to load pre-vectorized libSVM data
def load_test_data(test_file):
    # Load the test data in libSVM format
    X_test, y_test = load_svmlight_file(test_file)
    return X_test, y_test

# Preprocess test data
X_test, y_test = load_test_data(test_data_path)

# Evaluate the model directly
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Test Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, predictions))

# Save predictions to a file
with open('prediction_result.txt', 'w') as f:
    for prediction in predictions:
        f.write(f"{prediction:.0f}\n")