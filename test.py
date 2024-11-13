from sklearn.metrics import accuracy_score, classification_report
import joblib  # For loading the model

# Path to your test data file
test_data_path = './data/libsvm/test.txt'

# Function to preprocess test data
def preprocess_test_data(test_file, vectorizer):
    with open(test_file, 'r') as f:
        test_data = f.readlines()
    
    # Transform test data using the fitted vectorizer
    X_test = vectorizer.transform(test_data)
    
    # Extract labels
    y_test = [int(line.split()[0]) for line in test_data]
    
    return X_test, y_test

# Load the trained model and vectorizer
best_model = joblib.load('best_svm_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocess test data
X_test, y_test = preprocess_test_data(test_data_path, vectorizer)

# Evaluate the model
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Test Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, predictions))

# Save predictions to a file
with open('prediction_result.txt', 'w') as f:
    for prediction in predictions:
        f.write(f"{prediction}\n")