from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths to your data files
train_data_path = './data/libsvm/train.txt'
test_data_path = './data/libsvm/test.txt'

# Combine train and test data for consistent feature extraction
def preprocess_data(train_file, test_file):
    with open(train_file, 'r') as f:
        train_data = f.readlines()
    with open(test_file, 'r') as f:
        test_data = f.readlines()
    
    combined_data = train_data + test_data
    
    # Create vectorizer and fit on combined data
    vectorizer = TfidfVectorizer()
    combined_tfidf = vectorizer.fit_transform(combined_data)
    
    # Split the data back
    X_train = combined_tfidf[:len(train_data)]
    X_test = combined_tfidf[len(train_data):]
    
    # Extract labels (assuming first element in line is label)
    y_train = [int(line.split()[0]) for line in train_data]
    y_test = [int(line.split()[0]) for line in test_data]
    
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = preprocess_data(train_data_path, test_data_path)

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

# Best parameters and accuracy
print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)