from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
import joblib  # For saving the model

# Path to your training data file
train_data_path = './data/libsvm/train.txt'

# Function to load pre-vectorized training data
def load_train_data(train_file):
    # Load the data in libSVM format
    X_train, y_train = load_svmlight_file(train_file)
    
    return X_train, y_train

# Load the pre-vectorized training data
X_train, y_train = load_train_data(train_data_path)

# Initialize SVM with best parameters
C_param = 1
gamma_param = 1
kernel_param = 'linear'
svc = SVC(C=C_param, gamma=gamma_param, kernel=kernel_param)

# Fit the model on the entire training set
svc.fit(X_train, y_train)

# Create the model file name with parameters
model_filename = f'svm_model_c{C_param}_g{gamma_param}_{kernel_param}.pkl'

# Save the best model with the dynamic file name
joblib.dump(svc, model_filename)

print(f"Model saved as {model_filename}")
