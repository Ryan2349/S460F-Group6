import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# define input file paths
trainingInput = './data/webkb-train-stemmed.txt'
testInput = './data/webkb-test-stemmed.txt'
inputFiles = [trainingInput, testInput]

# define new class labels
classMap = {
  "student": 1,
  "faculty": 2,
  "course": 3,
  "project": 4
}

# combine training data and test data into single corpus
allDocuments = []
fileDocumentCount = []
for file in inputFiles:
  with open(file, 'r') as inputStream:
    documents = inputStream.read().split('\n')
    allDocuments += documents
    fileDocumentCount.append(len(documents))

# separate class label from each document
documentClasses = []
corpus = []
for document in allDocuments:
  arr = document.split()
  documentClass = arr.pop(0).strip()
  documentClasses.append(classMap[documentClass])
  corpus.append(' '.join(arr))

# create tf-idf matrix from the content
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# Apply Linear Discriminant Analysis (LDA) for dimensionality reduction
lda = LinearDiscriminantAnalysis(n_components=3)  # LDA reduces to 3 components (n_classes - 1 = 3)
reduced_matrix = lda.fit_transform(tfidf_matrix.toarray(), documentClasses)

# Visualize the data using Matplotlib

# Check if we are doing a 2D or 3D plot
if reduced_matrix.shape[1] == 2:
    # 2D scatter plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], c=documentClasses, cmap='viridis', s=50)
    plt.title('LDA-Reduced Data (2D)')
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.colorbar(scatter, label='Class Labels')
    plt.show()

elif reduced_matrix.shape[1] == 3:
    # 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], reduced_matrix[:, 2], c=documentClasses, cmap='viridis', s=50)
    ax.set_title('LDA-Reduced Data (3D)')
    ax.set_xlabel('LD1')
    ax.set_ylabel('LD2')
    ax.set_zlabel('LD3')
    fig.colorbar(scatter, ax=ax, label='Class Labels')
    plt.show()