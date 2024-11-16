import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# define input file paths
trainingInput = './data/webkb-train-stemmed.txt'
testInput = './data/webkb-test-stemmed.txt'
inputFiles = [trainingInput, testInput]

# define libsvm-formatted file paths
libsvmTrainingPath = './data/libsvm/train(LDA).txt'
libsvmTestPath = './data/libsvm/test(LDA).txt'

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
lda = LinearDiscriminantAnalysis(n_components=3)  # We have 4 classes, so n_components = 3 (n_classes - 1)
reduced_matrix = lda.fit_transform(tfidf_matrix.toarray(), documentClasses)

# for every document, construct string of the form (label idx:value idx:value idx:value) 
formattedDocuments = []
for idx, row in enumerate(reduced_matrix):
  result = str(documentClasses[idx])
  for idx, value in enumerate(row):
    if value > 0:
      result += ' ' + str(idx) + ':' + str(value)
  formattedDocuments.append(result)

# split formatted documents back into training / test data
trainingLength = fileDocumentCount[0]
testLength = fileDocumentCount[1]
formattedTrainingData = formattedDocuments[0:trainingLength]
formattedTestData = formattedDocuments[trainingLength:]

# write libsvm formatted training data to output file
directory = os.path.dirname(libsvmTrainingPath)
if not os.path.exists(directory):
    os.makedirs(directory)
with open(libsvmTrainingPath, 'w') as trainingStream:
    trainingStream.write('\n'.join(formattedTrainingData))

# write libsvm formatted test data to output file
directory = os.path.dirname(libsvmTestPath)
if not os.path.exists(directory):
    os.makedirs(directory)
with open(libsvmTestPath, 'w') as testStream:
    testStream.write('\n'.join(formattedTestData))