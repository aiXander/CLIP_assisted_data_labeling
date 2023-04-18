import numpy as np
np.random.seed(42)

# from Karpathy:
# https://twitter.com/karpathy/status/1647025230546886658

dim = 768
n = 1000

embeddings = np.random.randn(n, dim) # 1000 documents, 1536-dimensional embeddings
embeddings = embeddings / np.sqrt((embeddings**2).sum(1, keepdims=True)) # L2 normalize the rows, as is common

query = np.random.randn(dim) # the query vector
query = query / np.sqrt((query**2).sum())

# Tired: use kNN
similarities = embeddings.dot(query)
sorted_ix = np.argsort(-similarities)
print("top 10 results:")
for k in sorted_ix[:10]:
    print(f"row {k}, similarity {similarities[k]}")


# Wired: use an SVM
from sklearn import svm

# create the "Dataset"
x = np.concatenate([query[None,...], embeddings]) # x is (1001, 1536) array, with query now as the first row
y = np.zeros(n+1)
y[0] = 1 # we have a single positive example, mark it as such

# train SVM
# docs: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.1)
clf.fit(x, y) # train

# infer on whatever data you wish, e.g. the original data
similarities = clf.decision_function(x)
sorted_ix = np.argsort(-similarities)
print("\nSVM:")
print("top 10 results:")
for k in sorted_ix[:10]:
  print(f"row {k}, similarity {similarities[k]}")