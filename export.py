from sklearn.datasets import load_iris
from sklearn import svm
from sklearn_porter import Porter
from sklearn.externals import joblib
import pickle, os
import numpy, json_ops

#Loading data and training SVM
iris_data = load_iris()
X, y = iris_data.data, iris_data.target
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(X, y)

#Exporting to dict
classifier_data = clf.__dict__

fp = open("classifier_data.json","w")
json_content = json_ops.encode(classifier_data, fp)
