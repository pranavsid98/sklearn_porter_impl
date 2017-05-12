from sklearn.datasets import load_iris
from sklearn import svm
from sklearn_porter import Porter
from sklearn.externals import joblib

#Loading data and training SVM
iris_data = load_iris()
X, y = iris_data.data, iris_data.target
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(X, y)

#Exporting to JS:
porter = Porter(clf, language='js')
output = porter.export()

joblib.dump(clf, 'model.pkl')

os.system("python -m sklearn_porter -i model.pkl -l js")
