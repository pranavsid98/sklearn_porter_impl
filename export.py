from sklearn.datasets import load_iris
from sklearn import svm
from sklearn_porter import Porter
import pickle, os
import numpy

def dense_to_libsvm(x, dims):
    len_row = dims[1]
    tx = x
    node = {}
    for i in range(dims[0]):
        node[i] = {}
        node[i]['values'] = tx
        node[i]['dim'] = int(len_row)
        node[i]['ind'] = i
        tx += len_row
    return node

#Loading data and training SVM
iris_data = load_iris()
X, y = iris_data.data, iris_data.target
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(X, y)

#Exporting to dict
classifier_data = clf.__dict__

#Mapping SVM types to labels
mapping_svm_types_index = {
    'c_svc' : 0,
    'nu_svc' : 1,
    'one_class' : 2,
    'epsilon_svr' : 3,
    'nu_svr' : 4
}

#Mapping kernel types to labels
mapping_kernel_types_index = {
    'linear' : 0,
    'poly' : 1,
    'rbf' : 2,
    'sigmoid' : 3,
    'precomputed' : 4
}

#Prediction logic
y_pred = numpy.asarray([[1,2,3,4]])

svm_type = mapping_svm_types_index[classifier_data['_impl']]
kernel_type = mapping_kernel_types_index[classifier_data['kernel']]
degree = classifier_data['degree']
gamma = classifier_data['gamma']
coef0 = classifier_data['coef0']
support = classifier_data['support_']
sv = classifier_data['support_vectors_']
nsv = classifier_data['n_support_']
sv_coef = classifier_data['_dual_coef_']
intercept = classifier_data['_intercept_']
probA = classifier_data['probA_']
probB = classifier_data['probB_']
class_weight = classifier_data['class_weight_']
cache_size = classifier_data['cache_size']

class_weight_label = numpy.arange(class_weight.shape[0], dtype=numpy.int32)

#Set prediction time parameters
param = {}
param['svm_type'] = svm_type
param['kernel_type'] = kernel_type
param['degree'] = degree
param['coef0'] = coef0
param['cache_size'] = cache_size
param['gamma'] = gamma
param['C'] = classifier_data['C']
param['epsilon'] = classifier_data['epsilon']
param['max_iter'] = classifier_data['max_iter']
param['nu'] = classifier_data['nu']
param['shrinking'] = classifier_data['shrinking']
param['tol'] = classifier_data['tol']
param['random_seed'] = -1
param['probability'] = 0
param['nr_weight'] = class_weight.shape[0] 
param['weight_label'] = class_weight_label
param['weight'] = class_weight

#Set model for prediction
model = {}
dsv_coef = sv_coef
m = nsv.shape[0] * (nsv.shape[0] - 1)/2
model['nr_class'] = nsv.shape[0]
model['param'] =  param
model['label'] = {}
model['sv_coef'] = {}
model['rho'] = {}
model['l'] = support.shape[0]

#This is not for precomputed kernel type.
model['sv'] = dense_to_libsvm(sv, sv.shape)

if param['svm_type'] < 2:
    model['nsv'] = nsv
    for i in range(model['nr_class']):
        model['label'][i] = i

for i in range(model['nr_class']-1):
    model['sv_coef'][i] = dsv_coef + i*(model['l'])

for i in range(m):
    model['rho'][i] = -((intercept)[i])

if param['probability']:
    model['probA'] = probA.data
    model['probB'] = probB.data

else:
    model['probA'] = None
    model['probB'] = None

dec_values = numpy.empty(y_pred.shape[0])
# Predictions
predict_nodes = dense_to_libsvm(y_pred, y_pred.shape)
for i in range(y_pred.shape[0]):
    #svm_predict_values(model, predict_nodes[i], dec_values[i])

