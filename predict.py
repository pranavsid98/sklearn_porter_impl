import json_ops, numpy

#Prediction logic
y_pred = numpy.asarray([4,3,2,1])

fp = open("classifier_data.json","r")
cached_classifier_data = json_ops.decode(fp)

n_svs = cached_classifier_data['n_support_']
svs = cached_classifier_data['support_vectors_']
coeffs = cached_classifier_data['_dual_coef_']
inters = cached_classifier_data['_intercept_']
classes = cached_classifier_data['classes_']

kernels = {}
for i in range(len(svs)):
    kernel = 0.
    for j in range(cached_classifier_data['shape_fit_'][1]):
        kernel += pow(svs[i][j] - y_pred[j],2)
    kernels[i] = numpy.exp(-0.001 * kernel)

starts = {}
for i in range(len(n_svs)):
    if i != 0:
        start = 0
        for j in range(i):
            start += n_svs[j]
        starts[i] = start
    else:
        starts[0] = 0

ends = {}
for i in range(len(n_svs)):
    ends[i] = n_svs[i] + starts[i]

decisions = {}
d = 0
l = len(n_svs)
for i in range(l):
    for j in range(i+1,l):
        tmp = 0.
        for k in range(starts[j],ends[j]):
            tmp += kernels[k] * coeffs[i][k]
        for k in range(starts[i],ends[i]):
            tmp += kernels[k] * coeffs[j-1][k]
        decisions[d] = tmp + inters[d]
        d += 1

votes = {}
d = 0
l = len(n_svs)
for i in range(l):
    for j in range(i+1,l):
        votes[d] = i if decisions[d] > 0 else j
        d += 1

amounts = {}
for i in range(len(n_svs)):
    amounts[i] = 0

for i in range(len(n_svs)):
    amounts[votes[i]] += 1

class_val = -1
class_idx = -1
l = len(n_svs)
for i in range(l):
    if amounts[i] > class_val:
        class_val = amounts[i]
        class_idx = i

print class_idx
