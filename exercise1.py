# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn import svm
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import numpy as np
from sklearn import random_projection

#datasets
iris = datasets.load_iris()
digits = datasets.load_digits()


print(digits.data)
print("")

print(digits.target)
print("\n\n\n")


#learning and predicting
clf = svm.SVC(gamma=0.001, C=100.)
print(clf.fit(digits.data[:-1], digits.target[:-1]))
print("")
print(clf.predict(digits.data[-1:]))
print("\n\n\n")


#model persistence(save a model)
clf=svm.SVC()
iris=datasets.load_iris()
X, y = iris.data, iris.target
print(clf.fit(X, y))
print("")


s=pickle.dumps(clf)
clf2=pickle.loads(s)
print(clf2.predict(X[0:1]))
print(y[0])
print("\n\n\n")

"""
joblib의 dump method와 load method를 이용하여
데이터를 저장하고 불러올 수도 있음
(filename.pkl 형태로 저장하고 불러)
"""




#convention(cast to float64)
rng=np.random.RandomState(0)
X=rng.rand(10,2000)
X=np.array(X,dtype='float32')
print(X.dtype)

transformer=random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print(X_new.dtype)
print("")



"""
처음 predict은 integer array 반환,
(iris.target이 integer array)
두 번째 predict은 string array 반환,
(iris.target_names가 string array이기 때문)
"""

iris=datasets.load_iris()
clf=SVC()
print(clf.fit(iris.data,iris.target))
print(list(clf.predict(iris.data[:3])))
print("")
print(clf.fit(iris.data, iris.target_names[iris.target]))
print(list(clf.predict(iris.data[:3])))


#refitting and updating parameters(rbf -> linear - rbf)
rng=np.random.RandomState(0)
X=rng.rand(100,10)
y=rng.binomial(1,0.5,100)
X_test=rng.rand(5,10)

clf=SVC()
print(clf.set_params(kernel='linear').fit(X,y))
print(clf.predict(X_test))
print("")
print(clf.set_params(kernel='rbf').fit(X,y))
print(clf.predict(X_test))


#multiclass, multilabel fitting(dependent on the format of the target data fit upon)
X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]

classif = OneVsRestClassifier(estimator=SVC(random_state=0))
print(classif.fit(X, y).predict(X))
print("")


"""
cf)
2-dimensional array -> LabelBinarizer 이용
y = LabelBinarizer().fit_transform(y)
"""

y=[[0,1],[0,2],[1,3],[0,2,3],[2,4]]
y=MultiLabelBinarizer().fit_transform(y)
print(classif.fit(X,y).predict(X))