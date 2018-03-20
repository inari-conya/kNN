#python version 2.7.14

import kNN_test
from numpy import array

group,labels = kNN_test.createDataSet()
#group
#labels

kNN_test.classify0([0,0], group, labels, 3)

datingDataMat,datingLabels = kNN_test.file2matrix('datingTestSet.txt')
#datingDataMat
datingLabels[0:20]

import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(map(int,datingLabels)), 15.0*array(map(int,datingLabels)))
plt.show()

fig.add_subplot(111)
idx1=array(datingLabels)==1
idx2=array(datingLabels)==2
idx3=array(datingLabels)==3

plt.scatter(datingDataMat[idx1,0],datingDataMat[idx1,1],marker='+',s=50,c='b',label="Don't Like")
plt.scatter(datingDataMat[idx2,0],datingDataMat[idx2,1],marker='o',s=30,c='m',label="Small Doses")
plt.scatter(datingDataMat[idx3,0],datingDataMat[idx3,1],marker='x',s=80,c='c',label="Large Doses")

plt.legend(loc='upper left')
plt.show()

normMat, rangers, minVals = kNN_test.autoNorm(datingDataMat)

kNN_test.datingClassTest()

kNN_test.classifyPerson()
