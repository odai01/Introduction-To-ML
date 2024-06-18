from sklearn.datasets import fetch_openml
import numpy as np 
import numpy.random
import matplotlib.pyplot as plt

mnist = fetch_openml("mnist_784", as_frame=False)
data = mnist["data"]
labels = mnist["target"]
idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

#   question 2a
def kNNAlg(TrainImages,Labels,QueryImage,k):
    distances=np.linalg.norm(TrainImages-QueryImage,axis=1)
    SortedIndices = np.argsort(distances)
    kNeighborsIndices = SortedIndices[:k]
    kNeighborsLabels = Labels[kNeighborsIndices].astype(int)
    predict = np.argmax(np.bincount(kNeighborsLabels))
    return predict

#   question 2b
new_train_initial=train[:1000]
new_train_labels_initial=train_labels[:1000]
def CalculateAccuracy(K,n):
    new_train=new_train_initial
    new_train_labels=new_train_labels_initial
    if(n!=1000):
        new_train=train[:n]
        new_train_labels=train_labels[:n]
    PredictedLabels = np.array([kNNAlg(new_train, new_train_labels, TestImage, K) for TestImage in test])
    correct_predictions = np.sum(PredictedLabels == test_labels.astype(int))
    accuracy=correct_predictions/len(test_labels)
    return accuracy
print("Question 2b:The accuracy of the prediction is(k=10):",CalculateAccuracy(10,1000))
#   question 2c
Kvalues=np.arange(1,101)
Acc=np.array([CalculateAccuracy(x,1000) for x in Kvalues])
plt.figure()
plt.scatter(Kvalues, Acc,color='red' ,label='Accuracy',marker='*')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#   question 2d
nvalues=np.arange(100,5001,100)
Acc=np.array([CalculateAccuracy(1,a) for a in nvalues])
plt.figure()
plt.scatter(nvalues, Acc,color='navy' ,label='Accuracy',marker='*')
plt.xlabel('n')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


