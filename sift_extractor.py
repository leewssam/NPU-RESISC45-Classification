import cv2
import numpy as np
import glob
import pickle

from matplotlib import pyplot as plt
from scipy.cluster.vq import vq

############################################################################
# Obtain training files
############################################################################

classes = []
for file in glob.glob("/content/drive/My Drive/VIP/NWPU-RESISC12/train/*"):
    classes.append(file)

root=classes[0][:48]
for i in range(len(classes)):
    classes[i] = classes[i][48:]
print(classes)

subclasses = []
for i in classes:
    temp_path = str(root)+str(i)+"/*"
    for file in glob.glob(temp_path):
        subclasses.append(file)

print(len(subclasses))

############################################################################
# Obtain testing files
############################################################################
test_classes = []
for file in glob.glob("/content/drive/My Drive/VIP/NWPU-RESISC12/test/*"):
    test_classes.append(file)

test_root = test_classes[0][:47]
for i in range(len(test_classes)):
    test_classes[i] = test_classes[i][47:]
print(test_classes)

# For all test folder, obtain everything and throw into a list.
test_subclasses = []
for i in test_classes:
    temp_path = str(test_root)+str(i)+"/*"
    for file in glob.glob(temp_path):
        test_subclasses.append(file)

print(len(test_subclasses))


############################################################################
# SIFT Extractor
############################################################################
def extractSIFTFeatures(path,th):
    imgs = []
    feat = []
    for i in path:
        img = cv2.imread(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=th)
        kps, des = sift.detectAndCompute(gray, None)

        imgs.append(img)
        feat.append((kps, des))     # list contains a tuple of two arrays -- keypoint, descriptor

    return (feat,imgs)

############################################################################
# Extracting keypoints and descriptors on Train and Test
############################################################################
th = 0.001
trainFeat,trainImg = extractSIFTFeatures(subclasses,th)

th = 0.001
testFeat,testImg = extractSIFTFeatures(test_subclasses,th)

############################################################################
# Codebook generation
############################################################################
loc, des = list(zip(*feat))
alldes = np.vstack(des)
print(alldes.shape)

import time
import datetime

k = 50
alldes = np.float32(alldes)
e0 = time.time()
print(datetime.datetime.now())
codebook, distortion = kmeans(alldes, k)
# code, distortion = vq(alldes, codebook)
e1 = time.time()
print("Time to build {}-cluster codebook from {} images: {} seconds".format(k,alldes.shape[0],e1-e0))

#To dump the codebook for further processing and avoid of loss (codebook take very long!)
pickle.dump(codebook, open("codebook_001.pkl", "wb") )
codebook_001 = pickle.load(open("/content/drive/My Drive/VIP/NWPU-RESISC12/codebook_001.pkl", "rb" ))


############################################################################
# BOVW
############################################################################
def generateBOVW(feat, codebook, nImages, k):
    bow = np.empty([nImages, k])
    for i in np.arange(nImages):
        code, distortion = vq(feat[i][1], codebook)
        bowhist = np.histogram(code, k, density=True)
        bow[i][:]=bowhist[0]

    return bow

############################################################################
# BOVW In action
############################################################################
k = 50
trainFeat = trainFeat
NumberOfTrain = len(subclasses)
X = generateBOVW(trainFeat, codebook_001, NumberOfTrain, k)

k = 50
testFeat = testFeat
NumberOfTest = len(test_subclasses)
X_test = generateBOVW(testFeat, codebook_001, NumberOfTest, k)

############################################################################
# Labelling train and test set
############################################################################
X_test_label =[]
for i in test_subclasses:
    X_test_label.append(i[47:])

for i in range(len(X_test_label)):
    X_test_label[i] = X_test_label[i].split('/')[0]

y = np.zeros(len(subclasses))
for i in range(len(classes)):
    y[i*550:(i+1)*550] = i

y_test = np.zeros(len(test_subclasses))
for i in range(len(y_test)):
    for x in range(len(classes)):
        if (X_test_label[i]==classes[x]):
            y_test[i]=x

############################################################################
# Classifier
############################################################################
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def print_scores(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='macro'))
    print("Recall:", recall_score(y_true, y_pred, average='macro'))
    print("F1-Score:", f1_score(y_true, y_pred, average='macro'))

# KNN
lin_clf = KNeighborsClassifier()
lin_clf.fit(X, y)
y_pred=lin_clf.predict(X_test)
print_scores(y_test,y_pred)

# Random Forests: Tree = 1
lin_clf = RandomForestClassifier(n_estimators =1)
lin_clf.fit(X, y)
y_pred=lin_clf.predict(X_test)
print_scores(y_test,y_pred)

# Random Forests: Tree = 10
lin_clf = RandomForestClassifier(n_estimators =10)
lin_clf.fit(X, y)
y_pred=lin_clf.predict(X_test)
print_scores(y_test,y_pred)

# Random Forests: Tree = 100
lin_clf = RandomForestClassifier(n_estimators =100)
lin_clf.fit(X, y)
y_pred=lin_clf.predict(X_test)
print_scores(y_test,y_pred)

# Random Forests: Tree = 1000
lin_clf = RandomForestClassifier(n_estimators =1000)
lin_clf.fit(X, y)
y_pred=lin_clf.predict(X_test)
print_scores(y_test,y_pred)

# SVC on RBF
lin_clf = SVC(gamma='auto')
lin_clf.fit(X, y)
y_pred=lin_clf.predict(X_test)
print_scores(y_test,y_pred)

# SVC on Poly with Degree: 1(Linear)
lin_clf = SVC(kernel='poly',degree =1,gamma='auto')
lin_clf.fit(X, y)
y_pred=lin_clf.predict(X_test)
print_scores(y_test,y_pred)

# SVC on Poly with Degree: 3(Cubic)
lin_clf = SVC(kernel='poly',degree =3,gamma='auto')
lin_clf.fit(X, y)
y_pred=lin_clf.predict(X_test)
print_scores(y_test,y_pred)

# SVC on Poly with Degree: 5
lin_clf = SVC(kernel='poly',degree =5,gamma='auto')
lin_clf.fit(X, y)
y_pred=lin_clf.predict(X_test)
print_scores(y_test,y_pred)

# SVC on Poly with Degree: 10
lin_clf = SVC(kernel='poly',degree =10,gamma='auto')
lin_clf.fit(X, y)
y_pred=lin_clf.predict(X_test)
print_scores(y_test,y_pred)
