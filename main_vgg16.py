from vgg16_feature_extractor import get_features
from keras import models
from keras import layers
from keras import optimizers

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier


def print_scores(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='macro'))
    print("Recall:", recall_score(y_true, y_pred, average='macro'))
    print("F1-Score:", f1_score(y_true, y_pred, average='macro'))


train_features, train_labels, validation_features, validation_labels = get_features()
batch_size= 32

############################################################################
# Feed forward Neural Network
############################################################################
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(12, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=1e-3),
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(train_features,
                    train_labels,
                    epochs=50,
                    batch_size=batch_size,
                    validation_data=(validation_features,validation_labels))

pred = model.predict(validation_features)
print_scores(validation_labels.argmax(axis=1), pred)

############################################################################
# Random Forest
############################################################################
num_trees = [1, 10, 100, 1000]
for tree in num_trees:
    rf_classifier = RandomForestClassifier(n_estimators=tree)
    rf_classifier.fit(train_features, train_labels.argmax(axis=1))
    pred = rf_classifier.predict(validation_features)
    print("\nNumber of Trees:", tree)
    print_scores(validation_labels.argmax(axis=1), pred)

############################################################################
# Support Vector Machine
############################################################################
# SVM with Radial Basic Function as kernel
sv_classifier = SVC(gamma='auto', kernel='rbf')
sv_classifier.fit(train_features, train_labels.argmax(axis=1))
pred = sv_classifier.predict(validation_features)
print("\nSupport Vector Machine with rbf kernel")
print_scores(validation_labels.argmax(axis=1), pred)

# SVM with polynomical decision boundaries at different degree of polynomial
degrees = [1, 3, 5, 10]
print("\nSupport Vector Machine with polynomial kernel")
for d in degrees:
    sv_classifier = SVC(gamma='auto', kernel='poly', degree=d)
    sv_classifier.fit(train_features, train_labels.argmax(axis=1))
    pred = sv_classifier.predict(validation_features)
    print("\nDegree of Polynomial:", d)
    print_scores(validation_labels.argmax(axis=1), pred)

############################################################################
# K-Nearest Neighbour
############################################################################
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_features, train_labels.argmax(axis=1))
pred = knn.predict(validation_features)
print("\nK-Nearest Neighbour")
print_scores(validation_labels.argmax(axis=1), pred)