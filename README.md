# NPU-RESISC45-Classification

### Pre-requirement
In order to use this repo, kindly upload the local dataset, (NPU-RESISC45) into [Google Colab](https://colab.research.google.com/).

In-order to perform classification, it make uses of OpenCV SIFT (Patent by Google), if unable to perform SIFT(Errors), kindly downgrade OpenCV to 3.4.2 and below.
[Check here for more information on solving SIFT patent-ed by Google on OpenCV.](https://stackoverflow.com/questions/52305578/sift-cv2-xfeatures2d-sift-create-not-working-even-though-have-contrib-instal)


### Google Colab Files
In this repository, there are multiple steps involved to perform Classification on NPU-RESISC45, in-order to perform classification, kindly run them in-order:
1. Codeblock_Creation.ipynb - This will create a code block that is pickled with SIFT
2. Classifcation.ipynb - This will perform classification using the codeblock created on Step (1), then it will performs classification using SVM, Random Forest.

### Accuracy
There are multiple classification model used, which turns into multiple results.

The baseline used are SIFT feature extracted, then uses Keras SVM, Keras Random Forest.
