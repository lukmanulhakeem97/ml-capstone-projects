## Project: Handwritten Digits classification
More on dataset: 'http://yann.lecun.com/exdb/mnist/' 
### Project Analysis:
In this project I am trying to build a ML model on a multiclass image classification problem. Model’s objective is to predict which digit is actually written in the given input image sample. Model will try to classify given input image into any of the 10 single digit classes ranging from 0 to 9. I have found that deep learning approach is good for this problem, and more over it is computer vision related Convolutional neural network address best. By using Keras CNN, model got maximum 99.28% and 98.93% accuracy during training and cross validation respectively, for test data model achieved accuracy of 99.09%. Thus hopefully, I build  a good generalized model.
### Summary:
I used MNIST Handwritten image dataset. It consists of total 70,000 image instances, 60,000 for training and 10,000 for testing the model. Input is the image pixel intensity measure and output is the class of digit which it belongs. Each instance is a gray scale image of 28x28 pixel size. Dataset is almost balanced, though I applied smote oversampling to make it exactly balanced and to see whether performance increase or not. But I can’t find much difference in performance even after applying smote.

For modelling, prepared data in two ways; first for sklearn and ANN algorithms reshaped each input of 28x28 instances into 784 features, and then for CNN reshaped each input into 28x28x1 and each output into one-hot vectors of 1x10 shape.

I tried modelling with both sklearn and deep learning algorithms. Among sklearn algorithms, Support Vector Classifier performed well have accuracy of 97%. In deep learning CNN with two convolutional layer and single Dense ANN layer of 10 units with 0.25 dropout outperform other models, So selected it as final model. It has 99.28%, 98.93% and 99.09% as train, dev and test accuracy respectively.

On test set model has
- Accuracy(in percentage):  99.09
- Loss: 0.0277
### Files:
-	‘main.ipynb’ is the main notebook file where modelling done.
-	‘model_ann.h5’ saves best trained artificial neural network parameters. 
-	‘model_cnn.h5’ saves best trained convolutional neural network parameters.
