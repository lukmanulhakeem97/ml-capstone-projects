# ML-capstone-projects
My machine Learning capstone projects

## Project 1: Bank marketing 
Dataset: 'http://archive.ics.uci.edu/ml/datasets/Bank+Marketing#'.
### Project Analysis:
In this project I tried to build a decision making ML model on a classification problem that can predict or decide whether a bank customer subscribe for term deposit plan or not. For this I am using Portuguese Bank marketing Dataset. It is found that XGBClassifier algorithm is the best suit for this problem. Trained model can predict with 65% recall score and 88% of accuracy. I have found this score as best performance metrics scores compared to other available works in internet sources on this dataset.
### Summary:
I have gone through the main steps of ML pipeline. After several analysis, pass raw data through Label Encoder object to convert categorical to numerical and followed by splitting data into train and test. Then for cross checking, cross validation is done on several sklearn algorithms with imblearn pipeline by using train set. Within pipeline steps like balancing and scaling on data carried on using SMOTE oversampling technique and StandardScaler. By using pipeline for cross validation in this way, we can overcome 'data leakage' problem. Cross validation is done by choosing several metrics like 'recall_score', 'f1_score' and 'roc_auc'.

Cross check results shows XGBClassifier has overall better score in recall, roc_auc and f1. So I chose that algorithm and perform hyperparameter tuning. But hyperparameter tuning shows default hyperparameter have decent recall and f1 scores comparatively, so I choose default XGBoost model for testing. 

On testing model has
- recall score (of 1): 65%
- precision score (of 1): 51%
- f1 score: 57%
- accuracy score: 88%
### Files:
-	‘main.ipynb’: main notebook file where data modeling  is done.
-	‘s_scaler.pkl’: used to scale data when using the model externally or during deployment. Fitted on train data using StandardScaler method..
-	'bank-full.csv': main data set.
-	‘SWEETVIZ_REPORT.html’: shows features univariate analysis by ‘sweetviz’ library.
-	'labelencoder_reference.csv': dataframe contains values of categorical features, can be used as a reference to convert new test data during deployment.
-	'processed_data1.csv': semi preprocessed data.
-	'train_data.csv': train set split from 'bank-full.csv' for training.
-	'test_data.csv': test set split from 'bank-full.csv' for final evaluation.
-	‘xgbc_model.pkl': saved final xgboost model.



## Project 2: Handwritten Digits classification
More on dataset: 'http://yann.lecun.com/exdb/mnist/' 
### Project Analysis:
In this project I am trying to build a ML model on a multiclass image classification problem. Model’s objective is to predict which digit is actually written in the given input image sample. Model will try to classify given input image into any of the 10 single digit classes ranging from 0 to 9. I have found that deep learning approach is good for this problem, and more over it is computer vision related Convolutional neural network address best. By using Keras CNN, model got maximum 99.28% and 98.93% accuracy during training and cross validation respectively, for test data model achieved accuracy of 99.09%. Thus hopefully, I build  a good generalized model.
### Summary:
I used MNIST Handwritten image dataset. It consists of total 70,000 image instances, 60,000 for training and 10,000 for testing the model. Input is the image pixel intensity measure and output is the class of digit which it belongs. Each instance is a gray scale image of 28x28 pixel size. Dataset is almost balanced, though I applied smote oversampling to make it exactly balanced and to see whether performance increase or not. But I can’t find much difference in performance even after applying smote.

For modelling, prepared data in two ways; first for sklearn and ANN algorithms reshaped each input of 28x28 instances into 784 features, and then for CNN reshaped each input into 28x28x1 and each output into one-hot vectors of 1x10 shape.

I tried modelling with both sklearn and deep learning algorithms. Among sklearn algorithms, Support Vector Classifier performed well have accuracy of 97%. In deep learning CNN with two convolutional layer and single Dense ANN layer of 10 units with 0.25 dropout outperform other models, So selected it as final model. It has 99.28%, 98.93% and 99.09% as train, dev and test accuracy respectively.

On test set model has
- Accuracy(in percentage)
   - Train : 99.28%
   - Validation : 98.93%
   - Test : 99.09%

- Loss
   - Train : 0.0205
   - Validation : 0.0369
   - Test : 0.0277
### Files:
-	‘main.ipynb’ is the main notebook file where modelling done.
-	‘model_ann.h5’ saves best trained artificial neural network parameters. 
-	‘model_cnn.h5’ saves best trained convolutional neural network parameters.



## Project 3: Flight Price Prediction
Dataset can be available at https://www.kaggle.com/nikhilmittal/flight-fare-prediction-mh.
### Project Analysis:
In this project I am trying to build a machine learning regression model for predicting Indian Domestic Flight Price. Data set used has 10683 records and 11 features including dependent ‘Price’ feature. I have found sklearn’s ensemble algorithm ‘ExtraTreesRegressor’ is most suited for our problem. On this algorithm we build a model that able to achieve adjusted R2-score of 0.971 on train data and 0.920 on test data. 
### Summary:
Since this data set has several features with unstructured format which includes date, time and routes, I pass data through several pre processing and feature engineering steps before feeding into cross checking. There are some missing values and outliers present for some features.

Half of the features are categorical, with several trail-error steps found that using of label-encoding and one-hot encoding on perticular feature enhance performance of model in cross checking step. I use XGBoost, LGBM and several sklearn algorithms to cross check. Among all, found ensemble algorithm ‘ExtraTreesRegressor’ perform better compared to other.

On cross validation in cross checking step ‘ExtraTreesRegressor’ model shows R2-score of 0.906 with standard deviation of 0.022, which is better and tells model is more consistent in prediction compared to others. After hyper parameter tuning of model, model achieved adjusted R2-score of 0.971 and RMSE of 773.34 on train data.

On test set model has
- Adjusted R2-score(in percentage): 92.04
- RMSE: 1320.88
### Files:
-	‘main.ipynb’: main notebook file where modeling done.
-	‘Flight_Fare.xlsx’: data set file.
-	‘Route_labelencoder_ref.csv’: can be used as reference for converting 'Route' categorical values while deploying or external use.
-	‘etr_model.gzip’: this file used to saved final ExtraTreesRegressor model, can be used while deploying.
-	'SWEETVIZ_REPORT.html': analysis of data using sweetviz.
