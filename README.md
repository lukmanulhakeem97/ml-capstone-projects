# ml-capstone-projects
My capstone machine Learning projects

## Project 1: Bank marketing 
Dataset: 'http://archive.ics.uci.edu/ml/datasets/Bank+Marketing#'.
### Project Analysis:
In this project I tried to build a decision making ML model on a classification problem that can predict or decide whether a bank customer subscribe for term deposit plan or not. For this I am using Portuguese Bank marketing Dataset. It is found that XGBClassifier algorithm is the best suit for this problem. Trained model can predict with 65% recall score and 88% of accuracy. I have found this score as best performance metrics scores compared to other available works in internet sources on this dataset.
### Summary:
I have gone through the main steps of ML pipeline. After several analysis, pass raw data through Label Encoder object to convert categorical to numerical and followed by splitting data into train and test. Then for cross checking, cross validation is done on several sklearn algorithms with imblearn pipeline by using train set. Within pipeline steps like balancing and scaling on data carried on using SMOTE oversampling technique and StandardScaler. By using pipeline for cross validation in this way, we can overcome 'data leakage' problem. Cross validation is done by choosing several metrics like 'recall_score', 'f1_score' and 'roc_auc'.

Cross check results shows XGBClassifier has overall better score in recall, roc_auc and f1. So I chose that algorithm and perform hyperparameter tuning. But hyperparameter tuning shows default hyperparameter have decent recall and f1 scores comparatively, so I choose default XGBoost model for testing. 

On testing model has;
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
