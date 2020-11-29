# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The information in the dataset is about marketing campaign of a bank, and, each row corresponds to one contact made with a client by means of a phonecall. The goal is to predict if the client will subscribe a term deposit. Namely, we seek to predict the column 'y' in the dataset

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**
The best model was given by the AutoML model and turned out to be Voting Ensemble, with an accuracy of 0.91645. This algorithm was fed with XGBoostClassifier, LightGBM and SGD algorithms. 

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
First of all, the dataset is cleaned. This is done by converting categorical dta to dummy data and 1 or 0 data. Afterwards, the data is split into training and testing set in a proportion of 20% (test set) and 80% (training set). 
Then, an SKLearn estimator and then the hyperparameter tunning is performed by means of the Azure Hyperdrive tool. The hyperparameter tunning is made for a logistic regression algorithm. Therefore, the hyperparameters to be tunned are 'C' (inverse of regularization constant) and 'max_iter' which is the maximum number of iterations the algorithms can take to coverge.
Each model is scored and after the hyperparameter tunning the best model is registered and saved.

**What are the benefits of the parameter sampler you chose?**
The random sample is the one which takes the least amount of compute resources.


**What are the benefits of the early stopping policy you chose?**
The bandit policy prevents the algorithm to keep evaluating models when acurracy is falling outside an specified range.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**
The model was Voting Ensemble which uses as parameters other algorithms itself. In this run it used algorithms of XGBoostClassifier, ExtremeRandomTrees and SGD.

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
