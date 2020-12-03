# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The information in the dataset is about a marketing campaign of a bank and, each row corresponds to one contact made with a client by means of a phonecall. The goal is to predict if the client will subscribe a term deposit. Namely, we seek to predict the column 'y' in the dataset.

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**
The best model was given by the AutoML model and turned out to be Voting Ensemble, with an accuracy of 0.91645. This algorithm was fed with XGBoostClassifier, LightGBM and SGD algorithms. 

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
First of all, the dataset is cleaned. This is done by converting categorical dta to Dummy data or binary data, depending on the case. Afterwards, the data is split into training and testing set in a proportion of 20% (test set) and 80% (training set). 
Then, an SKLearn estimator is created and hyperparameter tunning is performed by means of the Azure Hyperdrive tool. The hyperparameter tunning is made for a logistic regression algorithm. Therefore, the hyperparameters to be tunned are 'C' (inverse of regularization constant) and 'max_iter' which is the maximum number of iterations the algorithms can take to coverge.
Each model is scored and, finally the hyperparameter tunning best model is registered and saved.

The Random parameter sampler is the one which takes the least amount of compute resources. On the other hando, the bandit policy prevents the algorithm to keep evaluating models when acurracy is falling outside an specified range.

## AutoML
The model was Voting Ensemble, which uses as parameters other algorithms itself. In this run it used algorithms of XGBoostClassifier, ExtremeRandomTrees and SGD.

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
The metric used for comparissons in both parts of the project was accuracy. Hyperdrive hyperparameter tunning lead us to an accuracy of 0.8915 with parameters given by C=1 and max_iter = 100, while the AutoML module gives us an accuracy of 0.91645. I think hyperdrive should perform better with a best choose of the model whose hyperparameters are supposed to be tunned. In this case, we can see that logistic regression is not the best model to choose.

## Future work
Now that we know that the best model for predictions in this dataset is Ensemble voting, we could use hyperdrive to compare hyperparameters in VotigEnsemble model, instead of Logistic regression. This might lead us to an even better performance. 

## Compute target deletion

