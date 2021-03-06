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
The model was Voting Ensemble, which uses as parameters other algorithms itself. In this run it used algorithms of XGBoostClassifier, ExtremeRandomTrees and SGD. Other parameters obtained from the autoML are the following:
alpha = 0.001
Class_weight = 'balanced'
eta = 0.001
fit_intercept = True
l1_ratio =0.8367
learning_rate = 'constant'
loss = 'modified huber'
max_iter = 1000
n_jobs = 1
power_t = 0.222
random_state = None

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
The metric used for comparissons in both parts of the project was accuracy. Hyperdrive hyperparameter tunning lead us to an accuracy of 0.8915 with parameters given by C=1 and max_iter = 100, while the AutoML module gives us an accuracy of 0.91645. While Hyperdrive uses a logistic regression algorithm, autoML end up with a best run where the performed algorithm is VOting Ensemble, where three different algorithms are combined and the labeling is made by choosing the majority of labels between the three of them. Regarding timing, voting ensemble model took 1 minute and 11 seconds, while the hyperdrive model took 55 seconds. However, the timing has variations among different runs. I think hyperdrive should perform better with a best choose of the model whose hyperparameters are supposed to be tunned. In this case, we can see that logistic regression is not the best model to choose.

## Future work
Now that we know that the best model for predictions in this dataset is Ensemble voting, we could use hyperdrive to compare hyperparameters in VotigEnsemble model, instead of Logistic regression. This might lead us to an even better performance. Other significant improvement could be choosing a new set for parameter sampler. For instance, Hyperdrive gave us C= 0.001 from the set [0.001,0.01,0.1,1,10,100]. We could chose a new set so that the given parameter C=0.001 is more central. FOr example, we could try the set [0.00001,0.0001,0.001,0.01,0.1,1]. The same for the parameter 'max_iter'. The sample set was [8,100,120] and the given parameter was 8. We could take a set like [4,6,8,10,12,14] instead. 

## Proof of compute target clean up 
<img width="524" alt="Cluster deletion3" src="https://user-images.githubusercontent.com/25666364/100962772-fe69ab00-34f2-11eb-95c7-5cee77020d3d.PNG">
