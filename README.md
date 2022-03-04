# FL_LogisticsRegression
This is a prototype of a simple federated learning system using logistic regression as the main ML model at clients and server sides. Aggregation is done with weighted averaging. This contains 3 parts:

## data
ETL data from the ATB hackathon:
select the relevant features

## clients
Data from data model is used to simulate 10 local clients. Each client corresponds to a geographical area

## logistic_regression
This module implement the logistic regression from crash

https://developer.ibm.com/articles/implementing-logistic-regression-from-scratch-in-python/


## Aggregation
This module performs weighted averaging at the server side to generte the global model

## global_prediction
This module performs prediction at the local clients using the global model

## local_train_predict
This module performs training and predicting at the local clients 







